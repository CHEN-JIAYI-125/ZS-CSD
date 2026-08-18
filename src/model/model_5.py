import logging

import torch
import torch.nn as nn
from transformers import AutoModel

from src.model.latent_reply_graph import LatentReplyGraph
from src.topology.topology_4 import (
    ContextTopologyEncoder,
    SpeakerHypergraphChannel,
    TargetTopologyEncoder,
)


def _parse_glan_branches(config):
    raw = getattr(config, 'glan_branches', 'global,local,struct,final')
    return [part.strip() for part in str(raw).split(',') if part.strip()]


def _graph_field(graph, name):
    if graph is None:
        return None
    if hasattr(graph, name):
        return getattr(graph, name)
    if isinstance(graph, dict) and name in graph:
        return graph[name]
    return None


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, query, keys, values):
        if query.dim() == 1:
            query = query.unsqueeze(0)
        q = self.query_proj(query)
        k = self.key_proj(keys)
        v = self.value_proj(values)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output.squeeze(0) if output.size(0) == 1 else output


class SSE(nn.Module):
    def __init__(self, hidden_dim=768, dropout=0.2):
        super().__init__()
        self.linear_intra = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention_intra = Attention(hidden_dim)
        self.speaker_hypergraph = SpeakerHypergraphChannel(hidden_dim, dropout=dropout)

    def forward(self, utterances, speakers):
        from src.common import map_sequence

        device = utterances.device
        speakers_mapped = map_sequence(speakers)
        speaker_ids = torch.tensor(speakers_mapped, device=device)
        inter_all = self.speaker_hypergraph(utterances, speaker_ids)

        v_lst = []
        last_speaker_idx = {}
        for i in range(len(speakers_mapped)):
            speaker_id = speakers_mapped[i]
            if speaker_id not in last_speaker_idx:
                v_lst.append(utterances[i])
            else:
                prev_idx = last_speaker_idx[speaker_id]
                vh_concat = torch.cat((v_lst[prev_idx], utterances[i]), dim=-1)
                q_intra = self.linear_intra(vh_concat)
                context = utterances[: i + 1]
                v_intra = self.attention_intra(q_intra, context, context)
                v_lst.append(v_intra + inter_all[i])
            last_speaker_idx[speaker_id] = i
        return torch.stack(v_lst)


class StanceKnowledgeAttention(nn.Module):
    def __init__(self, hidden_dim, bert_dim=768, dropout=0.1):
        super().__init__()
        self.know_norm = nn.LayerNorm(bert_dim)
        self.query_proj = nn.Linear(hidden_dim, bert_dim)
        self.key_proj = nn.Linear(bert_dim, bert_dim, bias=False)
        self.value_proj = nn.Linear(bert_dim, bert_dim, bias=False)
        self.out_proj = nn.Linear(bert_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = bert_dim ** -0.5

    def forward(self, h_sem, h_favor, h_against, h_neutral, valid_mask=None):
        stance_keys = torch.stack([h_favor, h_against, h_neutral], dim=0)
        keys = self.key_proj(self.know_norm(stance_keys))
        values = self.value_proj(self.know_norm(stance_keys))
        query = self.query_proj(h_sem.unsqueeze(0))
        scores = torch.matmul(query, keys.transpose(0, 1)) * self.scale
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask.view(1, -1), float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, values).squeeze(0)
        return self.out_proj(self.dropout(attended)), weights.squeeze(0)


class SITCL(nn.Module):
    """v3 backbone + PPED-aligned latent reply graph (posterior labels, prior sentences only)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_topology = bool(getattr(config, 'use_topology', 1))
        self.use_glan_topology = bool(getattr(config, 'use_glan_topology', 0))
        self.use_knowledge_stance_attention = bool(
            getattr(config, 'use_knowledge_stance_attention', 0)
            or getattr(config, 'use_knowledge_gate', 0)
        )
        self.use_latent_reply = bool(
            getattr(config, 'use_latent_reply', getattr(config, 'use_reply_posterior', 0))
        )
        self.reply_kl_lambda = float(getattr(config, 'reply_kl_lambda', 0.05))
        self.reply_pseudo_lambda = float(getattr(config, 'reply_pseudo_lambda', 0.02))
        self.reply_conf_threshold = float(getattr(config, 'reply_conf_threshold', 0.7))
        self.reply_kl_final_row_weight = float(getattr(config, 'reply_kl_final_row_weight', 1.0))
        self.reply_kl_full_weight = float(getattr(config, 'reply_kl_full_weight', 0.5))
        self.reply_beta_zero_epochs = int(getattr(config, 'reply_beta_zero_epochs', 2))
        self.reply_beta_mid_epochs = int(getattr(config, 'reply_beta_mid_epochs', 5))
        self.reply_beta_mid = float(getattr(config, 'reply_beta_mid', 0.05))
        self.reply_beta_max = float(getattr(config, 'reply_beta_max', 0.10))
        self.train_epoch = 0

        hidden = config.gru_hidden
        self.bert = AutoModel.from_pretrained(config.bert_dir)
        self.gru = nn.GRU(input_size=768, hidden_size=hidden, num_layers=config.gru_layer, batch_first=True)

        label_smoothing = float(getattr(config, 'label_smoothing', 0.05))
        class_weight = self._build_class_weights(config)
        self.criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=label_smoothing)

        dropout = float(getattr(config, 'sse_dropout', 0.2))
        self.SSE = SSE(hidden_dim=hidden, dropout=dropout)
        self.sem_norm = nn.LayerNorm(hidden)
        self.reply_fuse_norm = nn.LayerNorm(hidden)

        fusion_dim = hidden
        if self.use_topology:
            self.topology_encoder = ContextTopologyEncoder(
                hidden, dropout=float(getattr(config, 'topology_dropout', 0.2)),
            )
            self.topo_norm = nn.LayerNorm(hidden)
            fusion_dim += hidden
        else:
            self.topology_encoder = None
            self.topo_norm = None

        if self.use_glan_topology:
            self.glan_encoder = TargetTopologyEncoder(
                hidden,
                dropout=float(getattr(config, 'glan_dropout', 0.1)),
                local_window=int(getattr(config, 'topology_local_window', 3)),
                branches=_parse_glan_branches(config),
            )
            self.glan_norm = nn.LayerNorm(hidden)
            fusion_dim += hidden
        else:
            self.glan_encoder = None
            self.glan_norm = None

        if self.use_knowledge_stance_attention:
            self.stance_knowledge_attn = StanceKnowledgeAttention(
                hidden, bert_dim=768, dropout=float(getattr(config, 'knowledge_dropout', 0.1)),
            )
            fusion_dim += hidden
        else:
            self.stance_knowledge_attn = None

        self.fc = nn.Linear(fusion_dim, config.num_classes)

        if self.use_latent_reply:
            reply_dropout = float(getattr(config, 'reply_posterior_dropout', 0.1))
            reply_tau = float(getattr(config, 'reply_posterior_tau', 0.2))
            num_classes = int(getattr(config, 'num_classes', 3))
            self.latent_reply = LatentReplyGraph(
                hidden_dim=hidden,
                num_classes=num_classes,
                dropout=reply_dropout,
                tau=reply_tau,
            )
        else:
            self.latent_reply = None

        logging.info(
            'model_5 init: context=%s glan=%s knowledge=%s latent_reply=%s (fusion_dim=%d)',
            self.use_topology,
            self.use_glan_topology,
            self.use_knowledge_stance_attention,
            self.use_latent_reply,
            fusion_dim,
        )

    def set_train_epoch(self, epoch):
        self.train_epoch = int(epoch)

    def _reply_beta(self):
        if not self.use_latent_reply:
            return 0.0
        ep = self.train_epoch + 1
        if not self.training:
            return self.reply_beta_max
        if ep <= self.reply_beta_zero_epochs:
            return 0.0
        if ep <= self.reply_beta_mid_epochs:
            return self.reply_beta_mid
        return self.reply_beta_max

    def _build_class_weights(self, config):
        if not bool(getattr(config, 'use_class_weight', 0)):
            return None
        counts = getattr(config, 'class_counts', None)
        if not counts:
            return None
        weights = 1.0 / torch.sqrt(torch.tensor(counts, dtype=torch.float))
        weights = weights / weights.mean()
        return weights.to(config.device)

    def _encode_knowledge_batch(self, input_ids, input_masks, input_segments):
        if input_ids is None or input_masks is None or input_masks.sum().item() <= 0:
            return None
        know_out = self.bert(
            input_ids=input_ids,
            attention_mask=input_masks,
            token_type_ids=input_segments,
        ).last_hidden_state
        return know_out[:, 0, :]

    def _concat_features(self, parts):
        return torch.cat(parts, dim=-1)

    def get_topology_gates(self):
        return None

    def _extract_utterance_hidden(self, out, st, ed, mask_positions, dia_id):
        if mask_positions is not None:
            positions = mask_positions[dia_id]
            return torch.stack([out[st + i, positions[i], :] for i in range(ed - st)])
        return out[st:ed, -2, :]

    def _extract_target_repr(self, out, st, ed, target_idx, dia_id):
        if target_idx is None:
            return out[ed - 1, 0, :]
        last_turn = ed - st - 1
        span = target_idx[dia_id][last_turn]
        start, end = int(span[0]), int(span[1])
        utterance_hidden = out[st + last_turn]
        if end <= start:
            return utterance_hidden[start]
        return utterance_hidden[start:end].mean(dim=0)

    def _all_labels_tensor(self, all_label, dia_id, num_turns, device):
        labels = all_label[dia_id]
        if isinstance(labels, torch.Tensor):
            return labels[:num_turns].to(device=device, dtype=torch.long)
        return torch.tensor(labels[:num_turns], device=device, dtype=torch.long)

    def forward(self, **kwargs):
        input_ids = kwargs['input_ids']
        input_masks = kwargs['input_masks']
        input_segments = kwargs['input_segments']
        speakers = kwargs['speakers']
        label = kwargs['label']
        dia_idx = kwargs['dia_idx']
        all_label = kwargs.get('all_label')
        mask_positions = kwargs.get('mask_positions')
        target_idx = kwargs.get('target_idx')
        topology_graphs = kwargs.get('topology_graphs')
        knowledge_favor_input_ids = kwargs.get('knowledge_favor_input_ids')
        knowledge_favor_input_masks = kwargs.get('knowledge_favor_input_masks')
        knowledge_favor_input_segments = kwargs.get('knowledge_favor_input_segments')
        knowledge_against_input_ids = kwargs.get('knowledge_against_input_ids')
        knowledge_against_input_masks = kwargs.get('knowledge_against_input_masks')
        knowledge_against_input_segments = kwargs.get('knowledge_against_input_segments')
        knowledge_neutral_input_ids = kwargs.get('knowledge_neutral_input_ids')
        knowledge_neutral_input_masks = kwargs.get('knowledge_neutral_input_masks')
        knowledge_neutral_input_segments = kwargs.get('knowledge_neutral_input_segments')

        out = self.bert(
            input_ids=input_ids, attention_mask=input_masks, token_type_ids=input_segments,
        ).last_hidden_state
        h_favor_all = self._encode_knowledge_batch(
            knowledge_favor_input_ids, knowledge_favor_input_masks, knowledge_favor_input_segments,
        )
        h_against_all = self._encode_knowledge_batch(
            knowledge_against_input_ids, knowledge_against_input_masks, knowledge_against_input_segments,
        )
        h_neutral_all = self._encode_knowledge_batch(
            knowledge_neutral_input_ids, knowledge_neutral_input_masks, knowledge_neutral_input_segments,
        )

        stance = []
        reply_losses = []
        reply_beta = self._reply_beta()

        for dia_id, (st, ed) in enumerate(dia_idx):
            h = self._extract_utterance_hidden(out, st, ed, mask_positions, dia_id)
            o, _ = self.gru(h.unsqueeze(0))
            v = self.SSE(o.squeeze(0), speakers[dia_id])
            h_sem_base = self.sem_norm(v[-1])

            if self.use_latent_reply and self.latent_reply is not None and reply_beta > 0:
                h_reply = self.latent_reply.prior_reply_vector(v, speakers[dia_id])
                h_sem = self.reply_fuse_norm(h_sem_base + reply_beta * h_reply)
            else:
                h_sem = h_sem_base

            graph = topology_graphs[dia_id] if topology_graphs is not None else None
            parts = [h_sem]

            if self.use_topology and self.topology_encoder is not None and graph is not None:
                topology_v = self.topology_encoder(v, graph)
                parts.append(self.topo_norm(topology_v[-1]))

            if self.use_glan_topology and self.glan_encoder is not None and graph is not None:
                target_repr = self._extract_target_repr(out, st, ed, target_idx, dia_id)
                _, h_glan, _, _ = self.glan_encoder(v, target_repr, graph)
                parts.append(self.glan_norm(h_glan))

            if self.use_knowledge_stance_attention and self.stance_knowledge_attn is not None:
                if (
                    h_favor_all is not None
                    and h_against_all is not None
                    and h_neutral_all is not None
                    and dia_id < h_favor_all.size(0)
                ):
                    valid_mask = torch.tensor([
                        knowledge_favor_input_masks[dia_id].sum().item() > 0,
                        knowledge_against_input_masks[dia_id].sum().item() > 0,
                        knowledge_neutral_input_masks[dia_id].sum().item() > 0,
                    ], device=h_sem_base.device, dtype=torch.bool)
                    if valid_mask.any():
                        h_know, _ = self.stance_knowledge_attn(
                            h_sem_base,
                            h_favor_all[dia_id],
                            h_against_all[dia_id],
                            h_neutral_all[dia_id],
                            valid_mask=valid_mask,
                        )
                        parts.append(h_know)

            stance.append(self._concat_features(parts))

            if self.use_latent_reply and self.training and all_label is not None:
                turn_labels = self._all_labels_tensor(all_label, dia_id, v.size(0), v.device)
                parents = _graph_field(graph, 'reply_parent')
                confidences = _graph_field(graph, 'reply_confidence')
                losses = self.latent_reply.distillation_losses(
                    v.detach(),
                    turn_labels,
                    speakers[dia_id],
                    reply_parents=parents,
                    reply_confidences=confidences,
                    reply_conf_threshold=self.reply_conf_threshold,
                    kl_final_row_weight=self.reply_kl_final_row_weight,
                    kl_full_weight=self.reply_kl_full_weight,
                )
                if losses is not None:
                    reply_losses.append(losses)

        logits = self.fc(torch.stack(stance))
        loss = self.criterion(logits, label)

        if reply_losses:
            distill_kl = torch.stack([item['distill_kl'] for item in reply_losses]).mean()
            if torch.isfinite(distill_kl) and self.reply_kl_lambda > 0:
                loss = loss + self.reply_kl_lambda * distill_kl
            if self.reply_pseudo_lambda > 0:
                pseudo_items = [item['reply_ce'] for item in reply_losses if 'reply_ce' in item]
                if pseudo_items:
                    reply_ce = torch.stack(pseudo_items).mean()
                    if torch.isfinite(reply_ce):
                        loss = loss + self.reply_pseudo_lambda * reply_ce

        return loss, logits, label
