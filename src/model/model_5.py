import logging

import torch
import torch.nn as nn
from transformers import AutoModel

from src.common import alllabel_supcon_loss, map_sequence
from src.model.pped_evidence import PPEDEvidenceModule
from src.topology.topology_4 import (
    ContextTopologyEncoder,
    SpeakerHypergraphChannel,
    TargetTopologyEncoder,
)


def _parse_glan_branches(config):
    raw = getattr(config, 'glan_branches', 'global,local,struct,final')
    return [part.strip() for part in str(raw).split(',') if part.strip()]


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
    """Experiment C: v3 backbone + all_label SupCon + PPED evidence distillation (prior residual at inference)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_topology = bool(getattr(config, 'use_topology', 1))
        self.use_glan_topology = bool(getattr(config, 'use_glan_topology', 0))
        self.use_knowledge_stance_attention = bool(
            getattr(config, 'use_knowledge_stance_attention', 0)
            or getattr(config, 'use_knowledge_gate', 0)
        )
        self.use_alllabel_supcon = bool(getattr(config, 'use_alllabel_supcon', 0))
        self.alllabel_supcon_lambda = float(getattr(config, 'alllabel_supcon_lambda', 0.05))
        self.alllabel_supcon_tau = float(getattr(config, 'alllabel_supcon_tau', 0.07))
        self.cross_target_positive_weight = float(getattr(config, 'cross_target_positive_weight', 0.5))

        self.use_pped_evidence = bool(getattr(config, 'use_pped_evidence', 0))
        self.evidence_gamma = float(getattr(config, 'evidence_gamma', 0.05))
        self.evidence_kl_lambda = float(getattr(config, 'evidence_kl_lambda', 0.01))
        self.evidence_post_ce_lambda = float(getattr(config, 'evidence_post_ce_lambda', 0.02))
        self.evidence_gamma_zero_epochs = int(getattr(config, 'evidence_gamma_zero_epochs', 2))
        self.evidence_gamma_mid_epochs = int(getattr(config, 'evidence_gamma_mid_epochs', 5))
        self.evidence_gamma_mid = float(getattr(config, 'evidence_gamma_mid', 0.02))
        self.evidence_kl_start_epoch = int(getattr(config, 'evidence_kl_start_epoch', 6))
        self.evidence_aux_zero_epochs = int(getattr(config, 'evidence_aux_zero_epochs', 2))
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
        self.target_proj = nn.Linear(768, hidden)

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

        if self.use_pped_evidence:
            reply_dropout = float(getattr(config, 'reply_posterior_dropout', 0.1))
            reply_tau = float(getattr(config, 'reply_posterior_tau', 0.2))
            num_classes = int(getattr(config, 'num_classes', 3))
            self.pped_evidence = PPEDEvidenceModule(
                hidden_dim=hidden,
                num_classes=num_classes,
                dropout=reply_dropout,
                tau=reply_tau,
            )
        else:
            self.pped_evidence = None

        logging.info(
            'model_5 init: context=%s glan=%s knowledge=%s supcon=%s pped_evidence=%s (fusion_dim=%d)',
            self.use_topology,
            self.use_glan_topology,
            self.use_knowledge_stance_attention,
            self.use_alllabel_supcon,
            self.use_pped_evidence,
            fusion_dim,
        )

    def set_train_epoch(self, epoch):
        self.train_epoch = int(epoch)

    def _evidence_gamma(self):
        if not self.use_pped_evidence:
            return 0.0
        ep = self.train_epoch + 1
        if ep <= self.evidence_gamma_zero_epochs:
            return 0.0
        if ep <= self.evidence_gamma_mid_epochs:
            return self.evidence_gamma_mid
        return self.evidence_gamma

    def _evidence_aux_active(self):
        return (self.train_epoch + 1) > self.evidence_aux_zero_epochs

    def _evidence_kl_active(self):
        return (self.train_epoch + 1) >= self.evidence_kl_start_epoch

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

    def _collect_supcon(self, supcon_vectors, supcon_stances, supcon_target_ids, target_to_id, v, turn_labels, target_str):
        if target_str not in target_to_id:
            target_to_id[target_str] = len(target_to_id)
        tid = target_to_id[target_str]
        supcon_vectors.append(v)
        supcon_stances.append(turn_labels)
        supcon_target_ids.extend([tid] * v.size(0))

    def forward(self, **kwargs):
        input_ids = kwargs['input_ids']
        input_masks = kwargs['input_masks']
        input_segments = kwargs['input_segments']
        speakers = kwargs['speakers']
        label = kwargs['label']
        dia_idx = kwargs['dia_idx']
        all_label = kwargs.get('all_label')
        targets = kwargs.get('target', [])
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
        logit_deltas = []
        evidence_losses = []
        supcon_vectors = []
        supcon_stances = []
        supcon_target_ids = []
        target_to_id = {}
        gamma = self._evidence_gamma()

        for dia_id, (st, ed) in enumerate(dia_idx):
            h = self._extract_utterance_hidden(out, st, ed, mask_positions, dia_id)
            o, _ = self.gru(h.unsqueeze(0))
            v = self.SSE(o.squeeze(0), speakers[dia_id])
            h_sem = self.sem_norm(v[-1])
            target_repr = self.target_proj(self._extract_target_repr(out, st, ed, target_idx, dia_id))

            if self.use_alllabel_supcon and self.training and all_label is not None:
                turn_labels = self._all_labels_tensor(all_label, dia_id, v.size(0), v.device)
                target_str = str(targets[dia_id]) if dia_id < len(targets) else str(dia_id)
                self._collect_supcon(
                    supcon_vectors, supcon_stances, supcon_target_ids,
                    target_to_id, v, turn_labels, target_str,
                )

            parts = [h_sem]
            graph = topology_graphs[dia_id] if topology_graphs is not None else None

            if self.use_topology and self.topology_encoder is not None and graph is not None:
                topology_v = self.topology_encoder(v, graph)
                parts.append(self.topo_norm(topology_v[-1]))

            if self.use_glan_topology and self.glan_encoder is not None and graph is not None:
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
                    ], device=h_sem.device, dtype=torch.bool)
                    if valid_mask.any():
                        h_know, _ = self.stance_knowledge_attn(
                            h_sem,
                            h_favor_all[dia_id],
                            h_against_all[dia_id],
                            h_neutral_all[dia_id],
                            valid_mask=valid_mask,
                        )
                        parts.append(h_know)

            stance.append(self._concat_features(parts))

            if self.use_pped_evidence and self.pped_evidence is not None and gamma > 0 and v.size(0) > 1:
                delta = self.pped_evidence.evidence_logit_delta(
                    h_sem, v[-1], v[:-1], speakers[dia_id], target_repr,
                )
                logit_deltas.append(delta)
            else:
                logit_deltas.append(None)

            if (
                self.use_pped_evidence
                and self.pped_evidence is not None
                and self.training
                and all_label is not None
                and self._evidence_aux_active()
                and v.size(0) > 1
            ):
                turn_labels = self._all_labels_tensor(all_label, dia_id, v.size(0), v.device)
                final_label = label[dia_id]
                ev = self.pped_evidence.training_losses(
                    v, speakers[dia_id], target_repr, turn_labels, final_label,
                )
                if ev is not None:
                    evidence_losses.append(ev)

        logits = self.fc(torch.stack(stance))
        if gamma > 0:
            for i, delta in enumerate(logit_deltas):
                if delta is not None:
                    logits[i] = logits[i] + gamma * delta

        loss = self.criterion(logits, label)

        if (
            self.use_alllabel_supcon
            and self.training
            and supcon_vectors
            and self.alllabel_supcon_lambda > 0
        ):
            vectors = torch.cat(supcon_vectors, dim=0)
            stances = torch.cat(supcon_stances, dim=0)
            target_ids = torch.tensor(supcon_target_ids, device=vectors.device, dtype=torch.long)
            cl_loss = alllabel_supcon_loss(
                vectors,
                stances,
                target_ids,
                tau=self.alllabel_supcon_tau,
                cross_target_weight=self.cross_target_positive_weight,
            )
            if torch.isfinite(cl_loss):
                loss = loss + self.alllabel_supcon_lambda * cl_loss

        if evidence_losses:
            if self._evidence_kl_active() and self.evidence_kl_lambda > 0:
                distill_kl = torch.stack([item['distill_kl'] for item in evidence_losses]).mean()
                if torch.isfinite(distill_kl):
                    loss = loss + self.evidence_kl_lambda * distill_kl
            if self.evidence_post_ce_lambda > 0:
                post_ce = torch.stack([item['post_ce'] for item in evidence_losses]).mean()
                if torch.isfinite(post_ce):
                    loss = loss + self.evidence_post_ce_lambda * post_ce

        return loss, logits, label
