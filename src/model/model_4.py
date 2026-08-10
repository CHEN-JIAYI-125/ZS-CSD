"""
v4 — v3 stack + all_label posterior knowledge (train-only teacher, distilled student).

Teacher (train only): history text + gold all_label + final text → importance α_t
Student (train/infer):  history text + stance prob (no labels) + final text → importance α_s
Distill KL(α_t || α_s); pool history with α_s and concat to classifier.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.common import map_sequence
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
    """Speaker-aware encoding: intra dialogue attention + speaker hypergraph inter."""

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
    """Attend from current utterance to favor / against / neutral knowledge sides."""

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


class PosteriorImportanceBranch(nn.Module):
    """
    Shared importance scorer over history utterances.
    Teacher side input: gold label embedding.
    Student side input: stance probability vector (no labels at inference).
    """

    def __init__(self, hidden_dim, num_classes=3, side_dim=64, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.label_embed = nn.Embedding(num_classes, side_dim)
        self.prob_proj = nn.Linear(num_classes, side_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + side_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.pool_norm = nn.LayerNorm(hidden_dim)

    def side_from_labels(self, labels):
        return self.label_embed(labels.long())

    def side_from_probs(self, probs):
        return self.prob_proj(probs)

    def forward(self, v, side_features):
        """
        v: [T, H], side_features: [T-1, side_dim] aligned with history turns 0..T-2
        Returns alpha [T-1], h_pool [H]
        """
        num_turns = v.size(0)
        if num_turns <= 1 or side_features.size(0) == 0:
            return None, self.pool_norm(v[-1])

        h_final = v[-1]
        hist_len = min(num_turns - 1, side_features.size(0))
        scores = []
        for i in range(hist_len):
            feat = torch.cat([v[i], h_final, side_features[i]], dim=-1)
            scores.append(self.scorer(feat).squeeze(-1))
        scores = torch.stack(scores)
        alpha = F.softmax(scores, dim=0)
        hist_v = v[:hist_len]
        h_pool = torch.matmul(alpha.unsqueeze(0), hist_v).squeeze(0)
        return alpha, self.pool_norm(h_pool)


class SITCL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_topology = bool(getattr(config, 'use_topology', 1))
        self.use_glan_topology = bool(getattr(config, 'use_glan_topology', 0))
        self.use_knowledge_stance_attention = bool(
            getattr(config, 'use_knowledge_stance_attention', 0)
            or getattr(config, 'use_knowledge_gate', 0)
        )
        self.use_posterior_knowledge = bool(getattr(config, 'use_posterior_knowledge', 1))
        self.posterior_distill_weight = float(getattr(config, 'posterior_distill_weight', 0.5))
        self.posterior_aux_ce_weight = float(getattr(config, 'posterior_aux_ce_weight', 0.2))
        self.posterior_temperature = float(getattr(config, 'posterior_temperature', 1.0))

        hidden = config.gru_hidden
        num_classes = int(getattr(config, 'num_classes', 3))
        self.num_classes = num_classes
        self.bert = AutoModel.from_pretrained(config.bert_dir)
        self.gru = nn.GRU(input_size=768, hidden_size=hidden, num_layers=config.gru_layer, batch_first=True)

        label_smoothing = float(getattr(config, 'label_smoothing', 0.05))
        class_weight = self._build_class_weights(config)
        self.criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=label_smoothing)

        dropout = float(getattr(config, 'sse_dropout', 0.2))
        self.SSE = SSE(hidden_dim=hidden, dropout=dropout)
        self.sem_norm = nn.LayerNorm(hidden)

        fusion_dim = hidden
        if self.use_topology:
            topo_dropout = float(getattr(config, 'topology_dropout', 0.2))
            self.topology_encoder = ContextTopologyEncoder(hidden, dropout=topo_dropout)
            self.topo_norm = nn.LayerNorm(hidden)
            fusion_dim += hidden
        else:
            self.topology_encoder = None
            self.topo_norm = None

        if self.use_glan_topology:
            glan_dropout = float(getattr(config, 'glan_dropout', 0.1))
            glan_local_window = int(getattr(config, 'topology_local_window', 3))
            self.glan_encoder = TargetTopologyEncoder(
                hidden,
                dropout=glan_dropout,
                local_window=glan_local_window,
                branches=_parse_glan_branches(config),
            )
            self.glan_norm = nn.LayerNorm(hidden)
            fusion_dim += hidden
        else:
            self.glan_encoder = None
            self.glan_norm = None

        if self.use_knowledge_stance_attention:
            know_dropout = float(getattr(config, 'knowledge_dropout', 0.1))
            self.stance_knowledge_attn = StanceKnowledgeAttention(hidden, bert_dim=768, dropout=know_dropout)
            fusion_dim += hidden
        else:
            self.stance_knowledge_attn = None

        if self.use_posterior_knowledge:
            post_dropout = float(getattr(config, 'posterior_dropout', 0.1))
            side_dim = int(getattr(config, 'posterior_side_dim', 64))
            self.posterior_branch = PosteriorImportanceBranch(
                hidden,
                num_classes=num_classes,
                side_dim=side_dim,
                dropout=post_dropout,
            )
            self.stance_prior_head = nn.Linear(hidden, num_classes)
            fusion_dim += hidden
        else:
            self.posterior_branch = None
            self.stance_prior_head = None

        self.fc = nn.Linear(fusion_dim, num_classes)
        self.last_posterior_kl = None

        logging.info(
            'model_4 init: context=%s glan=%s knowledge=%s posterior=%s (fusion_dim=%d)',
            self.use_topology,
            self.use_glan_topology,
            self.use_knowledge_stance_attention,
            self.use_posterior_knowledge,
            fusion_dim,
        )

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
        if input_ids is None or input_masks is None:
            return None
        if input_masks.sum().item() <= 0:
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

    def _history_labels_tensor(self, all_label, dia_id, num_turns, device):
        labels = all_label[dia_id]
        if isinstance(labels, torch.Tensor):
            hist = labels[: num_turns - 1].to(device=device, dtype=torch.long)
        else:
            hist = torch.tensor(labels[: num_turns - 1], device=device, dtype=torch.long)
        return hist

    def _stance_probs_from_text(self, v):
        logits = self.stance_prior_head(v) / max(self.posterior_temperature, 1e-6)
        return F.softmax(logits, dim=-1)

    def _posterior_importance(self, v, side_features):
        return self.posterior_branch(v, side_features)

    def _apply_posterior_knowledge(self, v, all_label, dia_id, parts, distill_losses, aux_losses):
        if self.posterior_branch is None or v.size(0) <= 1:
            return

        num_turns = v.size(0)
        hist_len = num_turns - 1
        stance_probs = self._stance_probs_from_text(v[:hist_len])
        side_student = self.posterior_branch.side_from_probs(stance_probs)
        alpha_s, h_pool = self._posterior_importance(v, side_student)
        parts.append(h_pool)

        if not self.training:
            return

        if all_label is None:
            return

        hist_labels = self._history_labels_tensor(all_label, dia_id, num_turns, v.device)
        if hist_labels.numel() == 0:
            return

        side_teacher = self.posterior_branch.side_from_labels(hist_labels)
        alpha_t, _ = self._posterior_importance(v, side_teacher)
        if alpha_t is None or alpha_s is None:
            return

        distill_losses.append(
            F.kl_div(
                alpha_s.log().clamp_min(-20.0),
                alpha_t.detach(),
                reduction='sum',
            )
        )

        if self.posterior_aux_ce_weight > 0:
            aux_losses.append(
                F.cross_entropy(
                    self.stance_prior_head(v[:hist_len]),
                    hist_labels,
                )
            )

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

        out = self.bert(input_ids=input_ids, attention_mask=input_masks, token_type_ids=input_segments).last_hidden_state
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
        distill_losses = []
        aux_losses = []
        for dia_id, (st, ed) in enumerate(dia_idx):
            h = self._extract_utterance_hidden(out, st, ed, mask_positions, dia_id)
            o, _ = self.gru(h.unsqueeze(0))
            o = o.squeeze(0)
            v = self.SSE(o, speakers[dia_id])
            h_sem = v[-1]

            parts = [self.sem_norm(h_sem)]

            graph = topology_graphs[dia_id] if topology_graphs is not None else None

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

            if self.use_posterior_knowledge:
                self._apply_posterior_knowledge(
                    v, all_label, dia_id, parts, distill_losses, aux_losses,
                )

            final_state = self._concat_features(parts)
            stance.append(final_state)

        stance = torch.stack(stance)
        logits = self.fc(stance)
        loss = self.criterion(logits, label)

        if distill_losses:
            kl = torch.stack(distill_losses).mean()
            loss = loss + self.posterior_distill_weight * kl
            self.last_posterior_kl = float(kl.detach().item())
        else:
            self.last_posterior_kl = None

        if aux_losses:
            loss = loss + self.posterior_aux_ce_weight * torch.stack(aux_losses).mean()

        return loss, logits, label
