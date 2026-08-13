"""
v5 — v4 + implicit latent relation matrix on topology graph, distill, augmented GCN.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.common import map_sequence
from src.model.history_posterior import HistoryStancePosterior
from src.topology.topology_5 import (
    ContextTopologyEncoder,
    ImplicitTopologyGCN,
    SpeakerHypergraphChannel,
    TargetTopologyEncoder,
    stance_relation_loss,
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
        self.use_implicit_topology = bool(getattr(config, 'use_implicit_topology', 1))
        self.posterior_fusion_mode = str(getattr(config, 'posterior_fusion_mode', 'student_path'))
        self.posterior_logit_blend_weight = float(getattr(config, 'posterior_logit_blend_weight', 0.25))
        self.implicit_fusion_mode = str(getattr(config, 'implicit_fusion_mode', 'loss_only'))
        self.posterior_distill_weight = float(getattr(config, 'posterior_distill_weight', 0.1))
        self.posterior_logit_distill_weight = float(getattr(config, 'posterior_logit_distill_weight', 0.15))
        self.posterior_relation_distill_weight = float(getattr(config, 'posterior_relation_distill_weight', 0.1))
        self.posterior_pool_distill_weight = float(getattr(config, 'posterior_pool_distill_weight', 0.05))
        self.posterior_aux_ce_weight = float(getattr(config, 'posterior_aux_ce_weight', 0.05))
        self.posterior_temperature = float(getattr(config, 'posterior_temperature', 1.0))
        self.implicit_distill_weight = float(getattr(config, 'implicit_distill_weight', 0.08))
        self.use_stance_relation_loss = bool(getattr(config, 'use_stance_relation_loss', 1))
        self.stance_relation_weight = float(getattr(config, 'stance_relation_weight', 0.1))
        self.aux_start_epoch = int(getattr(
            config, 'aux_start_epoch',
            getattr(config, 'posterior_start_epoch', 0),
        ))
        self._train_epoch = 0

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
            self.posterior_branch = HistoryStancePosterior(
                hidden,
                num_classes=num_classes,
                side_dim=side_dim,
                dropout=post_dropout,
            )
            self.stance_prior_head = nn.Linear(hidden, num_classes)
            if self.posterior_fusion_mode in ('concat', 'student_path'):
                fusion_dim += hidden
        else:
            self.posterior_branch = None
            self.stance_prior_head = None

        if self.use_implicit_topology:
            implicit_dropout = float(getattr(config, 'implicit_dropout', 0.2))
            implicit_side_dim = int(getattr(config, 'implicit_side_dim', 32))
            implicit_edge_weight = float(getattr(config, 'implicit_edge_weight', 0.5))
            self.implicit_topology = ImplicitTopologyGCN(
                hidden,
                num_classes=num_classes,
                side_dim=implicit_side_dim,
                dropout=implicit_dropout,
                implicit_edge_weight=implicit_edge_weight,
            )
            self.implicit_norm = nn.LayerNorm(hidden)
            if self.implicit_fusion_mode == 'concat':
                fusion_dim += hidden
            if self.stance_prior_head is None:
                self.stance_prior_head = nn.Linear(hidden, num_classes)
        else:
            self.implicit_topology = None
            self.implicit_norm = None

        self.fc = nn.Linear(fusion_dim, num_classes)
        self.last_posterior_kl = None
        self.last_implicit_mse = None

        logging.info(
            'model_5 init: context=%s glan=%s knowledge=%s posterior=%s(%s) implicit=%s(%s) aux_from=%d (fusion_dim=%d)',
            self.use_topology,
            self.use_glan_topology,
            self.use_knowledge_stance_attention,
            self.use_posterior_knowledge,
            self.posterior_fusion_mode if self.use_posterior_knowledge else 'off',
            self.use_implicit_topology,
            self.implicit_fusion_mode if self.use_implicit_topology else 'off',
            self.aux_start_epoch,
            fusion_dim,
        )

    def set_train_epoch(self, epoch):
        self._train_epoch = int(epoch)

    def _aux_loss_active(self):
        return self.training

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

    def _all_labels_tensor(self, all_label, dia_id, num_turns, device):
        labels = all_label[dia_id]
        if isinstance(labels, torch.Tensor):
            return labels[:num_turns].to(device=device, dtype=torch.long)
        return torch.tensor(labels[:num_turns], device=device, dtype=torch.long)

    def _history_labels_tensor(self, all_label, dia_id, num_turns, device):
        labels = all_label[dia_id]
        if isinstance(labels, torch.Tensor):
            hist = labels[: num_turns - 1].to(device=device, dtype=torch.long)
        else:
            hist = torch.tensor(labels[: num_turns - 1], device=device, dtype=torch.long)
        return hist

    def _stance_probs_from_text(self, v):
        if self.stance_prior_head is None:
            raise RuntimeError('stance_prior_head is required for posterior/implicit paths')
        logits = self.stance_prior_head(v) / max(self.posterior_temperature, 1e-6)
        return F.softmax(logits, dim=-1)

    def _apply_implicit_topology(self, v, graph, all_label, dia_id, parts, implicit_distill_losses):
        if self.implicit_topology is None:
            return

        concat_mode = self.implicit_fusion_mode == 'concat'

        if not self._aux_loss_active():
            if concat_mode:
                if graph is None or v.size(0) <= 1:
                    parts.append(self.implicit_norm(v[-1]))
                else:
                    stance_probs = self._stance_probs_from_text(v)
                    side_student = self.implicit_topology.relation.node_side_from_probs(
                        stance_probs, final_prob=stance_probs[-1],
                    )
                    h_implicit, _, _ = self.implicit_topology(
                        v, graph, side_student, node_side_teacher=None,
                    )
                    parts.append(self.implicit_norm(h_implicit[-1]))
            return

        if graph is None or v.size(0) <= 1:
            if concat_mode:
                parts.append(self.implicit_norm(v[-1]))
            return

        stance_probs = self._stance_probs_from_text(v)
        side_student = self.implicit_topology.relation.node_side_from_probs(
            stance_probs, final_prob=stance_probs[-1],
        )
        side_teacher = None
        if all_label is not None:
            turn_labels = self._all_labels_tensor(all_label, dia_id, v.size(0), v.device)
            side_teacher = self.implicit_topology.relation.node_side_from_labels(
                turn_labels, final_label=turn_labels[-1],
            )

        h_implicit, distill_loss, _ = self.implicit_topology(
            v, graph, side_student, node_side_teacher=side_teacher,
        )
        if concat_mode:
            parts.append(self.implicit_norm(h_implicit[-1]))
        if distill_loss is not None:
            implicit_distill_losses.append(distill_loss)

    def _posterior_use_concat(self):
        return self.posterior_fusion_mode in ('concat', 'student_path')

    def _posterior_use_logit_blend(self):
        return self.posterior_fusion_mode in ('logits_blend', 'student_path')

    def _apply_posterior_knowledge(
        self,
        v,
        all_label,
        dia_id,
        parts,
        distill_losses,
        aux_losses,
        logit_distill_losses,
        relation_distill_losses,
        pool_distill_losses,
        hist_logits_list,
    ):
        if self.posterior_branch is None:
            return

        stance_probs = self._stance_probs_from_text(v)
        alpha_s, h_pool_s, hist_logits_s, rel_logits_s = self.posterior_branch.student_forward(v, stance_probs)

        if self._posterior_use_concat():
            parts.append(h_pool_s)
        if self._posterior_use_logit_blend() and hist_logits_s is not None:
            hist_logits_list.append(hist_logits_s)

        if not self._aux_loss_active() or not self.training or all_label is None:
            return

        num_turns = v.size(0)
        if num_turns <= 1:
            return

        hist_len = num_turns - 1
        turn_labels = self._all_labels_tensor(all_label, dia_id, num_turns, v.device)
        hist_labels = turn_labels[:hist_len]
        if hist_labels.numel() == 0:
            return

        final_label = turn_labels[-1]
        alpha_t, h_pool_t, hist_logits_t, rel_gold = self.posterior_branch.teacher_forward(
            v, hist_labels, final_label,
        )

        if alpha_t is not None and alpha_s is not None:
            distill_losses.append(
                F.kl_div(
                    alpha_s.log().clamp_min(-20.0),
                    alpha_t.detach(),
                    reduction='sum',
                )
            )

        if (
            h_pool_t is not None
            and h_pool_s is not None
            and self.posterior_pool_distill_weight > 0
        ):
            pool_distill_losses.append(F.mse_loss(h_pool_s, h_pool_t.detach()))

        if (
            hist_logits_t is not None
            and hist_logits_s is not None
            and self.posterior_logit_distill_weight > 0
        ):
            logit_distill_losses.append(
                F.kl_div(
                    F.log_softmax(hist_logits_s / self.posterior_temperature, dim=-1),
                    F.softmax(hist_logits_t.detach() / self.posterior_temperature, dim=-1),
                    reduction='batchmean',
                )
            )

        if (
            rel_logits_s is not None
            and rel_gold is not None
            and self.posterior_relation_distill_weight > 0
        ):
            relation_distill_losses.append(F.cross_entropy(rel_logits_s, rel_gold))

        if self.posterior_aux_ce_weight > 0:
            aux_losses.append(
                F.cross_entropy(
                    self.stance_prior_head(v[:hist_len]),
                    hist_labels,
                )
            )

    def _apply_stance_relation_loss(self, v, all_label, dia_id, relation_losses):
        if not self.use_stance_relation_loss or not self._aux_loss_active() or all_label is None:
            return
        turn_labels = self._all_labels_tensor(all_label, dia_id, v.size(0), v.device)
        if turn_labels.numel() < 2:
            return
        relation_losses.append(
            stance_relation_loss(v, turn_labels.tolist(), self.config)
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
        logit_distill_losses = []
        relation_distill_losses = []
        pool_distill_losses = []
        implicit_distill_losses = []
        relation_losses = []
        hist_logits_list = []
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

            if self.use_implicit_topology:
                self._apply_implicit_topology(
                    v, graph, all_label, dia_id, parts, implicit_distill_losses,
                )

            if self.use_glan_topology and self.glan_encoder is not None and graph is not None:
                target_repr = self._extract_target_repr(out, st, ed, target_idx, dia_id)
                _, h_glan, _, _ = self.glan_encoder(v, target_repr, graph)
                parts.append(self.glan_norm(h_glan))

            if self.use_knowledge_stance_attention and self.stance_knowledge_attn is not None:
                h_know = torch.zeros_like(h_sem)
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
                    v,
                    all_label,
                    dia_id,
                    parts,
                    distill_losses,
                    aux_losses,
                    logit_distill_losses,
                    relation_distill_losses,
                    pool_distill_losses,
                    hist_logits_list,
                )

            self._apply_stance_relation_loss(v, all_label, dia_id, relation_losses)

            final_state = self._concat_features(parts)
            stance.append(final_state)

        stance = torch.stack(stance)
        logits_main = self.fc(stance)
        logits = logits_main
        if hist_logits_list and self.posterior_logit_blend_weight > 0:
            hist_logits = torch.stack(hist_logits_list)
            logits = logits_main + self.posterior_logit_blend_weight * hist_logits
        loss = self.criterion(logits_main, label)

        if distill_losses:
            kl = torch.stack(distill_losses).mean()
            loss = loss + self.posterior_distill_weight * kl
            self.last_posterior_kl = float(kl.detach().item())
        else:
            self.last_posterior_kl = None

        if aux_losses:
            loss = loss + self.posterior_aux_ce_weight * torch.stack(aux_losses).mean()

        if logit_distill_losses:
            loss = loss + self.posterior_logit_distill_weight * torch.stack(logit_distill_losses).mean()

        if pool_distill_losses:
            loss = loss + self.posterior_pool_distill_weight * torch.stack(pool_distill_losses).mean()

        if implicit_distill_losses:
            mse = torch.stack(implicit_distill_losses).mean()
            loss = loss + self.implicit_distill_weight * mse
            self.last_implicit_mse = float(mse.detach().item())
        else:
            self.last_implicit_mse = None

        if relation_distill_losses:
            loss = loss + self.posterior_relation_distill_weight * torch.stack(relation_distill_losses).mean()

        if relation_losses:
            loss = loss + self.stance_relation_weight * torch.stack(relation_losses).mean()

        return loss, logits, label
