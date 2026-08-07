"""
v4 — v3 stack with knowledge diff interaction (replaces v3 attention on knowledge).

Dialogue vector h_base = [sem, context_topology?]
Compare with BERT-encoded favor / against / neutral:
  diff_i = proj(h_base) - h_knowledge_i
  sim_i  = cosine(proj(h_base), h_knowledge_i)
Classifier input = concat(h_base, diffs, sims)
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from src.common import map_sequence
from src.topology.topology_4 import ContextTopologyEncoder, SpeakerHypergraphChannel


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


class KnowledgeDiffInteraction(nn.Module):
    """Diff + cosine similarity between dialogue vector and favor/against/neutral knowledge."""

    def __init__(self, base_dim, know_dim=768, dropout=0.1):
        super().__init__()
        self.base_proj = nn.Linear(base_dim, know_dim)
        self.know_norm = nn.LayerNorm(know_dim)
        self.diff_norm = nn.LayerNorm(know_dim)
        self.dropout = nn.Dropout(dropout)
        self.know_dim = know_dim

    def forward(self, h_base, h_favor, h_against, h_neutral, valid_mask=None):
        h_proj = self.base_proj(h_base)
        stance_keys = torch.stack([
            self.know_norm(h_favor),
            self.know_norm(h_against),
            self.know_norm(h_neutral),
        ], dim=0)

        diffs = []
        sims = []
        for idx in range(3):
            if valid_mask is not None and not bool(valid_mask[idx].item()):
                diffs.append(torch.zeros(self.know_dim, device=h_base.device, dtype=h_base.dtype))
                sims.append(torch.zeros((), device=h_base.device, dtype=h_base.dtype))
                continue
            diff = self.diff_norm(h_proj - stance_keys[idx])
            diff = self.dropout(diff)
            diffs.append(diff)
            h_norm = F.normalize(h_proj, p=2, dim=-1)
            k_norm = F.normalize(stance_keys[idx], p=2, dim=-1)
            sims.append(torch.dot(h_norm, k_norm))

        return torch.cat(diffs, dim=-1), torch.stack(sims, dim=0)


class SITCL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_topology = bool(getattr(config, 'use_topology', 1))
        self.use_knowledge_interaction = bool(
            getattr(config, 'use_knowledge_diff_interaction', 1)
            or getattr(config, 'use_knowledge_stance_attention', 0)
            or getattr(config, 'use_knowledge_gate', 0)
        )
        hidden = config.gru_hidden
        know_dim = 768
        self.bert = AutoModel.from_pretrained(config.bert_dir)
        self.gru = nn.GRU(input_size=know_dim, hidden_size=hidden, num_layers=config.gru_layer, batch_first=True)

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

        if self.use_knowledge_interaction:
            know_dropout = float(getattr(config, 'knowledge_dropout', 0.1))
            self.knowledge_diff = KnowledgeDiffInteraction(fusion_dim, know_dim=know_dim, dropout=know_dropout)
            fusion_dim += 3 * know_dim + 3
        else:
            self.knowledge_diff = None

        self.fc = nn.Linear(fusion_dim, config.num_classes)
        self.last_knowledge_sims = None

        logging.info(
            'model_4 init: topology=%s knowledge_diff=%s (fusion_dim=%d)',
            self.use_topology,
            self.use_knowledge_interaction,
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

    def get_topology_gates(self):
        return None

    def _extract_utterance_hidden(self, out, st, ed, mask_positions, dia_id):
        if mask_positions is not None:
            positions = mask_positions[dia_id]
            return torch.stack([out[st + i, positions[i], :] for i in range(ed - st)])
        return out[st:ed, -2, :]

    def forward(self, **kwargs):
        input_ids = kwargs['input_ids']
        input_masks = kwargs['input_masks']
        input_segments = kwargs['input_segments']
        speakers = kwargs['speakers']
        label = kwargs['label']
        dia_idx = kwargs['dia_idx']
        mask_positions = kwargs.get('mask_positions')
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
        sim_records = []
        for dia_id, (st, ed) in enumerate(dia_idx):
            h = self._extract_utterance_hidden(out, st, ed, mask_positions, dia_id)
            o, _ = self.gru(h.unsqueeze(0))
            o = o.squeeze(0)
            v = self.SSE(o, speakers[dia_id])
            h_sem = v[-1]

            parts = [self.sem_norm(h_sem)]
            if self.use_topology and self.topology_encoder is not None and topology_graphs is not None:
                topology_v = self.topology_encoder(v, topology_graphs[dia_id])
                parts.append(self.topo_norm(topology_v[-1]))

            h_base = torch.cat(parts, dim=-1)

            if self.knowledge_diff is not None and (
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
                    diff_cat, sim_vec = self.knowledge_diff(
                        h_base,
                        h_favor_all[dia_id],
                        h_against_all[dia_id],
                        h_neutral_all[dia_id],
                        valid_mask=valid_mask,
                    )
                    h_base = torch.cat([h_base, diff_cat, sim_vec], dim=-1)
                    sim_records.append(sim_vec.detach())

            stance.append(h_base)

        self.last_knowledge_sims = (
            torch.stack(sim_records).mean(dim=0).tolist() if sim_records else None
        )
        stance = torch.stack(stance)
        logits = self.fc(stance)
        loss = self.criterion(logits, label)
        return loss, logits, label
