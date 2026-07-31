import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F
from src.common import map_sequence, target_CL
from src.topology.topology_3 import TwoChannelTopologyEncoder


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
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.linear_intra = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear_inter = nn.Linear(hidden_dim, hidden_dim)
        self.attention_intra = Attention(hidden_dim)
        self.attention_inter = Attention(hidden_dim)

    def forward(self, utterances, speakers):
        device = utterances.device
        speakers = torch.tensor(map_sequence(speakers), device=device)
        V_lst = []
        last_speaker_idx = dict()
        for i in range(len(speakers)):
            speaker_id = speakers[i].item()
            if speaker_id not in last_speaker_idx:
                V_lst.append(utterances[i])
            else:
                prev_idx = last_speaker_idx[speaker_id]
                vh_concat = torch.cat((V_lst[prev_idx], utterances[i]), dim=-1)
                q_intra = self.linear_intra(vh_concat)
                c = utterances[:i+1]
                v_intra = self.attention_intra(q_intra, c, c)

                q_inter = self.linear_inter(utterances[i])
                k = torch.stack([V_lst[j] for j in range(prev_idx, i)]) if i > prev_idx else utterances[i].unsqueeze(0)
                v_inter = self.attention_inter(q_inter, k, k) if len(k) > 0 else torch.zeros_like(q_inter)

                V_lst.append(v_intra + v_inter)
            last_speaker_idx[speaker_id] = i
        return torch.stack(V_lst)


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
        self.alpha = config.alpha
        self.use_topology = bool(getattr(config, 'use_topology', 1))
        self.use_knowledge_stance_attention = bool(
            getattr(config, 'use_knowledge_stance_attention', 0)
            or getattr(config, 'use_knowledge_gate', 0)
        )
        hidden = config.gru_hidden
        self.bert = AutoModel.from_pretrained(config.bert_dir)
        self.gru = nn.GRU(input_size=768, hidden_size=hidden, num_layers=config.gru_layer, batch_first=True)

        label_smoothing = float(getattr(config, 'label_smoothing', 0.05))
        class_weight = self._build_class_weights(config)
        self.criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=label_smoothing)

        self.SSE = SSE(hidden_dim=hidden)
        self.sem_norm = nn.LayerNorm(hidden)

        fusion_dim = hidden
        if self.use_topology:
            dropout = float(getattr(config, 'topology_dropout', 0.2))
            gate_init = float(getattr(config, 'topology_gate_init', -2.0))
            self.topology_encoder = TwoChannelTopologyEncoder(
                hidden,
                dropout=dropout,
                gate_init=gate_init,
            )
            self.topo_norm = nn.LayerNorm(hidden)
            fusion_dim += hidden
        else:
            self.topology_encoder = None
            self.topo_norm = None

        if self.use_knowledge_stance_attention:
            dropout = float(getattr(config, 'knowledge_dropout', 0.1))
            self.stance_knowledge_attn = StanceKnowledgeAttention(hidden, bert_dim=768, dropout=dropout)
            fusion_dim += hidden
        else:
            self.stance_knowledge_attn = None

        self.fc = nn.Linear(fusion_dim, config.num_classes)

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

    def forward(self, **kwargs):
        input_ids = kwargs['input_ids']
        input_masks = kwargs['input_masks']
        input_segments = kwargs['input_segments']
        speakers = kwargs['speakers']
        label = kwargs['label']
        dia_idx = kwargs['dia_idx']
        targets = kwargs['target']
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

        H_final = []
        stance = []
        for dia_id, (st, ed) in enumerate(dia_idx):
            h = self._extract_utterance_hidden(out, st, ed, mask_positions, dia_id)
            o, _ = self.gru(h.unsqueeze(0))
            o = o.squeeze(0)
            v = self.SSE(o, speakers[dia_id])
            h_sem = v[-1]

            parts = [self.sem_norm(h_sem)]

            if self.use_topology and topology_graphs is not None:
                topology_v = self.topology_encoder(v, topology_graphs[dia_id])
                parts.append(self.topo_norm(topology_v[-1]))

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

            final_state = self._concat_features(parts)

            H_final.append(v)
            stance.append(final_state)

        stance = torch.stack(stance)
        logits = self.fc(stance)
        ce_loss = self.criterion(logits, label)
        target_contrastive_loss = target_CL(H_final, targets, self.config)
        loss = ce_loss + self.alpha * target_contrastive_loss
        return loss, logits, label
