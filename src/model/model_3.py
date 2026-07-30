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


class SITCL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.alpha = config.alpha
        self.use_topology = bool(getattr(config, 'use_topology', 1))
        self.use_knowledge_gate = bool(getattr(config, 'use_knowledge_gate', 1))
        self.last_topology_gates = None
        self.last_knowledge_gate = None
        self.bert = AutoModel.from_pretrained(config.bert_dir)
        self.gru = nn.GRU(input_size=768, hidden_size=config.gru_hidden, num_layers=config.gru_layer, batch_first=True)
        self.fc = nn.Linear(config.gru_hidden, config.num_classes)

        label_smoothing = float(getattr(config, 'label_smoothing', 0.05))
        class_weight = self._build_class_weights(config)
        self.criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=label_smoothing)

        self.SSE = SSE(hidden_dim=config.gru_hidden)
        if self.use_topology:
            dropout = float(getattr(config, 'topology_dropout', 0.2))
            gate_init = float(getattr(config, 'topology_gate_init', -2.0))
            self.topology_encoder = TwoChannelTopologyEncoder(
                config.gru_hidden,
                dropout=dropout,
                gate_init=gate_init,
            )
            self.fusion_gate = nn.Linear(config.gru_hidden * 2, config.gru_hidden)
            fusion_init = float(getattr(config, 'topology_fusion_gate_init', -1.0))
            nn.init.zeros_(self.fusion_gate.weight)
            nn.init.constant_(self.fusion_gate.bias, fusion_init)

        if self.use_knowledge_gate:
            self.knowledge_gate = nn.Linear(config.gru_hidden * 2, config.gru_hidden)
            kg_init = float(getattr(config, 'knowledge_gate_init', 2.0))
            nn.init.zeros_(self.knowledge_gate.weight)
            nn.init.constant_(self.knowledge_gate.bias, kg_init)
        else:
            self.knowledge_gate = None

    def _build_class_weights(self, config):
        if not bool(getattr(config, 'use_class_weight', 0)):
            return None
        counts = getattr(config, 'class_counts', None)
        if not counts:
            return None
        weights = 1.0 / torch.sqrt(torch.tensor(counts, dtype=torch.float))
        weights = weights / weights.mean()
        return weights.to(config.device)

    def get_knowledge_gate(self):
        if self.knowledge_gate is None or self.last_knowledge_gate is None:
            return None
        with torch.no_grad():
            return float(self.last_knowledge_gate.detach().cpu().mean().item())

    def _encode_knowledge(self, knowledge_input_ids, knowledge_input_masks, knowledge_input_segments):
        if knowledge_input_ids is None or knowledge_input_masks is None:
            return None
        if knowledge_input_masks.sum().item() <= 0:
            return None
        know_out = self.bert(
            input_ids=knowledge_input_ids,
            attention_mask=knowledge_input_masks,
            token_type_ids=knowledge_input_segments,
        ).last_hidden_state
        return know_out[:, 0, :]

    def _fuse_knowledge(self, h_text, h_know):
        gate = torch.sigmoid(self.knowledge_gate(torch.cat([h_text, h_know], dim=-1)))
        self.last_knowledge_gate = gate.mean()
        return gate * h_text + (1.0 - gate) * h_know

    def get_topology_gates(self):
        if not self.use_topology or self.topology_encoder is None:
            return None
        with torch.no_grad():
            return self.topology_encoder.channel_weights().detach().cpu().tolist()

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
        knowledge_input_ids = kwargs.get('knowledge_input_ids')
        knowledge_input_masks = kwargs.get('knowledge_input_masks')
        knowledge_input_segments = kwargs.get('knowledge_input_segments')

        out = self.bert(input_ids=input_ids, attention_mask=input_masks, token_type_ids=input_segments).last_hidden_state
        h_know_all = self._encode_knowledge(
            knowledge_input_ids, knowledge_input_masks, knowledge_input_segments,
        )

        H_final = []
        stance = []
        for dia_id, (st, ed) in enumerate(dia_idx):
            h = self._extract_utterance_hidden(out, st, ed, mask_positions, dia_id)
            o, _ = self.gru(h.unsqueeze(0))
            o = o.squeeze(0)
            v = self.SSE(o, speakers[dia_id])
            h_sem = v[-1]

            if self.use_topology and topology_graphs is not None:
                topology_v = self.topology_encoder(v, topology_graphs[dia_id])
                topology_final = topology_v[-1]
                gate = torch.sigmoid(self.fusion_gate(torch.cat([h_sem, topology_final], dim=-1)))
                final_state = gate * topology_final + (1.0 - gate) * h_sem
            else:
                final_state = h_sem

            if (
                self.use_knowledge_gate
                and self.knowledge_gate is not None
                and h_know_all is not None
                and h_know_all.size(0) > dia_id
            ):
                h_know = h_know_all[dia_id]
                if knowledge_input_masks is not None and knowledge_input_masks[dia_id].sum().item() > 0:
                    final_state = self._fuse_knowledge(final_state, h_know)

            H_final.append(v)
            stance.append(final_state)

        if self.use_topology and self.topology_encoder is not None:
            self.last_topology_gates = self.get_topology_gates()

        stance = torch.stack(stance)
        logits = self.fc(stance)
        ce_loss = self.criterion(logits, label)
        target_contrastive_loss = target_CL(H_final, targets, self.config)
        loss = ce_loss + self.alpha * target_contrastive_loss
        return loss, logits, label
