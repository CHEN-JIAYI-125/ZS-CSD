"""
History stance posterior: teacher uses all_label to pick important history turns
and predict final stance; student learns the same path from text only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

REL_SAME = 0
REL_OPPOSE = 1
REL_NEUTRAL = 2
NUM_RELATIONS = 3


def gold_stance_relation(y_hist, y_final):
    y_hist = int(y_hist)
    y_final = int(y_final)
    if y_hist == y_final:
        return REL_SAME
    if y_hist == 2 or y_final == 2:
        return REL_NEUTRAL
    if y_hist in (0, 1) and y_final in (0, 1) and y_hist != y_final:
        return REL_OPPOSE
    return REL_NEUTRAL


def weighted_label_log_prior(alpha, labels, num_classes):
    one_hot = F.one_hot(labels.long(), num_classes=num_classes).float()
    vote = (alpha.unsqueeze(1) * one_hot).sum(dim=0)
    return torch.log(vote.clamp_min(1e-6))


def weighted_prob_log_prior(alpha, probs):
    vote = (alpha.unsqueeze(1) * probs).sum(dim=0)
    return torch.log(vote.clamp_min(1e-6))


class HistoryStancePosterior(nn.Module):
    """
    Teacher: (h_i, h_final, y_i, y_final) -> alpha, history logits
    Student: (h_i, h_final, p_i, p_final) -> alpha, history logits
    history logits = readout(h_pool) + log weighted stance prior
    """

    def __init__(self, hidden_dim, num_classes=3, side_dim=64, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.label_embed = nn.Embedding(num_classes, side_dim)
        self.prob_proj = nn.Linear(num_classes, side_dim)
        self.side_merge = nn.Linear(side_dim * 2, side_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + side_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.relation_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_RELATIONS),
        )
        self.pool_norm = nn.LayerNorm(hidden_dim)
        self.readout = nn.Linear(hidden_dim, num_classes)

    def _side_from_labels(self, hist_labels, final_label):
        hist = self.label_embed(hist_labels.long())
        final = self.label_embed(final_label.long()).unsqueeze(0).expand(hist.size(0), -1)
        return self.side_merge(torch.cat([hist, final], dim=-1))

    def _side_from_probs(self, hist_probs, final_prob):
        hist = self.prob_proj(hist_probs)
        final = self.prob_proj(final_prob.unsqueeze(0)).expand(hist.size(0), -1)
        return self.side_merge(torch.cat([hist, final], dim=-1))

    def _importance(self, v, side_features):
        num_turns = v.size(0)
        if num_turns <= 1 or side_features.size(0) == 0:
            return None, None, self.pool_norm(v[-1])

        h_final = v[-1]
        hist_len = min(num_turns - 1, side_features.size(0))
        scores = []
        for i in range(hist_len):
            feat = torch.cat([v[i], h_final, side_features[i]], dim=-1)
            scores.append(self.scorer(feat).squeeze(-1))
        alpha = F.softmax(torch.stack(scores), dim=0)
        h_pool = torch.matmul(alpha.unsqueeze(0), v[:hist_len]).squeeze(0)
        return alpha, v[:hist_len], self.pool_norm(h_pool)

    def _history_logits(self, h_pool, alpha, stance_prior):
        return self.readout(h_pool) + stance_prior

    def _relation_logits(self, hist_nodes, h_final):
        if hist_nodes is None or hist_nodes.size(0) == 0:
            return None
        return torch.stack([
            self.relation_head(torch.cat([hist_nodes[i], h_final], dim=-1))
            for i in range(hist_nodes.size(0))
        ])

    def student_forward(self, v, stance_probs):
        num_turns = v.size(0)
        if num_turns <= 1:
            h_pool = self.pool_norm(v[-1])
            logits = self.readout(h_pool)
            return None, h_pool, logits, None

        hist_len = num_turns - 1
        hist_probs = stance_probs[:hist_len]
        final_prob = stance_probs[-1]
        side = self._side_from_probs(hist_probs, final_prob)
        alpha, hist_nodes, h_pool = self._importance(v, side)
        if alpha is None:
            logits = self.readout(h_pool)
            return None, h_pool, logits, None

        prior = weighted_prob_log_prior(alpha, hist_probs)
        logits = self._history_logits(h_pool, alpha, prior)
        rel_logits = self._relation_logits(hist_nodes, v[-1])
        return alpha, h_pool, logits, rel_logits

    def teacher_forward(self, v, hist_labels, final_label):
        num_turns = v.size(0)
        hist_len = hist_labels.numel()
        if hist_len == 0:
            h_pool = self.pool_norm(v[-1])
            return None, h_pool, self.readout(h_pool), None

        side = self._side_from_labels(hist_labels, final_label)
        alpha, hist_nodes, h_pool = self._importance(v, side)
        if alpha is None:
            return None, h_pool, self.readout(h_pool), None

        prior = weighted_label_log_prior(alpha, hist_labels, self.num_classes)
        logits = self._history_logits(h_pool, alpha, prior)
        rel_gold = torch.tensor(
            [gold_stance_relation(y, final_label) for y in hist_labels.tolist()],
            device=v.device,
            dtype=torch.long,
        )
        return alpha, h_pool, logits, rel_gold
