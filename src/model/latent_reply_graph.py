"""
Posterior-guided latent reply graph (PPED-aligned prior/posterior pair scorers).

Teacher (train): sentence h + gold all_label + speaker/distance -> P_post
Student/Prior (train+test): sentence h + speaker/distance only -> P_prior
KL distills P_post -> P_prior; prior final-row pool feeds stance classifier at test.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common import map_sequence


def lower_triangular_mask(num_turns, device):
    return torch.tril(torch.ones(num_turns, num_turns, device=device, dtype=torch.bool))


def _rowwise_kl(student_probs, teacher_probs, rows=None):
    num_turns = student_probs.size(0)
    if num_turns == 0:
        return student_probs.new_zeros(())

    if rows is None:
        rows = range(num_turns)

    losses = []
    for row in rows:
        end = row + 1
        student_row = student_probs[row, :end].clamp_min(1e-12)
        teacher_row = teacher_probs[row, :end].detach().clamp_min(1e-12)
        student_row = student_row / student_row.sum()
        teacher_row = teacher_row / teacher_row.sum()
        losses.append(torch.sum(teacher_row * (teacher_row.log() - student_row.log())))
    if not losses:
        return student_probs.new_zeros(())
    return torch.stack(losses).mean()


class PairReplyScorer(nn.Module):
    """Independent Q/K-style pair scorer; diagonal uses separate root MLP (not h·h)."""

    def __init__(self, hidden_dim, num_classes, use_labels, max_turn_dist=32, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_labels = use_labels
        self.label_emb = nn.Embedding(num_classes, hidden_dim) if use_labels else None
        self.same_spk_emb = nn.Embedding(2, 16)
        self.dist_emb = nn.Embedding(max_turn_dist + 1, 16)

        extra = hidden_dim * 2 if use_labels else 0
        feat_dim = hidden_dim * 4 + extra + 16 + 16
        self.pair_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.root_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _speaker_tensor(self, speakers, device):
        if not torch.is_tensor(speakers):
            speakers = torch.tensor(map_sequence(speakers), device=device, dtype=torch.long)
        return speakers.reshape(-1).long()

    def _pair_features(self, h_i, h_j, label_i, label_j, same_spk, dist_idx):
        parts = [h_i, h_j, h_i - h_j, h_i * h_j]
        if self.use_labels and label_i is not None and label_j is not None:
            parts.extend([label_i, label_j])
        parts.extend([
            self.same_spk_emb(same_spk),
            self.dist_emb(dist_idx),
        ])
        return torch.cat(parts, dim=-1)

    def forward_matrix(self, turns, labels, speakers, tau):
        num_turns = turns.size(0)
        if num_turns == 0:
            return turns.new_zeros(0, 0)

        device = turns.device
        speaker_ids = self._speaker_tensor(speakers, device)
        logits = turns.new_full((num_turns, num_turns), float('-inf'))

        label_vecs = None
        if self.use_labels and labels is not None:
            label_vecs = self.label_emb(labels.long())

        for i in range(num_turns):
            for j in range(i + 1):
                if j == i:
                    logits[i, j] = self.root_mlp(turns[i]).squeeze(-1)
                else:
                    li = label_vecs[i] if label_vecs is not None else None
                    lj = label_vecs[j] if label_vecs is not None else None
                    same_spk = (speaker_ids[i] == speaker_ids[j]).long()
                    dist_idx = torch.tensor(min(i - j, self.dist_emb.num_embeddings - 1), device=device)
                    feat = self._pair_features(turns[i], turns[j], li, lj, same_spk, dist_idx)
                    logits[i, j] = self.pair_mlp(feat).squeeze(-1)

        mask = lower_triangular_mask(num_turns, device)
        probs = F.softmax(logits / max(tau, 1e-4), dim=-1)
        return probs * mask.to(dtype=probs.dtype)


class LatentReplyGraph(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=3, dropout=0.1, tau=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau = max(tau, 1e-4)
        self.posterior_scorer = PairReplyScorer(
            hidden_dim, num_classes, use_labels=True, dropout=dropout,
        )
        self.prior_scorer = PairReplyScorer(
            hidden_dim, num_classes, use_labels=False, dropout=dropout,
        )
        self.reply_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool_norm = nn.LayerNorm(hidden_dim)

    def prior_matrix(self, turns, speakers):
        return self.prior_scorer.forward_matrix(turns, None, speakers, self.tau)

    def posterior_matrix(self, turns, gold_labels, speakers):
        return self.posterior_scorer.forward_matrix(turns, gold_labels, speakers, self.tau)

    def aggregate_reply(self, turns, reply_weights):
        if turns.size(0) == 0 or reply_weights.numel() == 0:
            device = turns.device if turns.numel() else reply_weights.device
            dtype = turns.dtype if turns.numel() else reply_weights.dtype
            return self.pool_norm(torch.zeros(self.hidden_dim, device=device, dtype=dtype))
        pool = torch.matmul(reply_weights.unsqueeze(0), turns).squeeze(0)
        return self.pool_norm(self.reply_proj(pool))

    def prior_reply_vector(self, turns, speakers):
        if turns.size(0) <= 1:
            return self.aggregate_reply(turns[:0], turns.new_zeros(0))
        prior = self.prior_matrix(turns, speakers)
        weights = prior[-1, : turns.size(0)]
        return self.aggregate_reply(turns, weights)

    def distillation_losses(
        self,
        turns,
        gold_labels,
        speakers,
        reply_parents=None,
        reply_confidences=None,
        reply_conf_threshold=0.7,
        kl_final_row_weight=1.0,
        kl_full_weight=0.5,
    ):
        if turns.size(0) <= 1:
            return None

        post = self.posterior_matrix(turns, gold_labels, speakers)
        prior = self.prior_matrix(turns, speakers)

        kl_full = _rowwise_kl(prior, post)
        kl_final = _rowwise_kl(prior, post, rows=[turns.size(0) - 1])
        distill_kl = kl_full_weight * kl_full + kl_final_row_weight * kl_final

        losses = {'distill_kl': distill_kl}

        if reply_parents is not None:
            reply_ce = turns.new_zeros(())
            count = 0
            num_turns = turns.size(0)
            for i in range(num_turns):
                if reply_parents is None or i >= len(reply_parents):
                    continue
                parent = int(reply_parents[i])
                conf = 1.0
                if reply_confidences is not None and i < len(reply_confidences):
                    conf = float(reply_confidences[i])
                if conf < reply_conf_threshold or parent < 0:
                    continue
                target_col = parent if 0 <= parent <= i else i
                row = prior[i, : i + 1].clamp_min(1e-12)
                row = row / row.sum()
                target = torch.tensor(target_col, device=turns.device)
                reply_ce = reply_ce + F.nll_loss(row.log(), target) * conf
                count += 1
            if count > 0:
                losses['reply_ce'] = reply_ce / count

        return losses
