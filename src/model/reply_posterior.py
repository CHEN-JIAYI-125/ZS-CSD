"""
Lower-triangular reply-relation posterior on contextualized utterance vectors (after SSE).

Teacher (train): v + gold all_label -> n×n matrix.
Student (train/test): label-only branch -> same matrix -> evidence pool + interaction.
Train: KL from epoch kl_warmup; gate ramps after gate_warmup (PPED: evidence used at inference).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def lower_triangular_mask(num_turns, device):
    return torch.tril(torch.ones(num_turns, num_turns, device=device, dtype=torch.bool))


def rowwise_kl(student_probs, teacher_probs):
    num_turns = student_probs.size(0)
    if num_turns == 0:
        return student_probs.new_zeros(())

    losses = []
    for row in range(num_turns):
        end = row + 1
        student_row = student_probs[row, :end].clamp_min(1e-12)
        teacher_row = teacher_probs[row, :end].detach().clamp_min(1e-12)
        student_row = student_row / student_row.sum()
        teacher_row = teacher_row / teacher_row.sum()
        losses.append(torch.sum(teacher_row * (teacher_row.log() - student_row.log())))
    return torch.stack(losses).mean()


class ReplyPosteriorDistiller(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=3, dropout=0.1, tau=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau = max(tau, 1e-4)
        self.dropout_p = dropout
        self.label_emb = nn.Embedding(num_classes, hidden_dim)

        self.teacher_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.student_linear = nn.Linear(hidden_dim, hidden_dim)
        self.label_head = nn.Linear(hidden_dim, num_classes)
        self.pool_norm = nn.LayerNorm(hidden_dim)
        self.inter_norm = nn.LayerNorm(hidden_dim)

    def _label_vectors_from_logits(self, label_logits):
        soft = F.softmax(label_logits, dim=-1)
        return soft @ self.label_emb.weight

    def _reply_matrix(self, turn_repr):
        num_turns = turn_repr.size(0)
        if num_turns == 0:
            return turn_repr.new_zeros(0, 0)

        logits = torch.matmul(turn_repr, turn_repr.transpose(0, 1)) / self.tau
        mask = lower_triangular_mask(num_turns, turn_repr.device)
        logits = logits.masked_fill(~mask, float('-inf'))
        probs = F.softmax(logits, dim=-1)
        return probs * mask.to(dtype=probs.dtype)

    def _student_matrix(self, turns, use_dropout):
        label_logits = self.label_head(turns)
        label_vecs = self._label_vectors_from_logits(label_logits)
        hidden = self.student_linear(label_vecs)
        hidden = F.gelu(hidden)
        if use_dropout and self.training:
            hidden = F.dropout(hidden, p=self.dropout_p, training=True)
        return self._reply_matrix(hidden), label_logits

    def _teacher_matrix(self, turns, gold_labels, use_dropout):
        gold_labels = gold_labels.long()
        repr_in = turns + self.label_emb(gold_labels)
        hidden = self.teacher_proj[0](repr_in)
        hidden = self.teacher_proj[1](hidden)
        if use_dropout and self.training:
            hidden = F.dropout(hidden, p=self.dropout_p, training=True)
        return self._reply_matrix(hidden)

    def student_evidence(self, turns):
        """PPED-style pool + |h_n - pool| for final turn (student matrix, no dropout)."""
        if turns.size(0) <= 1:
            zero = torch.zeros_like(turns[-1])
            return self.pool_norm(zero), self.inter_norm(zero)

        student_matrix, _ = self._student_matrix(turns, use_dropout=False)
        weights = student_matrix[-1, : turns.size(0)]
        pool = torch.matmul(weights.unsqueeze(0), turns).squeeze(0)
        pool = self.pool_norm(pool)
        interaction = self.inter_norm(torch.abs(turns[-1] - pool))
        return pool, interaction

    def blend_final_semantic(self, h_sem, turns, gate):
        """Bounded residual fusion into h_sem (no extra fc dim)."""
        if gate <= 0 or turns.size(0) <= 1:
            return h_sem
        pool, interaction = self.student_evidence(turns)
        evidence = pool + interaction
        return h_sem + gate * (evidence - h_sem)

    def distillation_losses(self, turns, gold_labels):
        if turns.size(0) <= 1:
            return None

        gold_labels = gold_labels.long()
        teacher_matrix = self._teacher_matrix(turns, gold_labels, use_dropout=True)
        student_matrix, label_logits = self._student_matrix(turns, use_dropout=True)
        return {
            'distill_kl': rowwise_kl(student_matrix, teacher_matrix),
            'label_ce': F.cross_entropy(label_logits, gold_labels),
        }
