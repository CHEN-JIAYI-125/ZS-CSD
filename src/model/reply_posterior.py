"""
Lower-triangular reply-relation posterior on BERT utterance vectors.

Teacher (train only): h + gold all_label -> n×n matrix (posterior).
Student (train + test): label-only branch via label_head(h) -> same matrix.
Train: KL(student || teacher.detach()) on detached h (after main forward).
Infer: student last-row weights pool history into final turn (bounded gate before GRU).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def lower_triangular_mask(num_turns, device):
    return torch.tril(torch.ones(num_turns, num_turns, device=device, dtype=torch.bool))


def masked_row_targets(reply_parents, num_turns, device):
    targets = []
    for i in range(num_turns):
        parent = reply_parents[i] if i < len(reply_parents) else i
        if not isinstance(parent, int):
            parent = int(parent)
        if parent < 0 or parent > i:
            parent = i
        targets.append(parent)
    return torch.tensor(targets, device=device, dtype=torch.long)


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

    def _student_repr(self, h_bert, use_dropout):
        label_logits = self.label_head(h_bert)
        label_vecs = self._label_vectors_from_logits(label_logits)
        hidden = self.student_linear(label_vecs)
        hidden = F.gelu(hidden)
        if use_dropout and self.training:
            hidden = F.dropout(hidden, p=self.dropout_p, training=True)
        return self._reply_matrix(hidden), label_logits

    def _teacher_matrix(self, h_bert, gold_labels, use_dropout):
        gold_labels = gold_labels.long()
        repr_in = h_bert + self.label_emb(gold_labels)
        hidden = self.teacher_proj[0](repr_in)
        hidden = self.teacher_proj[1](hidden)
        if use_dropout and self.training:
            hidden = F.dropout(hidden, p=self.dropout_p, training=True)
        return self._reply_matrix(hidden)

    def enrich_gru_input(self, h_bert, gate):
        """Student reply pool on final turn; no dropout (stable GRU RNG vs v3)."""
        if h_bert.size(0) <= 1 or gate <= 0:
            return h_bert

        student_matrix, _ = self._student_repr(h_bert, use_dropout=False)
        weights = student_matrix[-1, : h_bert.size(0)]
        pool = self.pool_norm(torch.matmul(weights.unsqueeze(0), h_bert).squeeze(0))
        h_out = h_bert.clone()
        h_out[-1] = h_bert[-1] + gate * (pool - h_bert[-1])
        return h_out

    def distillation_losses(self, h_bert, gold_labels):
        """Aux losses on detached h; runs after main forward."""
        if h_bert.size(0) <= 1:
            return None

        gold_labels = gold_labels.long()
        teacher_matrix = self._teacher_matrix(h_bert, gold_labels, use_dropout=True)
        student_matrix, label_logits = self._student_repr(h_bert, use_dropout=True)
        return {
            'distill_kl': rowwise_kl(student_matrix, teacher_matrix),
            'label_ce': F.cross_entropy(label_logits, gold_labels),
        }
