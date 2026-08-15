"""
Reply-relation posterior on BERT utterance vectors (before GRU).

Teacher (train only): h_bert + gold all_label embedding -> n×n lower-tri matrix.
Student (parallel, label-only): predicted/gold label embedding -> same-shaped matrix.
Loss: row-wise KL(student || teacher.detach()) + optional reply-parent CE + label CE.

Main context path (GRU -> SSE -> topology) uses sentences only; no extra fc dims.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def lower_triangular_mask(num_turns, device):
    return torch.tril(torch.ones(num_turns, num_turns, device=device, dtype=torch.bool))


def masked_row_targets(reply_parents, num_turns, device):
    """Gold reply target column for each row (defaults to self)."""
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
    """Stable KL on lower-triangular rows only (avoids 0*log(0) on upper triangle)."""
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
    """Dual-branch lower-triangular reply-relation posterior (train auxiliary only)."""

    def __init__(self, hidden_dim=768, num_classes=3, dropout=0.1, tau=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau = max(tau, 1e-4)
        self.label_emb = nn.Embedding(num_classes, hidden_dim)

        self.teacher_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.student_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.label_head = nn.Linear(hidden_dim, num_classes)

    def _label_vectors_from_logits(self, label_logits):
        soft = F.softmax(label_logits, dim=-1)
        return soft @ self.label_emb.weight

    def _reply_matrix(self, turn_repr):
        """turn_repr: [n, D] -> row-softmax lower-triangular [n, n]."""
        num_turns = turn_repr.size(0)
        if num_turns == 0:
            return turn_repr.new_zeros(0, 0)

        logits = torch.matmul(turn_repr, turn_repr.transpose(0, 1)) / self.tau
        mask = lower_triangular_mask(num_turns, turn_repr.device)
        logits = logits.masked_fill(~mask, float('-inf'))
        probs = F.softmax(logits, dim=-1)
        return probs * mask.to(dtype=probs.dtype)

    def forward(self, h_bert, gold_labels, reply_parents):
        """
        h_bert: [n, D] BERT utterance vectors (detached by caller if needed).
        gold_labels: [n] all_label per turn.
        reply_parents: list/int sequence length n.
        """
        num_turns = h_bert.size(0)
        if num_turns <= 1:
            return None

        gold_labels = gold_labels.long()
        label_logits = self.label_head(h_bert)

        teacher_repr = self.teacher_proj(h_bert + self.label_emb(gold_labels))
        teacher_matrix = self._reply_matrix(teacher_repr)

        student_label_vecs = self._label_vectors_from_logits(label_logits)
        student_repr = self.student_proj(student_label_vecs)
        student_matrix = self._reply_matrix(student_repr)

        distill_kl = rowwise_kl(student_matrix, teacher_matrix)

        row_targets = masked_row_targets(reply_parents, num_turns, h_bert.device)
        teacher_row_logits = []
        for row in range(num_turns):
            teacher_row_logits.append(teacher_matrix[row, : row + 1].clamp_min(1e-12).log())
        teacher_reply_ce = torch.stack([
            F.nll_loss(teacher_row_logits[row].unsqueeze(0), row_targets[row].unsqueeze(0))
            for row in range(num_turns)
        ]).mean()

        label_ce = F.cross_entropy(label_logits, gold_labels)

        return {
            'distill_kl': distill_kl,
            'teacher_reply_ce': teacher_reply_ce,
            'label_ce': label_ce,
        }
