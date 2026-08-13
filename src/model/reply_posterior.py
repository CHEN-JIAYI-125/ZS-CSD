"""
Reply-relation posterior on raw BERT utterance vectors (768-d), before GRU.

Placement:  BERT -> ReplyPosteriorDistiller (train aux loss) -> GRU -> SSE -> topology
Not parallel to GRU: distillation reads h_bert once, then the same h_bert enters GRU.

Teacher/student are two parallel heads on h_bert (teacher uses h+gold label; student label-only).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def lower_triangular_mask(num_turns, device):
    return torch.tril(torch.ones(num_turns, num_turns, device=device, dtype=torch.bool))


def masked_row_targets(reply_parents, num_turns, device):
    """Gold reply parent index for each row (self if no valid parent)."""
    targets = []
    for i in range(num_turns):
        parent = int(reply_parents[i]) if i < len(reply_parents) else i
        if parent < 0 or parent > i:
            parent = i
        targets.append(parent)
    return torch.tensor(targets, device=device, dtype=torch.long)


class ReplyPosteriorDistiller(nn.Module):
    """
    Row i predicts a distribution over columns j<=i (reply-to history including self).
    Both branches emit the same [n, n] lower-triangular stochastic matrix.
    """

    def __init__(self, hidden_dim=768, num_classes=3, label_dim=64, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.teacher_label_embed = nn.Embedding(num_classes, label_dim)
        self.posterior_fuse = nn.Sequential(
            nn.Linear(hidden_dim + label_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        self.student_label_embed = nn.Embedding(num_classes, label_dim)
        self.label_head = nn.Linear(hidden_dim, num_classes)
        self.student_fuse = nn.Sequential(
            nn.Linear(label_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        self.scale = hidden_dim ** -0.5

    def _reply_matrix(self, node_repr):
        logits = torch.matmul(node_repr, node_repr.transpose(0, 1)) * self.scale
        mask = lower_triangular_mask(node_repr.size(0), node_repr.device)
        logits = logits.masked_fill(~mask, float('-inf'))
        probs = F.softmax(logits, dim=-1)
        return probs, mask

    def teacher_forward(self, hidden, gold_labels):
        label_vec = self.teacher_label_embed(gold_labels.long())
        posterior_repr = self.posterior_fuse(torch.cat([hidden, label_vec], dim=-1))
        return self._reply_matrix(posterior_repr)

    def student_forward(self, hidden):
        label_logits = self.label_head(hidden)
        label_probs = F.softmax(label_logits, dim=-1)
        label_vec = torch.matmul(label_probs, self.student_label_embed.weight)
        student_repr = self.student_fuse(label_vec)
        matrix, mask = self._reply_matrix(student_repr)
        return matrix, mask, label_logits

    def training_losses(self, hidden, gold_labels, reply_parents):
        probs_teacher, mask = self.teacher_forward(hidden, gold_labels)
        probs_student, _, label_logits = self.student_forward(hidden)

        valid = mask.float()
        denom = valid.sum().clamp_min(1.0)

        distill_kl = F.kl_div(
            probs_student.log().clamp_min(-20.0),
            probs_teacher.detach(),
            reduction='none',
        )
        distill_kl = (distill_kl * valid).sum() / denom

        distill_mse = F.mse_loss(probs_student, probs_teacher.detach(), reduction='none')
        distill_mse = (distill_mse * valid).sum() / denom

        row_targets = masked_row_targets(reply_parents, hidden.size(0), hidden.device)
        teacher_ce = F.nll_loss(probs_teacher.log().clamp_min(-20.0), row_targets)
        label_ce = F.cross_entropy(label_logits, gold_labels.long())

        return {
            'distill_kl': distill_kl,
            'distill_mse': distill_mse,
            'teacher_ce': teacher_ce,
            'label_ce': label_ce,
            'probs_teacher': probs_teacher,
            'probs_student': probs_student,
        }
