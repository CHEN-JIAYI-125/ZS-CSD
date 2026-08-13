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

    def _reply_logits(self, node_repr):
        logits = torch.matmul(node_repr, node_repr.transpose(0, 1)) * self.scale
        mask = lower_triangular_mask(node_repr.size(0), node_repr.device)
        logits = logits.masked_fill(~mask, -1e4)
        return logits, mask

    def _reply_probs(self, logits):
        return F.softmax(logits, dim=-1)

    def teacher_forward(self, hidden, gold_labels):
        label_vec = self.teacher_label_embed(gold_labels.long())
        posterior_repr = self.posterior_fuse(torch.cat([hidden, label_vec], dim=-1))
        logits, mask = self._reply_logits(posterior_repr)
        return self._reply_probs(logits), logits, mask

    def student_forward(self, hidden):
        label_logits = self.label_head(hidden)
        label_probs = F.softmax(label_logits, dim=-1)
        label_vec = torch.matmul(label_probs, self.student_label_embed.weight)
        student_repr = self.student_fuse(label_vec)
        logits, mask = self._reply_logits(student_repr)
        probs = self._reply_probs(logits)
        return probs, logits, mask, label_logits

    @staticmethod
    def _row_distill_loss(probs_student, probs_teacher):
        """KL + MSE on each lower-triangular row only (avoid 0*log(0) on masked cells)."""
        num_turns = probs_student.size(0)
        kl_rows = []
        mse_rows = []
        for i in range(num_turns):
            end = i + 1
            p_s = probs_student[i, :end].clamp_min(1e-8)
            p_t = probs_teacher[i, :end].detach().clamp_min(1e-8)
            kl_rows.append(F.kl_div(p_s.log(), p_t, reduction='sum'))
            mse_rows.append(F.mse_loss(p_s, p_t, reduction='sum'))
        return torch.stack(kl_rows).mean(), torch.stack(mse_rows).mean()

    def training_losses(self, hidden, gold_labels, reply_parents):
        probs_teacher, logits_teacher, _ = self.teacher_forward(hidden, gold_labels)
        probs_student, _, _, label_logits = self.student_forward(hidden)

        distill_kl, distill_mse = self._row_distill_loss(probs_student, probs_teacher)

        row_targets = masked_row_targets(reply_parents, hidden.size(0), hidden.device)
        teacher_rows = []
        for i in range(hidden.size(0)):
            teacher_rows.append(logits_teacher[i, : i + 1])
        teacher_ce = torch.stack([
            F.cross_entropy(teacher_rows[i].unsqueeze(0), row_targets[i].unsqueeze(0))
            for i in range(hidden.size(0))
        ]).mean()

        label_ce = F.cross_entropy(label_logits, gold_labels.long())

        return {
            'distill_kl': distill_kl,
            'distill_mse': distill_mse,
            'teacher_ce': teacher_ce,
            'label_ce': label_ce,
            'probs_teacher': probs_teacher,
            'probs_student': probs_student,
        }
