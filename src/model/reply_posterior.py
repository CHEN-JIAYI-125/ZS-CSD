"""
all_label posterior-to-prior evidence distillation (PPED core, minimal extras).

all_label (per dialogue, same as official PPED):
  hist_labels = all_label[:-1]
  final_label = all_label[-1]  (= sample label)

Train: posterior(selector + label_emb) -> KL(post || prior) on history weights.
Infer: loss_only mode skips prior at test (identical classifier to v3);
       pool mode concat prior evidence pool (+hidden_dim) to classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common import map_sequence


class EvidenceSelector(nn.Module):
    """Relevance over history turns: f([h_query; h_hist + speaker_emb])."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1, tau: float = 0.2):
        super().__init__()
        self.tau = tau
        self.speaker_emb = nn.Embedding(2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, query_repr, history_repr, query_speaker, hist_speakers):
        length = history_repr.size(0)
        if length == 0:
            return torch.empty(0, device=query_repr.device)

        if not torch.is_tensor(hist_speakers):
            hist_speakers = torch.as_tensor(hist_speakers, device=query_repr.device)
        hist_speakers = hist_speakers.reshape(-1)

        speaker_type = (hist_speakers != query_speaker).long()
        speaker_feat = self.speaker_emb(speaker_type)

        query_exp = query_repr.unsqueeze(0).expand(length, -1)
        history_cond = history_repr + speaker_feat
        features = torch.cat([query_exp, history_cond], dim=-1)
        logits = self.scorer(self.dropout(features)).squeeze(-1)
        return F.softmax(logits / max(self.tau, 1e-4), dim=-1)


class PosteriorEvidenceModule(nn.Module):
    """
    Prior:  h_final, h_i + speaker
    Posterior (train): [h_final; e_y] query, h_i + e_y_i + speaker
    """

    def __init__(self, hidden_dim=768, num_classes=3, dropout=0.1, tau=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.label_emb = nn.Embedding(num_classes, hidden_dim)
        self.posterior_query_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
        )
        self.prior_selector = EvidenceSelector(hidden_dim, dropout=dropout, tau=tau)
        self.posterior_selector = EvidenceSelector(hidden_dim, dropout=dropout, tau=tau)
        self.pool_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _safe_pool(prob, history):
        if history.size(0) == 0 or prob.numel() == 0:
            return torch.zeros(history.size(-1), device=history.device)
        return torch.matmul(prob.unsqueeze(0), history).squeeze(0)

    @staticmethod
    def _kl_post_prior(prior_prob, posterior_prob):
        # Both branches train (no L_post in minimal setup); inference uses prior only.
        teacher = posterior_prob.clamp_min(1e-12)
        student = prior_prob.clamp_min(1e-12)
        return torch.sum(teacher * (teacher.log() - student.log()))

    def _speaker_ids(self, speakers, device):
        return torch.tensor(map_sequence(speakers), device=device, dtype=torch.long)

    def forward_prior(self, utterances, speakers):
        """Label-free path (train + optional inference pool)."""
        if utterances.size(0) <= 1:
            return None, self.pool_norm(torch.zeros_like(utterances[-1]))

        h_final = utterances[-1]
        history = utterances[:-1]
        speaker_ids = self._speaker_ids(speakers, utterances.device)
        prob = self.prior_selector(
            h_final, history, speaker_ids[-1], speaker_ids[:-1],
        )
        pool = self.pool_norm(self._safe_pool(prob, history))
        return prob, pool

    def forward_posterior(self, utterances, speakers, gold_labels):
        """Label-aware teacher (train only)."""
        if utterances.size(0) <= 1:
            return None

        h_final = utterances[-1]
        history = utterances[:-1]
        hist_labels = gold_labels[:-1]
        speaker_ids = self._speaker_ids(speakers, utterances.device)

        query_label = self.label_emb(gold_labels[-1].long())
        posterior_query = self.posterior_query_proj(torch.cat([h_final, query_label], dim=-1))
        history_post = history + self.label_emb(hist_labels.long())

        return self.posterior_selector(
            posterior_query, history_post, speaker_ids[-1], speaker_ids[:-1],
        )

    def training_losses(self, utterances, speakers, gold_labels, return_pool=False):
        prior_prob, pool = self.forward_prior(utterances, speakers)
        post_prob = self.forward_posterior(utterances, speakers, gold_labels)
        if prior_prob is None or post_prob is None or prior_prob.numel() == 0:
            return (None, pool) if return_pool else None
        losses = {'distill_kl': self._kl_post_prior(prior_prob, post_prob)}
        return (losses, pool) if return_pool else losses
