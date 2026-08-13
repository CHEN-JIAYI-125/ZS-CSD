"""
PPED-style all_label posterior distillation (adapted for v4 pre-GRU placement).

Prior selector (inference): text-only query + history -> importance over history.
Posterior selector (train teacher): label-augmented query/history -> same-shaped distribution.
Loss: KL(posterior.detach() || prior) on detached BERT vectors so main v3 path is preserved.
Reference: model_pped_final — SpeakerAwareEvidenceEstimator + posterior_query_proj + label_emb.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common import map_sequence


class SpeakerAwareHistorySelector(nn.Module):
    """Score historical utterances for the final query (PPED SpeakerAwareEvidenceEstimator)."""

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

    def forward(self, query_repr, history_repr, query_speaker, hist_speakers, target_repr):
        length = history_repr.size(0)
        if length == 0:
            return torch.empty(0, device=query_repr.device)

        if not torch.is_tensor(hist_speakers):
            hist_speakers = torch.as_tensor(hist_speakers, device=query_repr.device)
        hist_speakers = hist_speakers.reshape(-1)

        speaker_type = (hist_speakers != query_speaker).long()
        speaker_feat = self.speaker_emb(speaker_type)

        query_exp = query_repr.unsqueeze(0).expand(length, -1)
        target_exp = target_repr.unsqueeze(0).expand(length, -1)
        query_cond = query_exp + target_exp
        history_cond = history_repr + speaker_feat
        features = torch.cat([query_cond, history_cond], dim=-1)
        logits = self.scorer(self.dropout(features)).squeeze(-1)
        return F.softmax(logits / max(self.tau, 1e-4), dim=-1)


class PosteriorHistoryDistiller(nn.Module):
    """
    Train-only teacher/student over dialog history before GRU.
    Student = prior selector (text); teacher = posterior selector (h + all_label).
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
        self.prior_selector = SpeakerAwareHistorySelector(hidden_dim, dropout=dropout, tau=tau)
        self.posterior_selector = SpeakerAwareHistorySelector(hidden_dim, dropout=dropout, tau=tau)

    @staticmethod
    def _posterior_kl(prior_prob, posterior_prob):
        teacher = posterior_prob.detach().clamp_min(1e-12)
        student = prior_prob.clamp_min(1e-12)
        return torch.sum(teacher * (teacher.log() - student.log()))

    def training_losses(self, hidden, speakers, gold_labels, target_repr, detach_hidden=True):
        if hidden.size(0) <= 1:
            return None

        if detach_hidden:
            hidden = hidden.detach()
            target_repr = target_repr.detach()

        final_utt = hidden[-1]
        history = hidden[:-1]
        hist_labels = gold_labels[:-1]

        speakers_mapped = map_sequence(speakers)
        speaker_ids = torch.tensor(speakers_mapped, device=hidden.device, dtype=torch.long)
        query_speaker = speaker_ids[-1]
        hist_speakers = speaker_ids[:-1]

        prior_prob = self.prior_selector(
            final_utt, history, query_speaker, hist_speakers, target_repr,
        )
        if prior_prob.numel() == 0:
            return None

        query_label = self.label_emb(gold_labels[-1].long())
        posterior_query = self.posterior_query_proj(torch.cat([final_utt, query_label], dim=-1))
        history_post = history + self.label_emb(hist_labels.long())
        posterior_prob = self.posterior_selector(
            posterior_query, history_post, query_speaker, hist_speakers, target_repr,
        )
        if posterior_prob.numel() == 0:
            return None

        return {
            'distill_kl': self._posterior_kl(prior_prob, posterior_prob),
        }
