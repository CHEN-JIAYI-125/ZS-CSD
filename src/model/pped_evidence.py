"""
PPED-style target-conditioned evidence (final utterance -> history distribution).

Posterior (train): h + all_label + target + speaker
Prior (train/test):  h + target + speaker
KL distills posterior weights -> prior; posterior CE trains teacher on live representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common import map_sequence


class EvidenceSelector(nn.Module):
    """score = MLP([query+target; hist+speaker])."""

    def __init__(self, hidden_dim, dropout=0.1, tau=0.2):
        super().__init__()
        self.tau = max(tau, 1e-4)
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
            return history_repr.new_zeros(0)

        if not torch.is_tensor(hist_speakers):
            hist_speakers = torch.as_tensor(hist_speakers, device=history_repr.device)
        hist_speakers = hist_speakers.reshape(-1)

        speaker_type = (hist_speakers != query_speaker).long()
        speaker_feat = self.speaker_emb(speaker_type)
        query_cond = query_repr.unsqueeze(0).expand(length, -1) + target_repr.unsqueeze(0).expand(length, -1)
        hist_cond = history_repr + speaker_feat
        features = torch.cat([query_cond, hist_cond], dim=-1)
        logits = self.scorer(self.dropout(features)).squeeze(-1)
        return F.softmax(logits / self.tau, dim=-1)


class PPEDEvidenceModule(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=3, dropout=0.1, tau=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.label_emb = nn.Embedding(num_classes, hidden_dim)
        self.posterior_query_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.prior_selector = EvidenceSelector(hidden_dim, dropout=dropout, tau=tau)
        self.posterior_selector = EvidenceSelector(hidden_dim, dropout=dropout, tau=tau)
        self.pool_norm = nn.LayerNorm(hidden_dim)
        self.inter_norm = nn.LayerNorm(hidden_dim)
        self.post_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.evidence_head = nn.Linear(hidden_dim * 4, num_classes, bias=False)
        nn.init.zeros_(self.evidence_head.weight)

    def _speaker_ids(self, speakers, device):
        return torch.tensor(map_sequence(speakers), device=device, dtype=torch.long)

    @staticmethod
    def _pool(prob, history):
        if history.size(0) == 0 or prob.numel() == 0:
            return history.new_zeros(history.size(-1))
        return torch.matmul(prob.unsqueeze(0), history).squeeze(0)

    def forward_prior(self, final_utt, history, speakers, target_repr):
        if history.size(0) == 0:
            zero = torch.zeros_like(final_utt)
            return zero, zero, final_utt.new_zeros(0)

        speaker_ids = self._speaker_ids(speakers, final_utt.device)
        prob = self.prior_selector(
            final_utt, history, speaker_ids[-1], speaker_ids[:-1], target_repr,
        )
        pool = self.pool_norm(self._pool(prob, history))
        interaction = self.inter_norm(torch.abs(final_utt - pool))
        return pool, interaction, prob

    def forward_posterior(self, final_utt, history, speakers, target_repr, hist_labels, final_label):
        if history.size(0) == 0:
            return final_utt.new_zeros(0), None, None

        speaker_ids = self._speaker_ids(speakers, final_utt.device)
        hist_post = history + self.label_emb(hist_labels.long())
        final_label_emb = self.label_emb(final_label.long())
        post_query = self.posterior_query_proj(torch.cat([final_utt, final_label_emb], dim=-1))
        prob = self.posterior_selector(
            post_query, hist_post, speaker_ids[-1], speaker_ids[:-1], target_repr,
        )
        pool = self.pool_norm(self._pool(prob, history))
        interaction = self.inter_norm(torch.abs(final_utt - pool))
        return prob, pool, interaction

    def prior_evidence_features(self, h_sem, final_utt, history, speakers, target_repr):
        pool, interaction, _ = self.forward_prior(final_utt, history, speakers, target_repr)
        return torch.cat([h_sem, pool, interaction, target_repr], dim=-1)

    def evidence_logit_delta(self, h_sem, final_utt, history, speakers, target_repr):
        if history.size(0) == 0:
            return h_sem.new_zeros(self.evidence_head.out_features)
        feat = self.prior_evidence_features(h_sem, final_utt, history, speakers, target_repr)
        return self.evidence_head(feat.unsqueeze(0)).squeeze(0)

    def training_losses(self, v, speakers, target_repr, hist_labels, final_label):
        """Live v (no detach): posterior CE trains backbone; KL uses detached post weights."""
        if v.size(0) <= 1:
            return None

        final_utt = v[-1]
        history = v[:-1]
        hist_lbl = hist_labels[:-1]

        post_prob, post_pool, post_inter = self.forward_posterior(
            final_utt, history, speakers, target_repr, hist_lbl, final_label,
        )
        _, _, prior_prob = self.forward_prior(final_utt, history, speakers, target_repr)

        post_repr = torch.cat([final_utt, post_pool, post_inter, target_repr], dim=-1)
        post_logits = self.post_classifier(post_repr.unsqueeze(0)).squeeze(0)
        post_ce = F.cross_entropy(
            post_logits.unsqueeze(0),
            final_label.reshape(()).long().unsqueeze(0),
        )

        teacher = post_prob.detach().clamp_min(1e-12)
        student = prior_prob.clamp_min(1e-12)
        distill_kl = torch.sum(teacher * (teacher.log() - student.log()))

        return {'distill_kl': distill_kl, 'post_ce': post_ce}
