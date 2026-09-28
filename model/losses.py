"""
model.losses (PyTorch version)
=================================
CE loss (Eq. 32) and the multi-perspective Cross-modal Label Contrastive
(CLC) loss (Eqs. 33-40), same design as the NumPy version, on real torch
tensors (autograd, GPU-capable).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PerspectivePooling(nn.Module):
    """Eqs. 33-35: K learnable masks pool a (B, L, D) sequence into K
    (B, D') vectors, concatenated to (B, K*D')."""
    def __init__(self, d_model, d_proj, n_perspectives=4):
        super().__init__()
        self.K = n_perspectives
        self.mask_proj = nn.Linear(d_model, n_perspectives)
        self.heads = nn.ModuleList([nn.Linear(d_model, d_proj) for _ in range(n_perspectives)])

    def forward(self, x, key_padding_mask=None):
        # x: (B, L, D); key_padding_mask: (B, L) True where PADDING (to exclude)
        mask_logits = self.mask_proj(x)   # (B, L, K)
        if key_padding_mask is not None:
            mask_logits = mask_logits.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf"))
        weights = F.softmax(mask_logits, dim=1)   # softmax over L, per perspective
        zs = []
        for k in range(self.K):
            w = weights[:, :, k:k + 1]            # (B, L, 1)
            pooled = (x * w).sum(dim=1)           # (B, D)
            z = self.heads[k](pooled)             # (B, D')
            z = F.normalize(z, p=2, dim=-1)
            zs.append(z)
        return torch.cat(zs, dim=-1)               # (B, K*D')


def clc_pairwise_loss(Z_a, Z_b, labels, temperature=0.1):
    """One direction of the label-aware contrastive loss (generalizes Eq. 37)."""
    sim = Z_a @ Z_b.T / temperature   # (B, B) -- Eq. 36
    log_prob = F.log_softmax(sim, dim=1)

    same_label = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()   # (B,B)
    pos_count = same_label.sum(dim=1).clamp(min=1.0)

    pos_log_prob = (log_prob * same_label).sum(dim=1) / pos_count
    return -pos_log_prob.mean()


def clc_stage_loss(Z_t, Z_a, Z_m, labels, temperature=0.1):
    """Eq. 40: average of symmetrized text<->audio and text<->mfcc losses."""
    l_ta = (clc_pairwise_loss(Z_t, Z_a, labels, temperature) +
            clc_pairwise_loss(Z_a, Z_t, labels, temperature)) * 0.5
    l_tm = (clc_pairwise_loss(Z_t, Z_m, labels, temperature) +
            clc_pairwise_loss(Z_m, Z_t, labels, temperature)) * 0.5
    return (l_ta + l_tm) * 0.5


class MultiObjectiveLoss(nn.Module):
    """Eq. 31: L = delta * L_CE + lambda * sum_{l=1..3} L_CLC^(l)."""
    def __init__(self, d_model, d_proj=64, n_perspectives=4, delta=0.5, lam=0.2, temperature=0.1):
        super().__init__()
        self.delta = delta
        self.lam = lam
        self.temperature = temperature
        self.pool_t = nn.ModuleList([PerspectivePooling(d_model, d_proj, n_perspectives) for _ in range(3)])
        self.pool_a = nn.ModuleList([PerspectivePooling(d_model, d_proj, n_perspectives) for _ in range(3)])
        self.pool_m = nn.ModuleList([PerspectivePooling(d_model, d_proj, n_perspectives) for _ in range(3)])

    def clc_total(self, stage_idx, X_t, X_a, X_m, labels, text_mask=None, audio_mask=None, mfcc_mask=None):
        Z_t = self.pool_t[stage_idx](X_t, key_padding_mask=text_mask)
        Z_a = self.pool_a[stage_idx](X_a, key_padding_mask=audio_mask)
        Z_m = self.pool_m[stage_idx](X_m, key_padding_mask=mfcc_mask)
        return clc_stage_loss(Z_t, Z_a, Z_m, labels, self.temperature)

    def forward(self, logits, labels, stage1_feats, stage2_feats, stage3_feats,
                text_mask=None, audio_mask=None, mfcc_mask=None):
        ce = F.cross_entropy(logits, labels)
        clc1 = self.clc_total(0, *stage1_feats, labels, text_mask, audio_mask, mfcc_mask)
        clc2 = self.clc_total(1, *stage2_feats, labels, text_mask, audio_mask, mfcc_mask)
        clc3 = self.clc_total(2, *stage3_feats, labels, text_mask, audio_mask, mfcc_mask)
        total = ce * self.delta + (clc1 + clc2 + clc3) * self.lam
        return total, {"ce": ce, "clc1": clc1, "clc2": clc2, "clc3": clc3}
