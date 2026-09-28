"""
model.losses
============
CE loss (Eq. 32) and the multi-perspective Cross-modal Label Contrastive
(CLC) loss (Eqs. 33-40).

CLC loss, simplified but faithful sketch of the paper's mechanism:
  1. Aggregate a sequence (B, L, D) down to K "perspective" vectors via
     learnable masks over the sequence length (Eq. 33), project + L2
     normalize each (Eq. 34), concatenate into one global vector (Eq. 35).
  2. Compute cross-modal similarity between two modalities' global vectors
     (Eq. 36), then an InfoNCE-style loss using the emotion LABELS (not
     the batch index) as the positive-pair signal (Eq. 37) - i.e. samples
     sharing a label are treated as positives for each other.
  3. Average the two directions (a->t and t->a) for a symmetric loss
     (Eqs. 38-39), then average text-audio and text-mfcc losses (Eq. 40).
"""
import numpy as np
from nptorch.tensor import Tensor, concat, randn, zeros
from nptorch.nn import Module, Linear


class PerspectivePooling(Module):
    """Implements Eqs. 33-35: K learnable attention-style masks pool a
    (B, L, D) sequence into K (B, D') vectors, concatenated to (B, K*D')."""
    def __init__(self, d_model, d_proj, n_perspectives=4):
        self.K = n_perspectives
        self.mask_proj = Linear(d_model, n_perspectives)   # produces raw mask logits per token
        self.heads = [Linear(d_model, d_proj) for _ in range(n_perspectives)]

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, D)
        mask_logits = self.mask_proj(x)                 # (B, L, K)
        masks = mask_logits.softmax(axis=1)              # softmax over L -> weights per perspective (Eq. 33 denom/numer)
        zs = []
        for k in range(self.K):
            w = masks[:, :, k:k + 1]                     # (B, L, 1)
            pooled = (x * w).sum(axis=1)                 # (B, D)  -- weighted mean, Eq. 33
            z = self.heads[k](pooled)                    # (B, D') -- Eq. 34 projection
            z = z.l2_normalize(axis=-1)
            zs.append(z)
        return concat(zs, axis=-1)                        # (B, K*D') -- Eq. 35


def clc_pairwise_loss(Z_a: Tensor, Z_b: Tensor, labels: np.ndarray, temperature: float = 0.1) -> Tensor:
    """
    One direction of the label-aware contrastive loss (generalizes Eq. 37):
    for each anchor i in modality A, all j in modality B sharing the same
    label are positives; everything else is a negative.
    """
    B = Z_a.shape[0]
    sim = Z_a.matmul(Z_b.T) * (1.0 / temperature)   # (B, B) -- Eq. 36 (S_a->b)
    logits = sim - sim.data.max(axis=1, keepdims=True)  # stabilize (numeric only, no grad issue: constant shift)
    log_prob = logits - logits.exp().sum(axis=1, keepdims=True).log()  # log-softmax over each row

    same_label = (labels[:, None] == labels[None, :]).astype(np.float64)  # (B,B) positive mask
    pos_count = same_label.sum(axis=1, keepdims=True)
    pos_count = np.clip(pos_count, 1.0, None)

    pos_log_prob = log_prob * Tensor(same_label)
    per_anchor = pos_log_prob.sum(axis=1) / Tensor(pos_count[:, 0])
    loss = -per_anchor.mean()
    return loss


def clc_stage_loss(Z_t: Tensor, Z_a: Tensor, Z_m: Tensor, labels: np.ndarray, temperature: float = 0.1) -> Tensor:
    """One stage's CLC loss (Eq. 40): average of text<->audio and text<->mfcc,
    each itself symmetrized (Eqs. 38-39)."""
    l_ta = (clc_pairwise_loss(Z_t, Z_a, labels, temperature) +
            clc_pairwise_loss(Z_a, Z_t, labels, temperature)) * 0.5
    l_tm = (clc_pairwise_loss(Z_t, Z_m, labels, temperature) +
            clc_pairwise_loss(Z_m, Z_t, labels, temperature)) * 0.5
    return (l_ta + l_tm) * 0.5


def cross_entropy_loss(logits: Tensor, labels: np.ndarray) -> Tensor:
    """Eq. 32. logits: (B, C), labels: (B,) int array of class indices."""
    B, C = logits.shape
    log_probs = logits - logits.exp().sum(axis=1, keepdims=True).log()
    one_hot = np.zeros((B, C))
    one_hot[np.arange(B), labels] = 1.0
    picked = (log_probs * Tensor(one_hot)).sum(axis=1)
    return -picked.mean()


class MultiObjectiveLoss(Module):
    """Combines CE with the 3-stage CLC loss (Eq. 31):
    L = delta * L_CE + lambda * sum_{l=1..3} L_CLC^(l)
    """
    def __init__(self, d_model, d_proj=64, n_perspectives=4, delta=0.5, lam=0.2, temperature=0.1):
        self.delta = delta
        self.lam = lam
        self.temperature = temperature
        # one pooling module per modality per stage (3 stages x 3 modalities)
        self.pool_t = [PerspectivePooling(d_model, d_proj, n_perspectives) for _ in range(3)]
        self.pool_a = [PerspectivePooling(d_model, d_proj, n_perspectives) for _ in range(3)]
        self.pool_m = [PerspectivePooling(d_model, d_proj, n_perspectives) for _ in range(3)]

    def clc_total(self, stage_idx, X_t, X_a, X_m, labels):
        Z_t = self.pool_t[stage_idx](X_t)
        Z_a = self.pool_a[stage_idx](X_a)
        Z_m = self.pool_m[stage_idx](X_m)
        return clc_stage_loss(Z_t, Z_a, Z_m, labels, self.temperature)

    def forward(self, logits, labels,
                stage1_feats, stage2_feats, stage3_feats):
        """
        stageN_feats: (X_t, X_a, X_m) tuples of (B, L, D) tensors taken
        after (1) embedding layer, (2) CAA module, (3) HFE module.
        """
        ce = cross_entropy_loss(logits, labels)
        clc1 = self.clc_total(0, *stage1_feats, labels)
        clc2 = self.clc_total(1, *stage2_feats, labels)
        clc3 = self.clc_total(2, *stage3_feats, labels)
        total = ce * self.delta + (clc1 + clc2 + clc3) * self.lam
        return total, {"ce": ce, "clc1": clc1, "clc2": clc2, "clc3": clc3}
