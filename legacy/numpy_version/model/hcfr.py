"""
model.hcfr
==========
Hierarchical Cross-modal Feature Refinement (Section 3.2 of the paper).

CAA (Cross-modal Adaptive Alignment): Eqs. 10-13
  X_align1_a = CAE(X_e_a, X_e_t, X_e_t)
  X_align1_m = CAE(X_e_m, X_e_t, X_e_t)
  X_align2_a = CAE(X_align1_a, X_e_t, X_e_t)
  X_align2_m = CAE(X_align1_m, X_e_t, X_e_t)

HFE (Hierarchical Feature Enhancement): Eqs. 14-19
  aug1 = SAE(x)  for each modality
  aug2_x = CAE(aug1_x, lower_layer_x, lower_layer_x)
"""
from nptorch.nn import Module, CrossAttentionEncoder, SelfAttentionEncoder


class CAA(Module):
    """Cross-modal Adaptive Alignment: two-stage refinement of audio/MFCC
    against the text representation."""
    def __init__(self, d_model, n_heads):
        self.cae_a1 = CrossAttentionEncoder(d_model, n_heads)
        self.cae_m1 = CrossAttentionEncoder(d_model, n_heads)
        self.cae_a2 = CrossAttentionEncoder(d_model, n_heads)
        self.cae_m2 = CrossAttentionEncoder(d_model, n_heads)

    def forward(self, X_e_a, X_e_m, X_e_t):
        align1_a = self.cae_a1(X_e_a, X_e_t, X_e_t)   # Eq. 10
        align1_m = self.cae_m1(X_e_m, X_e_t, X_e_t)   # Eq. 11
        align2_a = self.cae_a2(align1_a, X_e_t, X_e_t)  # Eq. 13
        align2_m = self.cae_m2(align1_m, X_e_t, X_e_t)  # Eq. 12
        return align2_a, align2_m


class HFE(Module):
    """Hierarchical Feature Enhancement: self-attention then cross-layer
    cross-attention, applied independently to text, audio, and MFCC."""
    def __init__(self, d_model, n_heads):
        self.sae_t = SelfAttentionEncoder(d_model, n_heads)
        self.sae_a = SelfAttentionEncoder(d_model, n_heads)
        self.sae_m = SelfAttentionEncoder(d_model, n_heads)
        self.cae_t = CrossAttentionEncoder(d_model, n_heads)
        self.cae_a = CrossAttentionEncoder(d_model, n_heads)
        self.cae_m = CrossAttentionEncoder(d_model, n_heads)

    def forward(self, X_e_t, align2_a, align2_m):
        aug1_t = self.sae_t(X_e_t)             # Eq. 14
        aug1_a = self.sae_a(align2_a)          # Eq. 15
        aug1_m = self.sae_m(align2_m)          # Eq. 16

        aug2_t = self.cae_t(aug1_t, X_e_t, X_e_t)              # Eq. 17
        aug2_a = self.cae_a(aug1_a, align2_a, align2_a)        # Eq. 18
        aug2_m = self.cae_m(aug1_m, align2_m, align2_m)        # Eq. 19
        return aug2_t, aug2_a, aug2_m


class HCFR(Module):
    """Wraps CAA -> HFE, matching the HCFR box in Fig. 1."""
    def __init__(self, d_model, n_heads):
        self.caa = CAA(d_model, n_heads)
        self.hfe = HFE(d_model, n_heads)

    def forward(self, X_e_a, X_e_m, X_e_t):
        align2_a, align2_m = self.caa(X_e_a, X_e_m, X_e_t)
        aug2_t, aug2_a, aug2_m = self.hfe(X_e_t, align2_a, align2_m)
        # returned in (text, audio, mfcc) order to match the rest of the pipeline
        return aug2_t, aug2_a, aug2_m
