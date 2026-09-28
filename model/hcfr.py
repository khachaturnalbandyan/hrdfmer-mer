"""
model.hcfr (PyTorch version)
=============================
Hierarchical Cross-modal Feature Refinement (Section 3.2 of the paper).
Same architecture as the from-scratch NumPy version, now built on real
torch.nn so it runs on GPU.

CAA (Cross-modal Adaptive Alignment): Eqs. 10-13
HFE (Hierarchical Feature Enhancement): Eqs. 14-19
"""
import torch
import torch.nn as nn


class CrossAttentionEncoder(nn.Module):
    """CAE from the paper: X_align = LN(MHCA(Q,K,V) + Q)  (Eq. 9-10)."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.mhca = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, query, key, value, key_padding_mask=None):
        attn_out, _ = self.mhca(query, key, value, key_padding_mask=key_padding_mask)
        return self.ln(attn_out + query)


class SelfAttentionEncoder(nn.Module):
    """SAE: same as CAE but query=key=value=x (Eq. 14-16)."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.cae = CrossAttentionEncoder(d_model, n_heads, dropout)

    def forward(self, x, key_padding_mask=None):
        return self.cae(x, x, x, key_padding_mask=key_padding_mask)


class CAA(nn.Module):
    """Cross-modal Adaptive Alignment: two-stage refinement of audio/MFCC
    against the text representation."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.cae_a1 = CrossAttentionEncoder(d_model, n_heads)
        self.cae_m1 = CrossAttentionEncoder(d_model, n_heads)
        self.cae_a2 = CrossAttentionEncoder(d_model, n_heads)
        self.cae_m2 = CrossAttentionEncoder(d_model, n_heads)

    def forward(self, X_e_a, X_e_m, X_e_t, text_mask=None):
        align1_a = self.cae_a1(X_e_a, X_e_t, X_e_t, key_padding_mask=text_mask)   # Eq. 10
        align1_m = self.cae_m1(X_e_m, X_e_t, X_e_t, key_padding_mask=text_mask)   # Eq. 11
        align2_a = self.cae_a2(align1_a, X_e_t, X_e_t, key_padding_mask=text_mask)  # Eq. 13
        align2_m = self.cae_m2(align1_m, X_e_t, X_e_t, key_padding_mask=text_mask)  # Eq. 12
        return align2_a, align2_m


class HFE(nn.Module):
    """Hierarchical Feature Enhancement: self-attention then cross-layer
    cross-attention, applied independently to text, audio, and MFCC."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.sae_t = SelfAttentionEncoder(d_model, n_heads)
        self.sae_a = SelfAttentionEncoder(d_model, n_heads)
        self.sae_m = SelfAttentionEncoder(d_model, n_heads)
        self.cae_t = CrossAttentionEncoder(d_model, n_heads)
        self.cae_a = CrossAttentionEncoder(d_model, n_heads)
        self.cae_m = CrossAttentionEncoder(d_model, n_heads)

    def forward(self, X_e_t, align2_a, align2_m, text_mask=None, audio_mask=None, mfcc_mask=None):
        aug1_t = self.sae_t(X_e_t, key_padding_mask=text_mask)             # Eq. 14
        aug1_a = self.sae_a(align2_a, key_padding_mask=audio_mask)         # Eq. 15
        aug1_m = self.sae_m(align2_m, key_padding_mask=mfcc_mask)          # Eq. 16

        aug2_t = self.cae_t(aug1_t, X_e_t, X_e_t, key_padding_mask=text_mask)              # Eq. 17
        aug2_a = self.cae_a(aug1_a, align2_a, align2_a, key_padding_mask=audio_mask)        # Eq. 18
        aug2_m = self.cae_m(aug1_m, align2_m, align2_m, key_padding_mask=mfcc_mask)         # Eq. 19
        return aug2_t, aug2_a, aug2_m


class HCFR(nn.Module):
    """Wraps CAA -> HFE, matching the HCFR box in Fig. 1."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.caa = CAA(d_model, n_heads)
        self.hfe = HFE(d_model, n_heads)

    def forward(self, X_e_a, X_e_m, X_e_t, text_mask=None, audio_mask=None, mfcc_mask=None):
        align2_a, align2_m = self.caa(X_e_a, X_e_m, X_e_t, text_mask=text_mask)
        aug2_t, aug2_a, aug2_m = self.hfe(X_e_t, align2_a, align2_m,
                                           text_mask=text_mask, audio_mask=audio_mask, mfcc_mask=mfcc_mask)
        return aug2_t, aug2_a, aug2_m
