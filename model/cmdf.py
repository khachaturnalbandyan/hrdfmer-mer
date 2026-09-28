"""
model.cmdf (PyTorch version)
==============================
Cascaded Multimodal Dynamic Fusion (Section 3.3 / Fig. 2). Same cascade
structure as the NumPy version: three stages of cross-attention (text<->audio
at stages 1-2, audio<->mfcc cross-calibration + text self-attention at stage
3), then a two-stage gated fusion per modality (Eqs. 26-29).
"""
import torch
import torch.nn as nn
from .hcfr import CrossAttentionEncoder, SelfAttentionEncoder


class GatedFusion(nn.Module):
    """Eqs. 26-29: alpha = sigmoid(W_gate[x_secondary; x_primary] + b_gate);
    out = alpha * x_primary + (1-alpha) * x_secondary."""
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, d_model)

    def forward(self, x_primary, x_secondary):
        gate_in = torch.cat([x_secondary, x_primary], dim=-1)
        alpha = torch.sigmoid(self.gate(gate_in))
        return alpha * x_primary + (1 - alpha) * x_secondary


class CMDF(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.cae_t1 = CrossAttentionEncoder(d_model, n_heads)
        self.cae_a1 = CrossAttentionEncoder(d_model, n_heads)
        self.cae_m1 = CrossAttentionEncoder(d_model, n_heads)
        self.cae_a2 = CrossAttentionEncoder(d_model, n_heads)
        self.cae_m2 = CrossAttentionEncoder(d_model, n_heads)
        self.sae_t3 = SelfAttentionEncoder(d_model, n_heads)
        self.cae_a3 = CrossAttentionEncoder(d_model, n_heads)
        self.cae_m3 = CrossAttentionEncoder(d_model, n_heads)

        self.gate_t_1 = GatedFusion(d_model)
        self.gate_t_2 = GatedFusion(d_model)
        self.gate_a_1 = GatedFusion(d_model)
        self.gate_a_2 = GatedFusion(d_model)
        self.gate_m_1 = GatedFusion(d_model)
        self.gate_m_2 = GatedFusion(d_model)

    def forward(self, aug2_t, aug2_a, aug2_m, text_mask=None, audio_mask=None, mfcc_mask=None):
        # ---- stage 1 (Eqs. 20-22) ----
        cross1_t = self.cae_t1(aug2_t, aug2_a, aug2_a, key_padding_mask=audio_mask)
        cross1_a = self.cae_a1(aug2_a, aug2_t, aug2_t, key_padding_mask=text_mask)
        cross1_m = self.cae_m1(aug2_m, aug2_t, aug2_t, key_padding_mask=text_mask)

        # ---- stage 2 ----
        cross2_a = self.cae_a2(cross1_a, cross1_t, cross1_t, key_padding_mask=text_mask)
        cross2_m = self.cae_m2(cross1_m, cross1_t, cross1_t, key_padding_mask=text_mask)
        cross2_t = cross1_t

        # ---- stage 3 (Eqs. 23-25) ----
        cross3_t = self.sae_t3(cross2_t, key_padding_mask=text_mask)
        cross3_a = self.cae_a3(cross2_a, cross2_m, cross2_m, key_padding_mask=mfcc_mask)
        cross3_m = self.cae_m3(cross2_m, cross2_a, cross2_a, key_padding_mask=audio_mask)

        # ---- two-stage gated fusion (Eqs. 26-29) ----
        final_t = self._gated_fusion(aug2_t, cross3_t, self.gate_t_1, self.gate_t_2)
        final_a = self._gated_fusion(aug2_a, cross3_a, self.gate_a_1, self.gate_a_2)
        final_m = self._gated_fusion(aug2_m, cross3_m, self.gate_m_1, self.gate_m_2)
        return final_t, final_a, final_m

    @staticmethod
    def _gated_fusion(aug2, cross3, gate1, gate2):
        o_gate = gate1(aug2, cross3)     # Eq. 26-27
        o_final = gate2(aug2, o_gate)    # Eq. 28-29
        return o_final
