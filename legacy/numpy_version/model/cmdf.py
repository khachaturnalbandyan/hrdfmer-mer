"""
model.cmdf
==========
Cascaded Multimodal Dynamic Fusion (Section 3.3 / Fig. 2).

Three cascaded CAE stages (cross1 -> cross2 -> cross3), where:
  - text: cross-attends to audio at stage 1-2, then self-attends at stage 3
  - audio/mfcc: cross-attend to text at stage 1-2, then cross-attend to
    EACH OTHER at stage 3 (Eqs. 24-25) - because MFCC captures timbre and
    audio captures prosody, letting them calibrate each other at the end.

Then a two-stage gated fusion (Eqs. 26-29) blends the "aug2" (post-HFE)
features with the "cross3" (post-cascade) features per modality.
"""
from nptorch.nn import Module, CrossAttentionEncoder, SelfAttentionEncoder, GatedFusion


class CMDF(Module):
    def __init__(self, d_model, n_heads):
        # stage 1: cross with the "other" modality (Eqs. 20-22)
        self.cae_t1 = CrossAttentionEncoder(d_model, n_heads)
        self.cae_a1 = CrossAttentionEncoder(d_model, n_heads)
        self.cae_m1 = CrossAttentionEncoder(d_model, n_heads)
        # stage 2: cross again with text (used inside stage-2 calc below)
        self.cae_t2 = CrossAttentionEncoder(d_model, n_heads)
        self.cae_a2 = CrossAttentionEncoder(d_model, n_heads)
        self.cae_m2 = CrossAttentionEncoder(d_model, n_heads)
        # stage 3: text self-attends; audio/mfcc cross-attend to each other (Eqs. 23-25)
        self.sae_t3 = SelfAttentionEncoder(d_model, n_heads)
        self.cae_a3 = CrossAttentionEncoder(d_model, n_heads)
        self.cae_m3 = CrossAttentionEncoder(d_model, n_heads)

        # two-stage gated fusion per modality (Eqs. 26-29)
        self.gate_t_1 = GatedFusion(d_model)
        self.gate_t_2 = GatedFusion(d_model)
        self.gate_a_1 = GatedFusion(d_model)
        self.gate_a_2 = GatedFusion(d_model)
        self.gate_m_1 = GatedFusion(d_model)
        self.gate_m_2 = GatedFusion(d_model)

    def forward(self, aug2_t, aug2_a, aug2_m):
        # ---- stage 1 (Eqs. 20-22) ----
        cross1_t = self.cae_t1(aug2_t, aug2_a, aug2_a)
        cross1_a = self.cae_a1(aug2_a, aug2_t, aug2_t)
        cross1_m = self.cae_m1(aug2_m, aug2_t, aug2_t)

        # ---- stage 2 ----
        cross2_a = self.cae_a2(cross1_a, cross1_t, cross1_t)
        cross2_m = self.cae_m2(cross1_m, cross1_t, cross1_t)
        cross2_t = cross1_t  # text's stage-2 representation feeds stage 3 directly (Fig. 2 middle column)

        # ---- stage 3 (Eqs. 23-25) ----
        cross3_t = self.sae_t3(cross2_t)
        cross3_a = self.cae_a3(cross2_a, cross2_m, cross2_m)
        cross3_m = self.cae_m3(cross2_m, cross2_a, cross2_a)

        # ---- two-stage gated fusion (Eqs. 26-29), one per modality ----
        final_t = self._gated_fusion(aug2_t, cross3_t, self.gate_t_1, self.gate_t_2)
        final_a = self._gated_fusion(aug2_a, cross3_a, self.gate_a_1, self.gate_a_2)
        final_m = self._gated_fusion(aug2_m, cross3_m, self.gate_m_1, self.gate_m_2)
        return final_t, final_a, final_m

    @staticmethod
    def _gated_fusion(aug2, cross3, gate1: GatedFusion, gate2: GatedFusion):
        # O_gate = alpha * aug2 + (1-alpha) * cross3          (Eq. 26-27)
        o_gate = gate1(aug2, cross3)
        # O_final = beta * aug2 + (1-beta) * O_gate            (Eq. 28-29)
        o_final = gate2(aug2, o_gate)
        return o_final
