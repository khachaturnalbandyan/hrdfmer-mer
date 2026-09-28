"""
model.hrdfmer
=============
Assembles the full pipeline from Fig. 1:

  [precomputed RoBERTa embeddings]  X_e_t  \
  [precomputed Wav2Vec2 embeddings] X_e_a    >--> HCFR --> CMDF --> concat --> classifier
  [raw MFCC features] -> our own LSTM -> X_e_m /

Everything from the LSTM onward is our own nptorch graph. X_e_t and X_e_a
are expected to already be numpy arrays produced offline by preprocess.py
(which is the only file in this project allowed to import torch/transformers).
"""
import numpy as np
from nptorch.tensor import Tensor, concat
from nptorch.nn import Module, LSTMCell, Linear
from .hcfr import HCFR
from .cmdf import CMDF


class Classifier(Module):
    """'Concatenation -> Classifier' box in Fig. 1: max-pool over the
    sequence dim per modality, concat, then a small feedforward head."""
    def __init__(self, d_model, n_classes, d_hidden=256):
        self.fc1 = Linear(3 * d_model, d_hidden)
        self.fc2 = Linear(d_hidden, n_classes)

    def forward(self, final_t, final_a, final_m):
        pooled_t = self._max_pool(final_t)
        pooled_a = self._max_pool(final_a)
        pooled_m = self._max_pool(final_m)
        O = concat([pooled_t, pooled_a, pooled_m], axis=-1)   # Eq. 30
        h = self.fc1(O).relu()
        return self.fc2(h)

    @staticmethod
    def _max_pool(x: Tensor) -> Tensor:
        # max over sequence length (axis=1): (B, L, D) -> (B, D)
        idx = np.argmax(x.data, axis=1)   # (B, D)
        B, L, D = x.shape
        out = Tensor(np.take_along_axis(x.data, idx[:, None, :], axis=1)[:, 0, :],
                      x.requires_grad, (x,), "maxpool")

        def _backward():
            if x.requires_grad:
                g = np.zeros_like(x.data)
                b_idx = np.arange(B)[:, None].repeat(D, axis=1)
                d_idx = np.arange(D)[None, :].repeat(B, axis=0)
                g[b_idx, idx, d_idx] = out.grad
                x._accumulate(g)
        out._backward = _backward
        return out


class HRDFMER(Module):
    def __init__(self, d_model=768, n_heads=8, n_classes=4, mfcc_dim=40, embed_dim=768):
        """
        d_model:   internal hidden width used by HCFR/CMDF (can be < embed_dim
                   for speed; a projection layer adapts the pretrained
                   embeddings down to this size).
        embed_dim: dimensionality of the PRECOMPUTED RoBERTa/Wav2Vec2
                   embeddings coming out of preprocess.py. Both are 768 for
                   the -base checkpoints the paper uses; only change this if
                   you swap in different pretrained encoders.
        """
        self.d_model = d_model
        self.text_proj = Linear(embed_dim, d_model) if embed_dim != d_model else None
        self.audio_proj = Linear(embed_dim, d_model) if embed_dim != d_model else None
        self.mfcc_lstm = LSTMCell(mfcc_dim, d_model)   # Eq. 3: X_e_m = LSTM(X_mfcc)
        self.hcfr = HCFR(d_model, n_heads)
        self.cmdf = CMDF(d_model, n_heads)
        self.classifier = Classifier(d_model, n_classes)

    def forward(self, X_e_t: Tensor, X_e_a: Tensor, X_mfcc_raw: Tensor):
        """
        X_e_t:      (B, L_t, embed_dim)  precomputed RoBERTa embeddings
        X_e_a:      (B, L_a, embed_dim)  precomputed Wav2Vec2 embeddings
        X_mfcc_raw: (B, L_m, mfcc_dim)   raw MFCC features (e.g. from librosa)
        Returns: logits (B, n_classes), plus the 3 stages of (t,a,m)
                 features needed by the CLC loss.
        """
        if self.text_proj is not None:
            X_e_t = self.text_proj(X_e_t)
        if self.audio_proj is not None:
            X_e_a = self.audio_proj(X_e_a)

        X_e_m = self.mfcc_lstm(X_mfcc_raw)                     # Eq. 3 -> (B, L_m, d_model)
        stage1 = (X_e_t, X_e_a, X_e_m)                          # post-embedding-layer stage

        align2_a, align2_m = self.hcfr.caa(X_e_a, X_e_m, X_e_t)
        stage2 = (X_e_t, align2_a, align2_m)                    # post-CAA stage

        aug2_t, aug2_a, aug2_m = self.hcfr.hfe(X_e_t, align2_a, align2_m)
        stage3 = (aug2_t, aug2_a, aug2_m)                       # post-HFE stage

        final_t, final_a, final_m = self.cmdf(aug2_t, aug2_a, aug2_m)
        logits = self.classifier(final_t, final_a, final_m)

        return logits, stage1, stage2, stage3
