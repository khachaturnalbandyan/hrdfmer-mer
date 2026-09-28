"""
model.hrdfmer (PyTorch version)
==================================
Assembles the full pipeline from Fig. 1, in real torch.nn, GPU-capable.
Same structure as the NumPy version, PLUS proper padding-mask support
throughout attention (something the pure-NumPy version didn't have) -
padded timesteps are now correctly excluded from attention and from the
CLC loss's perspective pooling, and max-pooling ignores padding too.
"""
import torch
import torch.nn as nn
from .hcfr import HCFR
from .cmdf import CMDF


class Classifier(nn.Module):
    """'Concatenation -> Classifier' box in Fig. 1: masked max-pool over
    the sequence dim per modality, concat, then a feedforward head."""
    def __init__(self, d_model, n_classes, d_hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(3 * d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, n_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, final_t, final_a, final_m, text_mask=None, audio_mask=None, mfcc_mask=None):
        pooled_t = self._masked_max_pool(final_t, text_mask)
        pooled_a = self._masked_max_pool(final_a, audio_mask)
        pooled_m = self._masked_max_pool(final_m, mfcc_mask)
        O = torch.cat([pooled_t, pooled_a, pooled_m], dim=-1)   # Eq. 30
        h = torch.relu(self.fc1(O))
        h = self.dropout(h)
        return self.fc2(h)

    @staticmethod
    def _masked_max_pool(x, key_padding_mask=None):
        # x: (B, L, D); key_padding_mask: (B, L) True where PADDING
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf"))
        pooled, _ = x.max(dim=1)
        return pooled


class HRDFMER(nn.Module):
    def __init__(self, d_model=768, n_heads=8, n_classes=4, mfcc_dim=40, embed_dim=768):
        """
        d_model:   internal hidden width used by HCFR/CMDF.
        embed_dim: dimensionality of the PRECOMPUTED RoBERTa/Wav2Vec2
                   embeddings (768 for -base models).
        """
        super().__init__()
        self.d_model = d_model
        self.text_proj = nn.Linear(embed_dim, d_model) if embed_dim != d_model else nn.Identity()
        self.audio_proj = nn.Linear(embed_dim, d_model) if embed_dim != d_model else nn.Identity()
        self.mfcc_lstm = nn.LSTM(mfcc_dim, d_model, batch_first=True)   # Eq. 3: X_e_m = LSTM(X_mfcc)
        self.hcfr = HCFR(d_model, n_heads)
        self.cmdf = CMDF(d_model, n_heads)
        self.classifier = Classifier(d_model, n_classes)

    def forward(self, X_e_t, X_e_a, X_mfcc_raw, text_mask=None, audio_mask=None, mfcc_mask=None):
        """
        X_e_t:      (B, L_t, embed_dim)  precomputed RoBERTa embeddings
        X_e_a:      (B, L_a, embed_dim)  precomputed Wav2Vec2 embeddings
        X_mfcc_raw: (B, L_m, mfcc_dim)   raw MFCC features
        *_mask:     (B, L) bool, True at PADDING positions (matches
                    torch's key_padding_mask convention). Optional - pass
                    None to skip masking (e.g. for un-padded single-batch use).
        Returns: logits (B, n_classes), plus the 3 stages of (t,a,m)
                 features needed by the CLC loss.
        """
        X_e_t = self.text_proj(X_e_t)
        X_e_a = self.audio_proj(X_e_a)

        X_e_m, _ = self.mfcc_lstm(X_mfcc_raw)                   # Eq. 3 -> (B, L_m, d_model)
        stage1 = (X_e_t, X_e_a, X_e_m)

        align2_a, align2_m = self.hcfr.caa(X_e_a, X_e_m, X_e_t, text_mask=text_mask)
        stage2 = (X_e_t, align2_a, align2_m)

        aug2_t, aug2_a, aug2_m = self.hcfr.hfe(X_e_t, align2_a, align2_m,
                                                text_mask=text_mask, audio_mask=audio_mask, mfcc_mask=mfcc_mask)
        stage3 = (aug2_t, aug2_a, aug2_m)

        final_t, final_a, final_m = self.cmdf(aug2_t, aug2_a, aug2_m,
                                               text_mask=text_mask, audio_mask=audio_mask, mfcc_mask=mfcc_mask)
        logits = self.classifier(final_t, final_a, final_m,
                                  text_mask=text_mask, audio_mask=audio_mask, mfcc_mask=mfcc_mask)

        return logits, stage1, stage2, stage3
