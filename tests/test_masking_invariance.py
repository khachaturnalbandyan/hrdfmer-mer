"""
Padding-mask correctness: appending extra padded timesteps to a sequence
(with the padding correctly marked in the mask) must not change the
model's output for the real, non-padded timesteps. This is the property
the from-scratch NumPy version was missing and the PyTorch rewrite exists
to fix - so it's the most important regression test in this repo.
"""
import torch

from model import HRDFMER
from model.losses import PerspectivePooling


def test_extra_padding_does_not_change_logits():
    torch.manual_seed(0)
    d_model, n_classes, embed_dim, mfcc_dim = 16, 7, 16, 13
    model = HRDFMER(d_model=d_model, n_heads=4, n_classes=n_classes, mfcc_dim=mfcc_dim, embed_dim=embed_dim)
    model.eval()   # disable dropout so the comparison is exact

    B, L_t, L_a, L_m = 2, 5, 6, 7
    X_e_t = torch.randn(B, L_t, embed_dim)
    X_e_a = torch.randn(B, L_a, embed_dim)
    X_mfcc = torch.randn(B, L_m, mfcc_dim)

    with torch.no_grad():
        logits_unpadded, *_ = model(X_e_t, X_e_a, X_mfcc)

        # Now pad every modality with 3 extra garbage timesteps and mask them out.
        pad_len = 3
        X_e_t_padded = torch.cat([X_e_t, torch.randn(B, pad_len, embed_dim)], dim=1)
        X_e_a_padded = torch.cat([X_e_a, torch.randn(B, pad_len, embed_dim)], dim=1)
        X_mfcc_padded = torch.cat([X_mfcc, torch.randn(B, pad_len, mfcc_dim)], dim=1)

        text_mask = torch.zeros(B, L_t + pad_len, dtype=torch.bool)
        text_mask[:, L_t:] = True
        audio_mask = torch.zeros(B, L_a + pad_len, dtype=torch.bool)
        audio_mask[:, L_a:] = True
        mfcc_mask = torch.zeros(B, L_m + pad_len, dtype=torch.bool)
        mfcc_mask[:, L_m:] = True

        logits_padded, *_ = model(X_e_t_padded, X_e_a_padded, X_mfcc_padded,
                                   text_mask=text_mask, audio_mask=audio_mask, mfcc_mask=mfcc_mask)

    # nn.MultiheadAttention + masked max-pool should make the garbage
    # padding invisible to the output, up to floating-point noise.
    assert torch.allclose(logits_unpadded, logits_padded, atol=1e-4), \
        "padding leaked into the model output - key_padding_mask is not being respected somewhere"


def test_perspective_pooling_ignores_padding():
    torch.manual_seed(0)
    d_model, d_proj = 8, 4
    pool = PerspectivePooling(d_model, d_proj, n_perspectives=2)
    pool.eval()

    B, L = 2, 5
    x = torch.randn(B, L, d_model)
    with torch.no_grad():
        z_unpadded = pool(x)

        pad_len = 4
        x_padded = torch.cat([x, torch.randn(B, pad_len, d_model)], dim=1)
        mask = torch.zeros(B, L + pad_len, dtype=torch.bool)
        mask[:, L:] = True
        z_padded = pool(x_padded, key_padding_mask=mask)

    assert torch.allclose(z_unpadded, z_padded, atol=1e-4)
