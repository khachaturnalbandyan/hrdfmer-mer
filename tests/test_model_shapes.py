"""
Sanity checks that HRDFMER (and its HCFR/CMDF sub-blocks) produce the
shapes the rest of the pipeline (Classifier, MultiObjectiveLoss, train.py)
assumes - the first thing to break silently after any refactor.
"""
import torch

from model import HRDFMER


def _make_batch(B=3, L_t=5, L_a=7, L_m=9, embed_dim=32, mfcc_dim=13):
    X_e_t = torch.randn(B, L_t, embed_dim)
    X_e_a = torch.randn(B, L_a, embed_dim)
    X_mfcc = torch.randn(B, L_m, mfcc_dim)
    return X_e_t, X_e_a, X_mfcc


def test_forward_shapes_no_mask():
    d_model, n_classes, embed_dim, mfcc_dim = 32, 7, 32, 13
    model = HRDFMER(d_model=d_model, n_heads=4, n_classes=n_classes, mfcc_dim=mfcc_dim, embed_dim=embed_dim)
    X_e_t, X_e_a, X_mfcc = _make_batch(embed_dim=embed_dim, mfcc_dim=mfcc_dim)
    B, L_t, L_a, L_m = X_e_t.shape[0], X_e_t.shape[1], X_e_a.shape[1], X_mfcc.shape[1]

    logits, stage1, stage2, stage3 = model(X_e_t, X_e_a, X_mfcc)

    assert logits.shape == (B, n_classes)
    for stage in (stage1, stage2, stage3):
        t, a, m = stage
        assert t.shape == (B, L_t, d_model)
        assert a.shape == (B, L_a, d_model)
        assert m.shape == (B, L_m, d_model)


def test_forward_shapes_with_padding_masks():
    """Same as above, but with ragged sequences padded + masked - the real
    usage pattern from collate_fn."""
    d_model, n_classes, embed_dim, mfcc_dim = 16, 7, 16, 13
    model = HRDFMER(d_model=d_model, n_heads=4, n_classes=n_classes, mfcc_dim=mfcc_dim, embed_dim=embed_dim)

    B, L_t, L_a, L_m = 4, 6, 8, 10
    X_e_t, X_e_a, X_mfcc = _make_batch(B, L_t, L_a, L_m, embed_dim, mfcc_dim)

    # Fake "true lengths" shorter than the padded length, per batch item.
    text_mask = torch.zeros(B, L_t, dtype=torch.bool)
    text_mask[:, 4:] = True   # last 2 timesteps are padding for everyone
    audio_mask = torch.zeros(B, L_a, dtype=torch.bool)
    audio_mask[0, 5:] = True
    mfcc_mask = torch.zeros(B, L_m, dtype=torch.bool)
    mfcc_mask[1, 7:] = True

    logits, *_ = model(X_e_t, X_e_a, X_mfcc, text_mask=text_mask, audio_mask=audio_mask, mfcc_mask=mfcc_mask)
    assert logits.shape == (B, n_classes)
    assert torch.isfinite(logits).all()


def test_embed_dim_decoupled_from_d_model():
    """Regression test for the text_proj/audio_proj bug: embed_dim (the
    precomputed RoBERTa/Wav2Vec2 width) must NOT be required to equal
    d_model (the internal HCFR/CMDF width)."""
    model = HRDFMER(d_model=32, n_heads=4, n_classes=7, mfcc_dim=13, embed_dim=768)
    X_e_t = torch.randn(2, 5, 768)
    X_e_a = torch.randn(2, 5, 768)
    X_mfcc = torch.randn(2, 5, 13)
    logits, *_ = model(X_e_t, X_e_a, X_mfcc)
    assert logits.shape == (2, 7)


def test_single_utterance_batch():
    """batch_size=1, no padding at all - the most common inference-time shape."""
    model = HRDFMER(d_model=16, n_heads=2, n_classes=7, mfcc_dim=13, embed_dim=16)
    X_e_t, X_e_a, X_mfcc = _make_batch(B=1, embed_dim=16, mfcc_dim=13)
    logits, *_ = model(X_e_t, X_e_a, X_mfcc)
    assert logits.shape == (1, 7)
