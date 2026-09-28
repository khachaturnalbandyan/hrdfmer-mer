"""
Backward-pass sanity checks: every trainable parameter must receive a
finite, non-None gradient from a single training step (forward + full
MultiObjectiveLoss + backward). This is the PyTorch-side equivalent of the
regression tests that caught the matmul-broadcast bug in the NumPy engine.
"""
import torch

from model import HRDFMER, MultiObjectiveLoss


def _make_batch(B=4, L_t=5, L_a=6, L_m=7, embed_dim=16, mfcc_dim=13, n_classes=7):
    X_e_t = torch.randn(B, L_t, embed_dim)
    X_e_a = torch.randn(B, L_a, embed_dim)
    X_mfcc = torch.randn(B, L_m, mfcc_dim)
    labels = torch.randint(0, n_classes, (B,))
    return X_e_t, X_e_a, X_mfcc, labels


def test_all_params_get_finite_gradients():
    torch.manual_seed(0)
    d_model, n_classes, embed_dim, mfcc_dim = 16, 7, 16, 13
    model = HRDFMER(d_model=d_model, n_heads=4, n_classes=n_classes, mfcc_dim=mfcc_dim, embed_dim=embed_dim)
    loss_fn = MultiObjectiveLoss(d_model=d_model, d_proj=8, n_perspectives=2)

    X_e_t, X_e_a, X_mfcc, labels = _make_batch(embed_dim=embed_dim, mfcc_dim=mfcc_dim, n_classes=n_classes)

    logits, stage1, stage2, stage3 = model(X_e_t, X_e_a, X_mfcc)
    total_loss, parts = loss_fn(logits, labels, stage1, stage2, stage3)

    assert torch.isfinite(total_loss)
    for name, value in parts.items():
        assert torch.isfinite(value), f"{name} is not finite"

    total_loss.backward()

    missing, non_finite = [], []
    for name, p in list(model.named_parameters()) + list(loss_fn.named_parameters()):
        if p.grad is None:
            missing.append(name)
        elif not torch.isfinite(p.grad).all():
            non_finite.append(name)

    assert not missing, f"parameters with no gradient at all: {missing}"
    assert not non_finite, f"parameters with non-finite gradients: {non_finite}"


def test_gradients_with_padding_masks():
    """Same check, but through the masked path (the one actually used in
    training via collate_fn) - masking must not break backward()."""
    torch.manual_seed(0)
    d_model, n_classes, embed_dim, mfcc_dim = 16, 7, 16, 13
    model = HRDFMER(d_model=d_model, n_heads=4, n_classes=n_classes, mfcc_dim=mfcc_dim, embed_dim=embed_dim)
    loss_fn = MultiObjectiveLoss(d_model=d_model, d_proj=8, n_perspectives=2)

    B, L_t, L_a, L_m = 3, 6, 8, 5
    X_e_t, X_e_a, X_mfcc, labels = _make_batch(B, L_t, L_a, L_m, embed_dim, mfcc_dim, n_classes)

    text_mask = torch.zeros(B, L_t, dtype=torch.bool)
    text_mask[0, 4:] = True
    audio_mask = torch.zeros(B, L_a, dtype=torch.bool)
    audio_mask[1, 6:] = True
    mfcc_mask = torch.zeros(B, L_m, dtype=torch.bool)
    mfcc_mask[2, 3:] = True

    logits, stage1, stage2, stage3 = model(X_e_t, X_e_a, X_mfcc,
                                            text_mask=text_mask, audio_mask=audio_mask, mfcc_mask=mfcc_mask)
    total_loss, _ = loss_fn(logits, labels, stage1, stage2, stage3,
                             text_mask=text_mask, audio_mask=audio_mask, mfcc_mask=mfcc_mask)
    total_loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no gradient with masking active"
        assert torch.isfinite(p.grad).all(), f"{name} got a non-finite gradient with masking active"


def test_optimizer_step_changes_weights():
    """A single Adam step should actually move the weights - catches
    accidental detach()/no_grad() misuse that would silently freeze the
    model while still 'training' without error."""
    torch.manual_seed(0)
    d_model, n_classes, embed_dim, mfcc_dim = 16, 7, 16, 13
    model = HRDFMER(d_model=d_model, n_heads=4, n_classes=n_classes, mfcc_dim=mfcc_dim, embed_dim=embed_dim)
    loss_fn = MultiObjectiveLoss(d_model=d_model, d_proj=8, n_perspectives=2)
    opt = torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-2)

    X_e_t, X_e_a, X_mfcc, labels = _make_batch(embed_dim=embed_dim, mfcc_dim=mfcc_dim, n_classes=n_classes)
    before = {n: p.clone() for n, p in model.named_parameters()}

    logits, stage1, stage2, stage3 = model(X_e_t, X_e_a, X_mfcc)
    total_loss, _ = loss_fn(logits, labels, stage1, stage2, stage3)
    opt.zero_grad()
    total_loss.backward()
    opt.step()

    changed = any(not torch.equal(before[n], p) for n, p in model.named_parameters())
    assert changed, "no model parameter changed after an optimizer step"
