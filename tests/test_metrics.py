"""
compute_metrics is hand-rolled (no sklearn), so it gets its own tests
against known-by-hand values rather than trusting the from-scratch math.
"""
import numpy as np

from model.metrics import compute_metrics


def test_perfect_predictions():
    labels = np.array([0, 0, 1, 1, 2, 2])
    preds = labels.copy()
    m = compute_metrics(preds, labels, n_classes=3)
    assert m["wa"] == 1.0
    assert m["ua"] == 1.0
    assert m["wf1"] == 1.0


def test_all_wrong_two_class():
    labels = np.array([0, 0, 1, 1])
    preds = np.array([1, 1, 0, 0])
    m = compute_metrics(preds, labels, n_classes=2)
    assert m["wa"] == 0.0
    assert m["ua"] == 0.0
    assert m["wf1"] == 0.0


def test_known_hand_computed_case():
    # class 0: 3 samples, class 1: 1 sample (imbalanced, like MELD).
    labels = np.array([0, 0, 0, 1])
    preds = np.array([0, 0, 1, 1])
    m = compute_metrics(preds, labels, n_classes=2)

    # class 0: TP=2, FN=1, FP=0 -> recall=2/3, precision=1.0, f1=2*1*(2/3)/(1+2/3)=0.8
    # class 1: TP=1, FN=0, FP=1 -> recall=1.0, precision=0.5, f1=2*0.5*1/(1.5)=0.6667
    assert abs(m["per_class_recall"][0] - 2 / 3) < 1e-6
    assert abs(m["per_class_recall"][1] - 1.0) < 1e-6
    assert abs(m["wa"] - 0.75) < 1e-6            # 3/4 correct overall
    assert abs(m["ua"] - (2 / 3 + 1.0) / 2) < 1e-6   # macro recall
    expected_wf1 = (0.8 * 3 + (2 / 3) * 1) / 4
    assert abs(m["wf1"] - expected_wf1) < 1e-6


def test_missing_class_does_not_crash_or_count_toward_ua():
    """A class with zero support in this split (e.g. 'fear'/'disgust' in a
    small MELD subset) must not blow up compute_metrics or be averaged
    into UA as a spurious 0."""
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 1, 1, 1])
    m = compute_metrics(preds, labels, n_classes=3)   # class 2 has 0 support
    assert m["support"][2] == 0
    assert np.isfinite(m["ua"])
    assert np.isfinite(m["wf1"])
