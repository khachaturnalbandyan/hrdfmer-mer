"""
model.metrics
=============
UA (Unweighted Accuracy = macro-averaged recall), WA (Weighted Accuracy =
overall accuracy), and WF1 (Weighted F1), matching the paper's Section
4.1.4. Implemented from scratch (numpy only) so there's no sklearn
dependency added just for evaluation.
"""
import numpy as np


def compute_metrics(preds: np.ndarray, labels: np.ndarray, n_classes: int) -> dict:
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    per_class_recall = np.zeros(n_classes)
    per_class_f1 = np.zeros(n_classes)
    support = np.zeros(n_classes)

    for c in range(n_classes):
        support[c] = (labels == c).sum()
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()

        recall = tp / support[c] if support[c] > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_recall[c] = recall
        per_class_f1[c] = f1

    wa = (preds == labels).mean() if len(labels) > 0 else 0.0   # overall accuracy
    ua = per_class_recall[support > 0].mean() if (support > 0).any() else 0.0  # macro recall
    total = support.sum()
    wf1 = (per_class_f1 * support).sum() / total if total > 0 else 0.0

    return {"wa": float(wa), "ua": float(ua), "wf1": float(wf1),
            "per_class_recall": per_class_recall, "support": support}
