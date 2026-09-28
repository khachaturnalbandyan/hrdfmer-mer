"""
evaluate.py
=============
Loads a trained checkpoint and reports WA/UA/WF1 on a given features split
(typically the test split, for a final, unbiased comparison against the
paper's reported numbers - never used for model selection during training,
unlike the dev split).

Usage:
    python evaluate.py --features data/meld_test_full/features --checkpoint checkpoints/model_full_v4.pt
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import HRDFMER
from model.labels import MELD_LABEL2IDX, MELD_IDX2LABEL
from model.dataset import MELDFeatureDataset, collate_fn
from model.metrics import compute_metrics


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to a preprocess.py output dir (e.g. the test split's features/)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=12)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt["args"]
    print(f"Checkpoint: epoch {ckpt.get('epoch', '?')}, dev_wf1 at save time = {ckpt.get('dev_wf1', '?')}")

    ds = MELDFeatureDataset(args.features, MELD_LABEL2IDX,
                             max_len_text=train_args.get("max_len_text", 64),
                             max_len_audio=train_args.get("max_len_audio", 300),
                             max_len_mfcc=train_args.get("max_len_mfcc", 300))
    print(f"Evaluating on {len(ds)} utterances from {args.features}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = HRDFMER(d_model=train_args["d_model"], n_heads=train_args["n_heads"],
                     n_classes=len(MELD_LABEL2IDX), mfcc_dim=train_args["mfcc_dim"],
                     embed_dim=train_args["embed_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_preds, all_labels = [], []
    for X_e_t, X_e_a, X_mfcc, text_mask, audio_mask, mfcc_mask, labels in loader:
        X_e_t, X_e_a, X_mfcc = X_e_t.to(device), X_e_a.to(device), X_mfcc.to(device)
        text_mask, audio_mask, mfcc_mask = text_mask.to(device), audio_mask.to(device), mfcc_mask.to(device)
        logits, *_ = model(X_e_t, X_e_a, X_mfcc, text_mask, audio_mask, mfcc_mask)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_preds, all_labels, n_classes=len(MELD_LABEL2IDX))

    print(f"\n=== Results on {args.features} ===")
    print(f"WA  (Weighted Accuracy) = {metrics['wa']*100:.2f}%")
    print(f"UA  (Unweighted Accuracy, macro recall) = {metrics['ua']*100:.2f}%")
    print(f"WF1 (Weighted F1) = {metrics['wf1']*100:.2f}%")
    print(f"\nPer-class recall (support):")
    for idx, label in MELD_IDX2LABEL.items():
        print(f"  {label:10s} recall={metrics['per_class_recall'][idx]*100:5.2f}%  "
              f"(n={int(metrics['support'][idx])})")

    print(f"\nPaper's reported HRDF-MER on MELD test: WA=64.76%, WF1=63.02%")


if __name__ == "__main__":
    main()
