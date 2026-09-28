"""
export_weights.py
====================
Strips a training checkpoint (model + loss_fn + optimizer state + scheduler
state) down to just the model weights + the args needed to reconstruct the
architecture. Training checkpoints are large (optimizer state roughly
doubles/triples the file size) and the optimizer/scheduler state is useless
once training is done - not something you'd want to commit to a repo or
attach to a release either way.

Use this to produce the file you actually publish (GitHub Release asset or
Hugging Face Hub upload), and keep the full training checkpoint locally only
if you might resume training later.

Usage:
    python export_weights.py --checkpoint checkpoints/model_full_v4.pt --out weights/hrdfmer_meld.pt
"""
import argparse
import os
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Full training checkpoint (from train.py)")
    parser.add_argument("--out", required=True, help="Output path for the weights-only file")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    slim = {
        "model": ckpt["model"],
        "loss_fn": ckpt["loss_fn"],   # kept: small (perspective-pooling heads), and train.py --init_from needs it
        "args": ckpt["args"],
        "epoch": ckpt.get("epoch"),
        "dev_wf1": ckpt.get("dev_wf1"),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(slim, args.out)

    before = os.path.getsize(args.checkpoint) / 1e6
    after = os.path.getsize(args.out) / 1e6
    print(f"{args.checkpoint}: {before:.1f} MB -> {args.out}: {after:.1f} MB")
    print(f"Dropped: optimizer state, LR scheduler state (the bulk of the size - Adam alone roughly "
          f"doubles/triples it). Kept: model + loss_fn weights, so this file works both for "
          f"evaluate.py/predict.py (inference) and as a --init_from warm-start source for train.py.")


if __name__ == "__main__":
    main()
