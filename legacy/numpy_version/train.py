"""
train.py
========
Real training loop: reads the .npy features preprocess.py produced,
batches them with dynamic per-batch padding, trains HRDF-MER with the
paper's multi-objective loss (CE + 3-stage CLC), evaluates on a dev split
each epoch, and checkpoints the best model.

Pure NumPy underneath (nptorch) - no torch/tf here. Expect this to be
considerably slower than a GPU framework; start with --limit-sized data
and a small --d_model to sanity check before a full run.

Usage:
    python train.py --train_features data/meld_train/features \
                     --dev_features data/meld_dev/features \
                     --checkpoint_out checkpoints/model.npz \
                     --epochs 10 --batch_size 8 --d_model 128 --n_heads 4
"""
import argparse
import time
import numpy as np

from nptorch.optim import Adam
from model import HRDFMER, MultiObjectiveLoss
from model.labels import MELD_LABEL2IDX
from model.dataset import MELDFeatureDataset, iterate_batches
from model.metrics import compute_metrics
from model.checkpoint import save_checkpoint, load_checkpoint


def evaluate(model, dataset, batch_size):
    all_preds, all_labels = [], []
    for X_e_t, X_e_a, X_mfcc, labels in iterate_batches(dataset, batch_size, shuffle=False):
        logits, *_ = model(X_e_t, X_e_a, X_mfcc)
        preds = np.argmax(logits.data, axis=1)
        all_preds.append(preds)
        all_labels.append(labels)
    if not all_preds:
        return {"wa": 0.0, "ua": 0.0, "wf1": 0.0}
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return compute_metrics(all_preds, all_labels, n_classes=len(MELD_LABEL2IDX))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features", required=True)
    parser.add_argument("--dev_features", default=None, help="Optional; if omitted, no per-epoch eval")
    parser.add_argument("--checkpoint_out", required=True)
    parser.add_argument("--resume_from", default=None, help="Optional checkpoint to resume from")

    parser.add_argument("--d_model", type=int, default=128, help="Internal hidden width (smaller = faster on pure NumPy)")
    parser.add_argument("--embed_dim", type=int, default=768, help="Dim of precomputed RoBERTa/Wav2Vec2 embeddings (768 for -base models)")
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--mfcc_dim", type=int, default=40)
    parser.add_argument("--max_len_text", type=int, default=40, help="Truncate text sequences beyond this many tokens")
    parser.add_argument("--max_len_audio", type=int, default=150, help="Truncate Wav2Vec2 sequences beyond this many frames")
    parser.add_argument("--max_len_mfcc", type=int, default=150, help="Truncate MFCC sequences beyond this many frames (biggest memory lever - lower this first if you hit OOM)")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=12)   # matches paper's batch size
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                         help="Paper uses 4 (effective batch 12*4=48-ish); set >1 to accumulate gradients")
    parser.add_argument("--lr", type=float, default=3e-3)       # matches paper
    parser.add_argument("--delta", type=float, default=0.5)     # matches paper's optimal (Table 7)
    parser.add_argument("--lam", type=float, default=0.2)       # matches paper's optimal (Table 7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("Loading datasets ...")
    train_ds = MELDFeatureDataset(args.train_features, MELD_LABEL2IDX,
                                   max_len_text=args.max_len_text, max_len_audio=args.max_len_audio,
                                   max_len_mfcc=args.max_len_mfcc)
    dev_ds = MELDFeatureDataset(args.dev_features, MELD_LABEL2IDX,
                                 max_len_text=args.max_len_text, max_len_audio=args.max_len_audio,
                                 max_len_mfcc=args.max_len_mfcc) if args.dev_features else None
    print(f"  train: {len(train_ds)} utterances" + (f", dev: {len(dev_ds)} utterances" if dev_ds else ""))
    if len(train_ds) < 2:
        raise SystemExit("Not enough training utterances found - check --train_features points at "
                          "a preprocess.py output directory with .npy files + labels.csv")

    model = HRDFMER(d_model=args.d_model, n_heads=args.n_heads,
                     n_classes=len(MELD_LABEL2IDX), mfcc_dim=args.mfcc_dim, embed_dim=args.embed_dim)
    loss_fn = MultiObjectiveLoss(d_model=args.d_model, delta=args.delta, lam=args.lam)
    params = model.parameters() + loss_fn.parameters()
    opt = Adam(params, lr=args.lr)
    print(f"Model has {sum(p.data.size for p in params):,} learnable parameters")

    start_epoch = 0
    if args.resume_from:
        extra = load_checkpoint(args.resume_from, model, opt)
        if "epoch" in extra:
            start_epoch = int(extra["epoch"]) + 1
            print(f"Resuming from epoch {start_epoch}")

    best_dev_wf1 = -1.0
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        epoch_losses = []
        step = 0

        opt.zero_grad()
        for X_e_t, X_e_a, X_mfcc, labels in iterate_batches(train_ds, args.batch_size, shuffle=True,
                                                              seed=args.seed + epoch):
            logits, stage1, stage2, stage3 = model(X_e_t, X_e_a, X_mfcc)
            total_loss, parts = loss_fn(logits, labels, stage1, stage2, stage3)
            # scale down so grad_accum_steps micro-batches sum to one effective step
            (total_loss * (1.0 / args.grad_accum_steps)).backward()
            step += 1

            if step % args.grad_accum_steps == 0:
                opt.step()
                opt.zero_grad()

            epoch_losses.append(total_loss.data.item())

        if step % args.grad_accum_steps != 0:   # flush a trailing partial accumulation
            opt.step()
            opt.zero_grad()

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        elapsed = time.time() - t0
        msg = f"epoch {epoch:03d}  train_loss={train_loss:.4f}  ({elapsed:.1f}s)"

        if dev_ds is not None and len(dev_ds) >= 2:
            metrics = evaluate(model, dev_ds, args.batch_size)
            msg += f"  dev: WA={metrics['wa']:.4f} UA={metrics['ua']:.4f} WF1={metrics['wf1']:.4f}"
            if metrics["wf1"] > best_dev_wf1:
                best_dev_wf1 = metrics["wf1"]
                save_checkpoint(args.checkpoint_out, model, opt, extra={"epoch": epoch, "dev_wf1": metrics["wf1"]})
                msg += "  (new best, saved)"
        else:
            # no dev set: just save every epoch's end state
            save_checkpoint(args.checkpoint_out, model, opt, extra={"epoch": epoch})

        print(msg)

    print("Training complete.")


if __name__ == "__main__":
    main()
