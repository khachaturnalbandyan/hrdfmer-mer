"""
train.py (PyTorch version)
=============================
Real training loop, on real torch, GPU-capable. Reads the .npy features
preprocess.py produces, batches with a DataLoader (dynamic padding +
attention masks via collate_fn), trains with the paper's CE+CLC loss,
evaluates on a dev split each epoch, checkpoints the best model.

Usage:
    python train.py --train_features data/meld_train/features \
                     --dev_features data/meld_dev/features \
                     --checkpoint_out checkpoints/model.pt \
                     --epochs 20 --batch_size 12 --d_model 768 --n_heads 8
"""
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import HRDFMER, MultiObjectiveLoss
from model.labels import MELD_LABEL2IDX
from model.dataset import MELDFeatureDataset, collate_fn
from model.metrics import compute_metrics


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():   # Apple Silicon, in case
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for X_e_t, X_e_a, X_mfcc, text_mask, audio_mask, mfcc_mask, labels in loader:
        X_e_t, X_e_a, X_mfcc = X_e_t.to(device), X_e_a.to(device), X_mfcc.to(device)
        text_mask, audio_mask, mfcc_mask = text_mask.to(device), audio_mask.to(device), mfcc_mask.to(device)
        logits, *_ = model(X_e_t, X_e_a, X_mfcc, text_mask, audio_mask, mfcc_mask)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    model.train()
    if not all_preds:
        return {"wa": 0.0, "ua": 0.0, "wf1": 0.0}
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return compute_metrics(all_preds, all_labels, n_classes=len(MELD_LABEL2IDX))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features", required=True)
    parser.add_argument("--dev_features", default=None)
    parser.add_argument("--checkpoint_out", required=True)
    parser.add_argument("--resume_from", default=None, help="Continue the SAME run: restores model+optimizer+scheduler+epoch")
    parser.add_argument("--init_from", default=None, help="WARM START: load only model weights from this checkpoint, then train fresh (new optimizer, epoch 0, new LR schedule). Mutually exclusive with --resume_from.")

    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--mfcc_dim", type=int, default=40)
    parser.add_argument("--max_len_text", type=int, default=64)
    parser.add_argument("--max_len_audio", type=int, default=300)
    parser.add_argument("--max_len_mfcc", type=int, default=300)

    parser.add_argument("--epochs", type=int, default=50)          # matches paper
    parser.add_argument("--batch_size", type=int, default=12)      # matches paper
    parser.add_argument("--grad_accum_steps", type=int, default=4)  # matches paper (12*4=48 effective)
    parser.add_argument("--lr", type=float, default=3e-3)          # matches paper
    parser.add_argument("--lr_min", type=float, default=1e-5, help="Cosine decay floor")
    parser.add_argument("--no_lr_schedule", action="store_true", help="Disable cosine LR decay, keep --lr constant")
    parser.add_argument("--delta", type=float, default=0.5)        # matches paper's optimum (Table 7)
    parser.add_argument("--lam", type=float, default=0.2)          # matches paper's optimum (Table 7)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    print("Loading datasets ...")
    train_ds = MELDFeatureDataset(args.train_features, MELD_LABEL2IDX,
                                   max_len_text=args.max_len_text, max_len_audio=args.max_len_audio,
                                   max_len_mfcc=args.max_len_mfcc)
    dev_ds = MELDFeatureDataset(args.dev_features, MELD_LABEL2IDX,
                                 max_len_text=args.max_len_text, max_len_audio=args.max_len_audio,
                                 max_len_mfcc=args.max_len_mfcc) if args.dev_features else None
    print(f"  train: {len(train_ds)} utterances" + (f", dev: {len(dev_ds)} utterances" if dev_ds else ""))
    if len(train_ds) < 2:
        raise SystemExit("Not enough training utterances found.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=args.num_workers, drop_last=True)
    dev_loader = (DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
                  if dev_ds and len(dev_ds) >= 2 else None)

    model = HRDFMER(d_model=args.d_model, n_heads=args.n_heads, n_classes=len(MELD_LABEL2IDX),
                     mfcc_dim=args.mfcc_dim, embed_dim=args.embed_dim).to(device)
    loss_fn = MultiObjectiveLoss(d_model=args.d_model, delta=args.delta, lam=args.lam).to(device)

    params = list(model.parameters()) + list(loss_fn.parameters())
    opt = Adam(params, lr=args.lr)
    scheduler = None if args.no_lr_schedule else CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr_min)
    n_params = sum(p.numel() for p in params)
    print(f"Model has {n_params:,} learnable parameters")
    if scheduler is not None:
        print(f"LR schedule: cosine decay from {args.lr} to {args.lr_min} over {args.epochs} epochs")

    if args.resume_from and args.init_from:
        raise SystemExit("--resume_from and --init_from are mutually exclusive - pick one.")

    start_epoch = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        loss_fn.load_state_dict(ckpt["loss_fn"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", -1) + 1
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        print(f"Resumed from epoch {start_epoch} (model+optimizer+scheduler restored)")
    elif args.init_from:
        ckpt = torch.load(args.init_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        loss_fn.load_state_dict(ckpt["loss_fn"])
        print(f"Warm-started model weights from {args.init_from} "
              f"(epoch {ckpt.get('epoch', '?')}, dev_wf1={ckpt.get('dev_wf1', '?')}) "
              f"- optimizer and LR schedule start fresh from epoch 0")

    best_dev_wf1 = -1.0
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        epoch_losses = []
        opt.zero_grad()

        for step, (X_e_t, X_e_a, X_mfcc, text_mask, audio_mask, mfcc_mask, labels) in enumerate(train_loader):
            X_e_t, X_e_a, X_mfcc = X_e_t.to(device), X_e_a.to(device), X_mfcc.to(device)
            text_mask, audio_mask, mfcc_mask = text_mask.to(device), audio_mask.to(device), mfcc_mask.to(device)
            labels = labels.to(device)

            logits, stage1, stage2, stage3 = model(X_e_t, X_e_a, X_mfcc, text_mask, audio_mask, mfcc_mask)
            total_loss, parts = loss_fn(logits, labels, stage1, stage2, stage3,
                                         text_mask, audio_mask, mfcc_mask)
            (total_loss / args.grad_accum_steps).backward()

            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                opt.step()
                opt.zero_grad()

            epoch_losses.append(total_loss.item())

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        cur_lr = opt.param_groups[0]["lr"]
        msg = f"epoch {epoch:03d}  train_loss={train_loss:.4f}  lr={cur_lr:.2e}  ({elapsed:.1f}s)"

        ckpt_dict = {"model": model.state_dict(), "loss_fn": loss_fn.state_dict(),
                     "optimizer": opt.state_dict(), "epoch": epoch, "args": vars(args)}
        if scheduler is not None:
            ckpt_dict["scheduler"] = scheduler.state_dict()

        if dev_loader is not None:
            metrics = evaluate(model, dev_loader, device)
            msg += f"  dev: WA={metrics['wa']:.4f} UA={metrics['ua']:.4f} WF1={metrics['wf1']:.4f}"
            if metrics["wf1"] > best_dev_wf1:
                best_dev_wf1 = metrics["wf1"]
                ckpt_dict["dev_wf1"] = metrics["wf1"]
                torch.save(ckpt_dict, args.checkpoint_out)
                msg += "  (new best, saved)"
        else:
            torch.save(ckpt_dict, args.checkpoint_out)

        print(msg)

    print("Training complete.")


if __name__ == "__main__":
    main()
