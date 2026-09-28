"""
demo.py
=======
Sanity check: random dummy arrays standing in for precomputed
RoBERTa/Wav2Vec2 embeddings + raw MFCC, run through the FULL custom
nptorch/model stack (no torch/tf anywhere below this line), for a few
Adam steps, to confirm forward + backward + optimizer all work and the
loss goes down.

Note: real d_model=768 (as in the paper) makes the pure-Python/NumPy
LSTM loop and attention slow. For a quick correctness check we use a
smaller d_model; swap in 768 once you're plugging in real precomputed
features and are OK with slower iteration.
"""
import numpy as np
from nptorch.tensor import Tensor
from nptorch.optim import Adam
from model import HRDFMER, MultiObjectiveLoss

np.random.seed(0)

# ---- config (small, for a fast smoke test) --------------------------------
B = 4            # batch size
L_t, L_a, L_m = 6, 8, 10   # sequence lengths per modality
d_model = 32
mfcc_dim = 13
n_classes = 4
n_steps = 15

# ---- build dummy inputs, standing in for preprocess.py's output -----------
X_e_t = Tensor(np.random.randn(B, L_t, d_model) * 0.1)
X_e_a = Tensor(np.random.randn(B, L_a, d_model) * 0.1)
X_mfcc = Tensor(np.random.randn(B, L_m, mfcc_dim) * 0.1)
labels = np.random.randint(0, n_classes, size=(B,))

# ---- build model + loss + optimizer ----------------------------------------
model = HRDFMER(d_model=d_model, n_heads=4, n_classes=n_classes, mfcc_dim=mfcc_dim, embed_dim=d_model)
loss_fn = MultiObjectiveLoss(d_model=d_model, d_proj=16, n_perspectives=2, delta=0.5, lam=0.2)

params = model.parameters() + loss_fn.parameters()
opt = Adam(params, lr=1e-2)

print(f"Total learnable parameters: {sum(p.data.size for p in params):,}")

for step in range(n_steps):
    opt.zero_grad()
    logits, stage1, stage2, stage3 = model(X_e_t, X_e_a, X_mfcc)
    total_loss, parts = loss_fn(logits, labels, stage1, stage2, stage3)
    total_loss.backward()
    opt.step()

    preds = np.argmax(logits.data, axis=1)
    acc = (preds == labels).mean()
    print(f"step {step:02d}  loss={total_loss.data.item():.4f}  "
          f"(ce={parts['ce'].data.item():.4f} "
          f"clc1={parts['clc1'].data.item():.4f} "
          f"clc2={parts['clc2'].data.item():.4f} "
          f"clc3={parts['clc3'].data.item():.4f})  acc={acc:.2f}")

print("\nShape check:")
print("logits:", logits.shape, "-> expected", (B, n_classes))
print("stage1 shapes:", [t.shape for t in stage1])
print("stage2 shapes:", [t.shape for t in stage2])
print("stage3 shapes:", [t.shape for t in stage3])
