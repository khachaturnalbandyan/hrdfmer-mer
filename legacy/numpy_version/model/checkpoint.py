"""
model.checkpoint
=================
Save/load for nptorch Modules. There's no built-in serialization (unlike
torch.save), so this walks a model's .parameters() in a fixed order and
stores them as a flat list of numpy arrays via np.savez - matched back up
by position on load, so the model architecture must be identical between
save and load (same d_model, n_heads, n_classes, etc).
"""
import numpy as np


def save_checkpoint(path: str, model, optimizer=None, extra: dict = None):
    params = model.parameters()
    arrays = {f"param_{i}": p.data for i, p in enumerate(params)}
    if optimizer is not None:
        for i, m in enumerate(optimizer.m):
            arrays[f"opt_m_{i}"] = m
        for i, v in enumerate(optimizer.v):
            arrays[f"opt_v_{i}"] = v
        arrays["opt_t"] = np.array(optimizer.t)
    if extra:
        for k, v in extra.items():
            arrays[f"extra_{k}"] = np.asarray(v)
    np.savez(path, **arrays)
    print(f"Saved checkpoint to {path} ({len(params)} parameter tensors)")


def load_checkpoint(path: str, model, optimizer=None):
    data = np.load(path, allow_pickle=True)
    params = model.parameters()
    for i, p in enumerate(params):
        key = f"param_{i}"
        if key not in data:
            raise ValueError(f"Checkpoint missing {key} - does the model architecture match "
                              f"what was saved? (same d_model/n_heads/n_classes etc.)")
        if data[key].shape != p.data.shape:
            raise ValueError(f"Shape mismatch at {key}: checkpoint has {data[key].shape}, "
                              f"model expects {p.data.shape}")
        p.data = data[key].copy()
    if optimizer is not None and "opt_t" in data:
        optimizer.t = int(data["opt_t"])
        for i in range(len(optimizer.m)):
            optimizer.m[i] = data[f"opt_m_{i}"].copy()
            optimizer.v[i] = data[f"opt_v_{i}"].copy()
    extras = {k[len("extra_"):]: data[k] for k in data.files if k.startswith("extra_")}
    print(f"Loaded checkpoint from {path} ({len(params)} parameter tensors)")
    return extras
