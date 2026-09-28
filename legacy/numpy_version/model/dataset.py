"""
model.dataset
=============
Loads the .npy features produced by preprocess.py and batches them with
DYNAMIC padding: each batch's text/audio/MFCC sequences are zero-padded
only up to that batch's own max length (not a fixed global max), matching
the paper's "sequences ... are dynamically padded to the maximum length
within each batch" (Section 3.2.1).

Expected directory layout (what preprocess.py produces):
    features_dir/
        <id>_text.npy    (L_t, 768)
        <id>_audio.npy   (L_a, 768)
        <id>_mfcc.npy    (L_m, mfcc_dim)
        labels.csv        id,label
"""
import csv
import os
import random
import numpy as np

from nptorch.tensor import Tensor


def load_labels(features_dir: str) -> dict:
    path = os.path.join(features_dir, "labels.csv")
    labels = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["label"]
    return labels


def _pad_batch(arrays, max_len_cap=None, pad_dim_axis=0):
    """arrays: list of (L_i, D) numpy arrays with the same D but varying L_i.
    If max_len_cap is set, each array is first TRUNCATED to at most that
    many frames (keeping the first max_len_cap timesteps) - this bounds
    memory/compute regardless of how long a real audio clip's sequence is,
    since pure-NumPy autograd keeps the whole graph in memory with no
    optimization (unlike PyTorch).
    Returns (B, L_max, D) zero-padded, post-padding (real values at the
    start of the sequence, zeros after)."""
    if max_len_cap is not None:
        arrays = [a[:max_len_cap] for a in arrays]
    max_len = max(a.shape[pad_dim_axis] for a in arrays)
    D = arrays[0].shape[1]
    B = len(arrays)
    out = np.zeros((B, max_len, D), dtype=np.float64)
    for i, a in enumerate(arrays):
        L = a.shape[pad_dim_axis]
        out[i, :L, :] = a
    return out


class MELDFeatureDataset:
    """Holds the list of utterance ids available in features_dir (i.e. ones
    that have all three .npy files AND a recognized label) and knows how
    to turn a list of ids into a padded batch of nptorch Tensors.

    max_len_text/audio/mfcc: cap on sequence length (truncates longer
    sequences before padding). Important for memory: this pure-NumPy
    autograd engine keeps the ENTIRE computation graph in memory for
    backward(), with none of PyTorch's memory optimizations, so an
    uncapped long audio clip (hundreds of MFCC/Wav2Vec2 frames) can
    exhaust RAM, especially once the MFCC LSTM unrolls per-timestep.
    Defaults are conservative; raise them only if you have RAM to spare.
    """

    def __init__(self, features_dir: str, label2idx: dict,
                 max_len_text: int = 40, max_len_audio: int = 150, max_len_mfcc: int = 150):
        self.features_dir = features_dir
        self.label2idx = label2idx
        self.max_len_text = max_len_text
        self.max_len_audio = max_len_audio
        self.max_len_mfcc = max_len_mfcc
        raw_labels = load_labels(features_dir)

        self.ids = []
        skipped_missing_files, skipped_bad_label = 0, 0
        for uid, label in raw_labels.items():
            if label not in label2idx:
                skipped_bad_label += 1
                continue
            paths = [os.path.join(features_dir, f"{uid}_{m}.npy") for m in ("text", "audio", "mfcc")]
            if not all(os.path.isfile(p) for p in paths):
                skipped_missing_files += 1
                continue
            self.ids.append(uid)

        if skipped_missing_files or skipped_bad_label:
            print(f"[MELDFeatureDataset] loaded {len(self.ids)} utterances "
                  f"(skipped {skipped_missing_files} with missing .npy files, "
                  f"{skipped_bad_label} with unrecognized labels)")
        self._raw_labels = raw_labels

    def __len__(self):
        return len(self.ids)

    def get_batch(self, ids):
        text_arrs, audio_arrs, mfcc_arrs, label_idxs = [], [], [], []
        for uid in ids:
            text_arrs.append(np.load(os.path.join(self.features_dir, f"{uid}_text.npy")))
            audio_arrs.append(np.load(os.path.join(self.features_dir, f"{uid}_audio.npy")))
            mfcc_arrs.append(np.load(os.path.join(self.features_dir, f"{uid}_mfcc.npy")))
            label_idxs.append(self.label2idx[self._raw_labels[uid]])

        X_e_t = Tensor(_pad_batch(text_arrs, self.max_len_text))
        X_e_a = Tensor(_pad_batch(audio_arrs, self.max_len_audio))
        X_mfcc = Tensor(_pad_batch(mfcc_arrs, self.max_len_mfcc))
        labels = np.array(label_idxs, dtype=np.int64)
        return X_e_t, X_e_a, X_mfcc, labels


def iterate_batches(dataset: MELDFeatureDataset, batch_size: int, shuffle: bool = True, seed: int = None):
    """Yields (X_e_t, X_e_a, X_mfcc, labels) batches. Drops a trailing
    partial batch smaller than 2 samples (the CLC loss needs at least 2
    samples per batch to form any pairs)."""
    ids = list(dataset.ids)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(ids)
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start:start + batch_size]
        if len(batch_ids) < 2:
            continue
        yield dataset.get_batch(batch_ids)
