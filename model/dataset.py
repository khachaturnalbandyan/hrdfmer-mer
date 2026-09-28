"""
model.dataset (PyTorch version)
==================================
Loads the .npy features from preprocess.py and batches them with dynamic
per-batch padding via a custom collate_fn (matching the paper's approach),
PLUS produces padding masks so attention/pooling correctly ignore padded
timesteps - real GPU memory management (pinned memory, pack-friendly
layout) is handled by DataLoader, so this is simpler and more efficient
than the from-scratch NumPy version.
"""
import csv
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def load_labels(features_dir: str) -> dict:
    path = os.path.join(features_dir, "labels.csv")
    labels = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["label"]
    return labels


class MELDFeatureDataset(Dataset):
    """One item = one utterance's (text, audio, mfcc, label) as tensors,
    truncated to max_len_* if given. Padding to a common length within a
    batch happens in collate_fn, not here (that's what makes the padding
    "dynamic": batch-dependent, not a fixed global max)."""

    def __init__(self, features_dir: str, label2idx: dict,
                 max_len_text: int = 64, max_len_audio: int = 300, max_len_mfcc: int = 300):
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

    def __getitem__(self, idx):
        uid = self.ids[idx]
        text = np.load(os.path.join(self.features_dir, f"{uid}_text.npy"))[:self.max_len_text]
        audio = np.load(os.path.join(self.features_dir, f"{uid}_audio.npy"))[:self.max_len_audio]
        mfcc = np.load(os.path.join(self.features_dir, f"{uid}_mfcc.npy"))[:self.max_len_mfcc]
        label = self.label2idx[self._raw_labels[uid]]
        return (torch.from_numpy(text).float(),
                torch.from_numpy(audio).float(),
                torch.from_numpy(mfcc).float(),
                label)


def collate_fn(batch):
    """Pads each modality to that batch's own max length, and builds the
    matching key_padding_mask (True = padding) for each."""
    texts, audios, mfccs, labels = zip(*batch)

    def pad_and_mask(seqs):
        lengths = torch.tensor([s.shape[0] for s in seqs])
        padded = pad_sequence(seqs, batch_first=True)   # (B, L_max, D), zero-padded
        L_max = padded.shape[1]
        mask = torch.arange(L_max).unsqueeze(0) >= lengths.unsqueeze(1)   # True where padding
        return padded, mask

    X_e_t, text_mask = pad_and_mask(texts)
    X_e_a, audio_mask = pad_and_mask(audios)
    X_mfcc, mfcc_mask = pad_and_mask(mfccs)
    labels = torch.tensor(labels, dtype=torch.long)

    return X_e_t, X_e_a, X_mfcc, text_mask, audio_mask, mfcc_mask, labels
