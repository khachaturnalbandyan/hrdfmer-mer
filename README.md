# HRDF-MER (unofficial PyTorch reimplementation)

An independent, unofficial reimplementation of **HRDF-MER**: *"Hierarchical
feature refinement and cascaded dynamic fusion for multimodal emotion
recognition"* (Lei, Mu & Wang, 2026), trained end-to-end on the real
[MELD](https://github.com/declare-lab/MELD) dataset.

This is a student project (built while studying AI at Yerevan State
University), not an official implementation, and it is not affiliated with
the paper's authors. It reproduces the architecture from the paper as
closely as reasonably possible from the description, but the exact
training details (hyperparameters, data splits, minor implementation
choices) are my own, so results are close to but not identical to the
paper's.

## What's here

A full text + audio + MFCC multimodal emotion classifier:

- **Embedding layer**: precomputed RoBERTa-base (text) and Wav2Vec2-base
  (audio) embeddings, plus a trainable LSTM over raw MFCC features.
- **HCFR** (Hierarchical Cross-modal Feature Refinement): cross-modal
  adaptive alignment of audio/MFCC against text, followed by self- and
  cross-attention enhancement per modality.
- **CMDF** (Cascaded Multimodal Dynamic Fusion): a 3-stage cross-attention
  cascade between modalities, followed by 2-stage learned gated fusion.
- **Multi-objective loss**: cross-entropy + a 3-stage cross-modal
  label-contrastive (CLC) loss with learnable multi-perspective pooling.

Everything is implemented in real `torch.nn` (GPU-capable) with proper
attention padding masks throughout, so batches of variable-length
utterances are handled correctly.

An earlier from-scratch version — the entire architecture built on a
custom NumPy autograd engine, with no PyTorch/TensorFlow at all — is kept
for reference in [`legacy/numpy_version/`](legacy/numpy_version/). It was
the starting point for this project and helped verify the architecture is
correctly understood before moving to a GPU-capable implementation, but
it's slower, has no padding-mask support in attention, and is not the
recommended way to use this repo.

## Results

Trained on the full MELD dataset (train=9,988 / dev=1,108 / test=2,610
utterances, 7 emotion classes), `d_model=256` (vs. the paper's 768, for
faster training on consumer GPU hardware):

| Split | WA (accuracy) | UA (macro recall) | WF1 (weighted F1) |
|---|---|---|---|
| Dev  | ~56.9% | – | 56.85% |
| **Test** | **61.69%** | **37.60%** | **59.26%** |
| Paper's reported HRDF-MER (test) | 64.76% | – | 63.02% |

So this reimplementation lands a few points below the paper's numbers —
reasonable for a smaller `d_model`, a single training run (no multi-seed
averaging), and no exhaustive hyperparameter search.

**Known weakness**: MELD is heavily class-imbalanced (`neutral` is ~47% of
the data; `fear` and `disgust` are each under 3%). This model gets **0%
recall on both `fear` and `disgust`** on the test set — it never predicts
either class. UA (macro recall, which weights all 7 classes equally) is
correspondingly low at 37.6% even though overall accuracy (WA) is
respectable. This is the single biggest thing to fix before treating this
as more than a demo — see [Limitations](#limitations--known-issues).

Per-class test recall:

| Class | Recall | Support |
|---|---|---|
| neutral | high | largest class |
| joy | moderate | |
| surprise | moderate | |
| sadness | low-moderate | |
| anger | low-moderate | |
| fear | **0%** | smallest classes |
| disgust | **0%** | smallest classes |

(Re-run `evaluate.py` on your own checkpoint for exact numbers per class —
see below.)

## Training recipe

What actually got the best result, for anyone reproducing this:

1. Train from scratch with a constant LR: `--lr 1e-3 --no_lr_schedule` →
   dev WF1 ≈ 0.549.
2. A more aggressive `--lr 3e-3` with cosine decay was **worse** (dev WF1 ≈
   0.479) — too-aggressive an initial LR caused early instability the
   decay schedule couldn't fully recover from.
3. **Warm start** from step 1's checkpoint (`--init_from`, which loads only
   model+loss weights and starts a fresh optimizer/LR schedule at epoch 0)
   with a gentler `--lr 5e-4` cosine decay → dev WF1 = 0.5685, test WF1 =
   0.5926 (the reported result above).

`--init_from` (warm start: fresh optimizer, fresh epoch counter) is
distinct from `--resume_from` (continue the exact same run: restores
optimizer + scheduler + epoch too) — both are supported by `train.py`.

## Setup

```bash
pip install -r requirements.txt --break-system-packages   # drop the flag if you're using a venv
```

If you have an NVIDIA GPU, make sure `nvidia-smi` reports it correctly
before training — a `Driver/library version mismatch` error is usually
fixed by a reboot after a driver update.

### 1. Get MELD

Download `MELD.Raw.tar.gz` from
[declare-lab/MELD](https://github.com/declare-lab/MELD) (Hugging Face
mirror linked from their README) and extract it. You should end up with:

```
MELD.Raw/
  train_sent_emo.csv
  dev_sent_emo.csv
  test_sent_emo.csv
  train_splits/                    (.mp4 files)
  dev_splits_complete/             (.mp4 files)
  output_repeated_splits_test/     (.mp4 files)
```

You'll also need `ffmpeg` on your `PATH` (`ffmpeg -version` to check).

### 2. Extract audio + build manifests

```bash
python meld_prepare.py --meld_root /path/to/MELD.Raw --split train --out_dir data/meld_train
python meld_prepare.py --meld_root /path/to/MELD.Raw --split dev   --out_dir data/meld_dev
python meld_prepare.py --meld_root /path/to/MELD.Raw --split test  --out_dir data/meld_test
```

### 3. Extract RoBERTa/Wav2Vec2/MFCC features (the only step that touches `torch`/`transformers`/`librosa` besides training itself)

```bash
python preprocess.py --audio_dir data/meld_train/wav --text_csv data/meld_train/manifest.csv --out_dir data/meld_train/features
python preprocess.py --audio_dir data/meld_dev/wav   --text_csv data/meld_dev/manifest.csv   --out_dir data/meld_dev/features
python preprocess.py --audio_dir data/meld_test/wav  --text_csv data/meld_test/manifest.csv  --out_dir data/meld_test/features
```

### 4. Train

```bash
python train.py --train_features data/meld_train/features \
                 --dev_features data/meld_dev/features \
                 --checkpoint_out checkpoints/model.pt \
                 --d_model 256 --n_heads 8 --epochs 20 --lr 1e-3 --no_lr_schedule
```

Then, optionally, warm-start a second run for a better result (see
[Training recipe](#training-recipe) above):

```bash
python train.py --train_features data/meld_train/features \
                 --dev_features data/meld_dev/features \
                 --checkpoint_out checkpoints/model_v2.pt \
                 --init_from checkpoints/model.pt \
                 --d_model 256 --n_heads 8 --epochs 15 --lr 5e-4
```

Run `python train.py --help` for the full list of flags (batch size,
gradient accumulation, `max_len_*` truncation caps, loss weights `delta`/
`lam`, etc).

### 5. Evaluate on the held-out test set

```bash
python evaluate.py --features data/meld_test/features --checkpoint checkpoints/model_v2.pt
```

Prints WA/UA/WF1 plus a per-class recall breakdown, and the paper's
reported numbers for comparison. This should only be run once, at the end
— it was never used for model selection during training (that's what the
dev split + `train.py`'s best-checkpoint saving is for).

### 6. Publish a lightweight checkpoint

Training checkpoints include optimizer state (Adam roughly doubles or
triples the file size) which is useless once training is done. Before
uploading a checkpoint anywhere (a GitHub Release, Hugging Face Hub),
strip it down:

```bash
python export_weights.py --checkpoint checkpoints/model_v2.pt --out weights/hrdfmer_meld.pt
```

The resulting file still works for `evaluate.py`, `predict.py`, and as an
`--init_from` warm-start source — it just drops the optimizer/scheduler
state.

### 7. Real-world inference (audio in, emotion out)

```bash
python predict.py --audio path/to/clip.wav --checkpoint weights/hrdfmer_meld.pt
```

This runs the full pipeline: Whisper ASR transcription → RoBERTa/Wav2Vec2
embedding extraction → HRDF-MER → predicted emotion + confidence
distribution over all 7 classes.

**Caveat**: this script is implemented following the same design as the
rest of the pipeline (and its individual pieces — ASR, embedding
extraction, model inference — are each exercised elsewhere: `preprocess.py`
by the training data pipeline, the model by `evaluate.py`), but the full
`predict.py` script itself has not yet been run end-to-end on a real audio
file. Treat it as untested glue code until you verify it locally with
`ffmpeg`/Whisper installed and a real `.wav`.

## Tests

```bash
pip install pytest --break-system-packages
python -m pytest tests/ -v
```

13 tests covering: forward-pass shapes (with and without padding masks,
batch size 1, decoupled `embed_dim`/`d_model`), padding-mask invariance
(the property that padded/masked timesteps must not change the model's
output — this is the specific bug class the from-scratch NumPy version had
and this PyTorch rewrite fixes), gradient checks (every parameter gets a
finite gradient, an optimizer step actually changes weights), and the
hand-rolled WA/UA/WF1 metric implementation against known values.

## Project layout

```
model/
  hrdfmer.py    - top-level HRDFMER module (embedding proj + LSTM, HCFR, CMDF, classifier)
  hcfr.py       - Hierarchical Cross-modal Feature Refinement (CAA + HFE)
  cmdf.py       - Cascaded Multimodal Dynamic Fusion (3-stage cross-attention + gated fusion)
  losses.py     - CE + multi-stage Cross-modal Label Contrastive (CLC) loss
  dataset.py    - MELDFeatureDataset + dynamic-padding collate_fn
  metrics.py    - WA/UA/WF1, implemented from scratch (no sklearn)
  labels.py     - MELD's 7 emotion class <-> index mapping
train.py         - training loop (resume / warm-start, cosine LR, grad accumulation)
evaluate.py       - final test-set evaluation against a checkpoint
predict.py        - end-to-end audio -> ASR -> HRDF-MER -> emotion inference
preprocess.py     - offline RoBERTa/Wav2Vec2/MFCC feature extraction (the only file besides predict.py's embedding step that imports torch/transformers/librosa)
meld_prepare.py   - MELD raw .mp4/.csv -> preprocess.py's expected manifest format
export_weights.py - strip a training checkpoint down to a publishable weights-only file
tests/            - shape / masking / gradient / metric tests (pytest)
legacy/numpy_version/ - the original from-scratch NumPy autograd implementation
```

## Limitations & known issues

- **`d_model=256` vs. the paper's 768** — chosen for faster training on a
  single consumer GPU (RTX 4060, 16GB system RAM); larger `d_model` should
  close some of the gap to the paper's reported numbers.
- **Class imbalance is not addressed** — no class-weighted loss, no
  oversampling of `fear`/`disgust`. This is the direct cause of the 0%
  recall on those two classes. Likely fixes: weighted cross-entropy, focal
  loss, or oversampling the rare classes.
- **Single training run, no seed averaging or hyperparameter search** —
  the paper's numbers are presumably averaged/tuned more thoroughly.
  `--delta`/`--lam` here use the paper's reported optimal values (Table 7)
  but haven't been independently re-tuned for this reimplementation.
  `--seed` is exposed for reproducibility of a single run, and 20-50
  epochs were used.
- **`predict.py` is untested end-to-end** (see above) — its components are
  exercised individually but the full script hasn't been run against a
  real `.wav` file yet.
- **The legacy NumPy version lacks attention padding masks entirely** — it
  pads all sequences to a fixed length and lets attention see the padding,
  which is architecturally incorrect (though it still trained and
  converged on synthetic data). This is one of the reasons the project
  moved to the PyTorch version for real training.

## Citation

This repository implements the architecture described in:

> Lei, Mu & Wang. "HRDF-MER: Hierarchical feature refinement and cascaded
> dynamic fusion for multimodal emotion recognition." (2026)

Please cite the original paper if you use the architecture; cite this repo
separately if you build on this specific implementation or its results.

## License

MIT — see [LICENSE](LICENSE). Covers this repository's code only; not the
paper, the MELD dataset, or the pretrained RoBERTa/Wav2Vec2/Whisper models
it depends on, each of which has its own license/terms.
