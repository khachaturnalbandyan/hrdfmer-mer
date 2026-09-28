"""
predict.py
==========
The real-world inference pipeline: you give it ONE audio file, and it
predicts an emotion. No ground-truth transcript required.

  audio.wav
     |
     |-- Whisper (ASR) -----------> text
     |-- Wav2Vec2-base -----------> X_e_a  (audio embedding)
     |-- librosa MFCC ------------> raw MFCC
     |
     text --> RoBERTa-base -------> X_e_t  (text embedding)
     (RoBERTa runs on the ASR OUTPUT, not a human transcript - this is
      the key difference from training, where MELD gives real transcripts)
     |
     [X_e_t, X_e_a, raw_MFCC] --> HRDF-MER (trained, loaded from checkpoint)
     |
     predicted emotion + confidence

This file imports torch/transformers/librosa (needed for Whisper, RoBERTa,
Wav2Vec2, MFCC) - that's expected and fine, since it's a one-shot inference
utility, not part of the model/training code itself. The HRDF-MER forward
pass underneath is still 100% the from-scratch nptorch engine.

Usage:
    python predict.py --audio path/to/clip.wav --checkpoint checkpoints/model.npz
"""
import argparse
import numpy as np

from nptorch.tensor import Tensor
from model import HRDFMER
from model.labels import MELD_IDX2LABEL
from model.checkpoint import load_checkpoint
from preprocess import (
    extract_text_embedding, extract_audio_embedding, extract_mfcc,
    transcribe_audio, build_asr_pipeline,
)


def load_pretrained_encoders():
    from transformers import RobertaTokenizer, RobertaModel, Wav2Vec2Processor, Wav2Vec2Model
    tok = RobertaTokenizer.from_pretrained("roberta-base")
    text_model = RobertaModel.from_pretrained("roberta-base").eval()
    wav_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval()
    return tok, text_model, wav_processor, audio_model


def predict(audio_path: str, checkpoint_path: str, d_model=128, n_heads=4, mfcc_dim=40, embed_dim=768):
    print(f"Transcribing {audio_path} ...")
    asr = build_asr_pipeline()
    text = transcribe_audio(audio_path, asr)
    print(f"  ASR transcript: \"{text}\"")

    print("Extracting embeddings ...")
    tok, text_model, wav_processor, audio_model = load_pretrained_encoders()
    text_emb = extract_text_embedding(text, tok, text_model)          # (L_t, 768)
    audio_emb = extract_audio_embedding(audio_path, wav_processor, audio_model)  # (L_a, 768)
    mfcc = extract_mfcc(audio_path, n_mfcc=mfcc_dim)                  # (L_m, 40)

    # wrap as batch-of-1 nptorch Tensors
    X_e_t = Tensor(text_emb[None, :, :])
    X_e_a = Tensor(audio_emb[None, :, :])
    X_mfcc = Tensor(mfcc[None, :, :])

    print("Loading trained HRDF-MER model ...")
    model = HRDFMER(d_model=d_model, n_heads=n_heads, n_classes=len(MELD_IDX2LABEL),
                     mfcc_dim=mfcc_dim, embed_dim=embed_dim)
    load_checkpoint(checkpoint_path, model)

    print("Running prediction ...")
    logits, _, _, _ = model(X_e_t, X_e_a, X_mfcc)
    probs = logits.softmax(axis=-1).data[0]
    pred_idx = int(np.argmax(probs))

    print(f"\nPredicted emotion: {MELD_IDX2LABEL[pred_idx]}  (confidence {probs[pred_idx]:.2%})")
    print("Full distribution:")
    for idx, label in MELD_IDX2LABEL.items():
        print(f"  {label:10s} {probs[idx]:.2%}")

    return MELD_IDX2LABEL[pred_idx], probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to a .wav (or any librosa/ffmpeg-readable) audio file")
    parser.add_argument("--checkpoint", required=True, help="Path to a .npz checkpoint from train.py")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--mfcc_dim", type=int, default=40)
    parser.add_argument("--embed_dim", type=int, default=768)
    args = parser.parse_args()
    predict(args.audio, args.checkpoint, args.d_model, args.n_heads, args.mfcc_dim, args.embed_dim)
