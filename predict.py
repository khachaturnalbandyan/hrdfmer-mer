"""
predict.py (PyTorch version)
===============================
Real-world inference: one audio file -> ASR transcript -> HRDF-MER ->
predicted emotion. Same pipeline design as the NumPy version's predict.py,
now running the model on GPU if available.

Usage:
    python predict.py --audio path/to/clip.wav --checkpoint checkpoints/model.pt
"""
import argparse
import torch

from model import HRDFMER
from model.labels import MELD_IDX2LABEL
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


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict(audio_path: str, checkpoint_path: str):
    device = get_device()
    print(f"Using device: {device}")

    print(f"Transcribing {audio_path} ...")
    asr = build_asr_pipeline()
    text = transcribe_audio(audio_path, asr)
    print(f"  ASR transcript: \"{text}\"")

    print("Extracting embeddings ...")
    tok, text_model, wav_processor, audio_model = load_pretrained_encoders()
    text_emb = extract_text_embedding(text, tok, text_model)
    audio_emb = extract_audio_embedding(audio_path, wav_processor, audio_model)
    mfcc = extract_mfcc(audio_path, n_mfcc=40)

    print("Loading trained HRDF-MER model ...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    train_args = ckpt.get("args", {})
    model = HRDFMER(d_model=train_args.get("d_model", 768), n_heads=train_args.get("n_heads", 8),
                     n_classes=len(MELD_IDX2LABEL), mfcc_dim=train_args.get("mfcc_dim", 40),
                     embed_dim=train_args.get("embed_dim", 768)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    X_e_t = torch.from_numpy(text_emb).float().unsqueeze(0).to(device)
    X_e_a = torch.from_numpy(audio_emb).float().unsqueeze(0).to(device)
    X_mfcc = torch.from_numpy(mfcc).float().unsqueeze(0).to(device)

    print("Running prediction ...")
    with torch.no_grad():
        logits, *_ = model(X_e_t, X_e_a, X_mfcc)   # no masks needed: batch of 1, no padding
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())

    print(f"\nPredicted emotion: {MELD_IDX2LABEL[pred_idx]}  (confidence {probs[pred_idx]:.2%})")
    print("Full distribution:")
    for idx, label in MELD_IDX2LABEL.items():
        print(f"  {label:10s} {probs[idx]:.2%}")

    return MELD_IDX2LABEL[pred_idx], probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    predict(args.audio, args.checkpoint)
