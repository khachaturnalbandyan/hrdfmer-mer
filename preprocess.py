"""
preprocess.py
=============
One-time, offline feature extraction. This is the ONLY file in the whole
project that imports torch / transformers. Its job is to turn raw
audio/text into plain numpy arrays and save them to disk - after this
runs, nothing else in the project (nptorch/, model/, train.py) ever
touches torch again.

Usage sketch (fill in with your actual IEMOCAP/MELD file lists):

    python preprocess.py --audio_dir data/wav --text_csv data/utterances.csv \
                          --out_dir data/features

For each utterance this produces three .npy files:
    <id>_text.npy   (L_t, 768)   RoBERTa-base last_hidden_state
    <id>_audio.npy  (L_a, 768)   Wav2Vec2-base last_hidden_state
    <id>_mfcc.npy   (L_m, 40)    raw MFCC via librosa (NOT run through LSTM here -
                                  the LSTM is part of our own model, trained end-to-end)

Install requirements for THIS FILE ONLY:
    pip install torch transformers librosa --break-system-packages
(the rest of the project needs none of these)
"""
import argparse
import os
import numpy as np


def extract_text_embedding(text: str, tokenizer, model) -> np.ndarray:
    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state.squeeze(0).cpu().numpy()   # (L_t, 768)


def extract_audio_embedding(wav_path: str, processor, model, sample_rate=16000) -> np.ndarray:
    import torch
    import librosa
    speech, _ = librosa.load(wav_path, sr=sample_rate)
    inputs = processor(speech, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs.input_values if isinstance(inputs.input_values, dict) else inputs)
    return out.last_hidden_state.squeeze(0).cpu().numpy()   # (L_a, 768)


def extract_mfcc(wav_path: str, sample_rate=16000, n_mfcc=40) -> np.ndarray:
    import librosa
    speech, _ = librosa.load(wav_path, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=speech, sr=sample_rate, n_mfcc=n_mfcc)
    return mfcc.T.astype(np.float64)   # (L_m, n_mfcc)


def transcribe_audio(wav_path: str, asr_pipeline) -> str:
    """
    ASR step for inference-time use, when no ground-truth transcript exists
    (unlike MELD, which already provides one). Uses OpenAI Whisper via
    transformers' pipeline() - a different pretrained model than the
    Wav2Vec2-base used for audio embeddings, since Wav2Vec2-base itself
    isn't CTC/ASR fine-tuned.
    """
    result = asr_pipeline(wav_path)
    return result["text"].strip()


def build_asr_pipeline(model_name="openai/whisper-base"):
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("automatic-speech-recognition", model=model_name, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--text_csv", required=True, help="CSV with columns: id,text,wav_filename,label")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    from transformers import RobertaTokenizer, RobertaModel, Wav2Vec2Processor, Wav2Vec2Model
    import csv

    os.makedirs(args.out_dir, exist_ok=True)

    tok = RobertaTokenizer.from_pretrained("roberta-base")
    text_model = RobertaModel.from_pretrained("roberta-base").eval()
    wav_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval()

    with open(args.text_csv) as f:
        rows = list(csv.DictReader(f))

    labels = {}
    for row in rows:
        uid, text, wav_file, label = row["id"], row["text"], row["wav_filename"], row["label"]
        wav_path = os.path.join(args.audio_dir, wav_file)

        text_emb = extract_text_embedding(text, tok, text_model)
        audio_emb = extract_audio_embedding(wav_path, wav_processor, audio_model)
        mfcc = extract_mfcc(wav_path)

        np.save(os.path.join(args.out_dir, f"{uid}_text.npy"), text_emb)
        np.save(os.path.join(args.out_dir, f"{uid}_audio.npy"), audio_emb)
        np.save(os.path.join(args.out_dir, f"{uid}_mfcc.npy"), mfcc)
        labels[uid] = label

    with open(os.path.join(args.out_dir, "labels.csv"), "w") as f:
        f.write("id,label\n")
        for uid, label in labels.items():
            f.write(f"{uid},{label}\n")

    print(f"Done. Wrote features for {len(rows)} utterances to {args.out_dir}")


if __name__ == "__main__":
    main()
