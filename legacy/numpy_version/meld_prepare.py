"""
meld_prepare.py
================
MELD ships as: per-split CSVs (train_sent_emo.csv / dev_sent_emo.csv /
test_sent_emo.csv) plus folders of .mp4 clips named dia{D}_utt{U}.mp4,
where D=Dialogue_ID and U=Utterance_ID from the CSV row.

This script:
  1. Reads a MELD split CSV.
  2. Extracts the audio track from each utterance's .mp4 into a .wav file
     (via ffmpeg - must be installed and on PATH: `ffmpeg -version` to check).
  3. Writes a normalized CSV (id,text,wav_filename,label) in exactly the
     format preprocess.py expects, with labels already mapped to the
     paper's class names.

Get MELD from: https://github.com/declare-lab/MELD (see their README for
the MELD.Raw.tar.gz download link). After extracting you should have:

    MELD.Raw/
      train_sent_emo.csv
      dev_sent_emo.csv
      test_sent_emo.csv
      train_splits/                    (.mp4 files)
      dev_splits_complete/             (.mp4 files)
      output_repeated_splits_test/     (.mp4 files)

Usage:
    python meld_prepare.py --meld_root /path/to/MELD.Raw --split train \
                            --out_dir data/meld_train

Then feed data/meld_train/wav/ and data/meld_train/manifest.csv straight
into preprocess.py:
    python preprocess.py --audio_dir data/meld_train/wav \
                          --text_csv data/meld_train/manifest.csv \
                          --out_dir data/meld_train/features
"""
import argparse
import csv
import os
import subprocess
import sys

from model.labels import MELD_LABEL2IDX

SPLIT_TO_FOLDER = {
    "train": ("train_sent_emo.csv", "train_splits"),
    "dev": ("dev_sent_emo.csv", "dev_splits_complete"),
    "test": ("test_sent_emo.csv", "output_repeated_splits_test"),
}


def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit("ffmpeg not found on PATH. Install it first (e.g. `apt install ffmpeg` / "
                 "`brew install ffmpeg`) - it's needed to pull audio out of MELD's .mp4 clips.")


def extract_wav(mp4_path: str, wav_path: str, sample_rate: int = 16000):
    """ffmpeg -i clip.mp4 -ar 16000 -ac 1 -vn clip.wav"""
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp4_path, "-ar", str(sample_rate), "-ac", "1", "-vn", wav_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meld_root", required=True, help="Path to extracted MELD.Raw/")
    parser.add_argument("--split", required=True, choices=["train", "dev", "test"])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on # utterances, for a quick test run")
    args = parser.parse_args()

    check_ffmpeg()

    csv_name, video_folder = SPLIT_TO_FOLDER[args.split]
    csv_path = os.path.join(args.meld_root, csv_name)
    video_dir = os.path.join(args.meld_root, video_folder)
    if not os.path.isfile(csv_path):
        sys.exit(f"Couldn't find {csv_path} - check --meld_root points at the extracted MELD.Raw folder.")

    wav_dir = os.path.join(args.out_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    manifest_rows = []
    n_missing, n_skipped_label = 0, 0

    for row in rows:
        dia_id = row["Dialogue_ID"]
        utt_id = row["Utterance_ID"]
        text = row["Utterance"]
        emotion = row["Emotion"].strip().lower()

        if emotion not in MELD_LABEL2IDX:
            n_skipped_label += 1
            continue

        uid = f"{args.split}_dia{dia_id}_utt{utt_id}"
        mp4_path = os.path.join(video_dir, f"dia{dia_id}_utt{utt_id}.mp4")
        wav_filename = f"{uid}.wav"
        wav_path = os.path.join(wav_dir, wav_filename)

        if not os.path.isfile(mp4_path):
            n_missing += 1
            continue

        if not os.path.isfile(wav_path):   # skip re-extracting if resuming a partial run
            try:
                extract_wav(mp4_path, wav_path)
            except subprocess.CalledProcessError:
                n_missing += 1
                continue

        manifest_rows.append({"id": uid, "text": text, "wav_filename": wav_filename, "label": emotion})

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "wav_filename", "label"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote {len(manifest_rows)} utterances to {manifest_path}")
    print(f"  wav files in: {wav_dir}")
    if n_missing:
        print(f"  skipped {n_missing} rows (video file missing or ffmpeg failed)")
    if n_skipped_label:
        print(f"  skipped {n_skipped_label} rows (unrecognized emotion label)")
    print(f"\nNext step:\n"
          f"  python preprocess.py --audio_dir {wav_dir} --text_csv {manifest_path} "
          f"--out_dir {args.out_dir}/features")


if __name__ == "__main__":
    main()
