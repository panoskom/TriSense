"""Download RAVDESS and build a small balanced subset for training.

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) is free on
Zenodo (record 1188976). We use the **speech**, **audio-visual** files only, so
every clip has a face *and* a voice for the tri-modal pipeline.

Filename convention (7 dash-separated fields), e.g. ``01-01-06-01-02-01-12.mp4``:
    modality (01=AV, 02=video-only, 03=audio-only)
    vocal channel (01=speech, 02=song)
    emotion (01..08)            <- our label
    intensity, statement, repetition, actor

We keep modality==01 (audio-visual) and vocal==01 (speech).

The raw dataset is never committed (see .gitignore); only the manifest and the
trained checkpoint are. Subset size and data path are configurable so the same
script builds the larger subset for the real GPU run.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.config import EMOTIONS  # noqa: E402

ZENODO_BASE = "https://zenodo.org/records/1188976/files"


def download_actor(actor: int, zips_dir: Path) -> Path:
    """Download one actor's audio-visual speech zip (resumable)."""
    name = f"Video_Speech_Actor_{actor:02d}.zip"
    dest = zips_dir / name
    url = f"{ZENODO_BASE}/{name}"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  [{actor:02d}] already downloaded")
        return dest

    print(f"  [{actor:02d}] downloading {name} ...")
    headers = {}
    mode = "wb"
    if dest.exists():
        headers["Range"] = f"bytes={dest.stat().st_size}-"
        mode = "ab"
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return dest


def extract_actor(zip_path: Path, raw_dir: Path) -> None:
    """Unzip an actor archive into ``raw_dir`` (idempotent)."""
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)


def find_av_speech_clips(raw_dir: Path) -> list[Path]:
    """Find audio-visual speech mp4s (modality 01, vocal 01)."""
    clips: list[Path] = []
    for mp4 in raw_dir.rglob("*.mp4"):
        parts = mp4.stem.split("-")
        if len(parts) != 7:
            continue
        modality, vocal = parts[0], parts[1]
        if modality == "01" and vocal == "01":
            clips.append(mp4)
    return clips


def emotion_of(mp4: Path) -> int:
    """0-based emotion index from the filename (field 3 is 1..8)."""
    return int(mp4.stem.split("-")[2]) - 1


def build_subset(
    clips: list[Path],
    per_class: int,
    seed: int,
    splits: tuple[float, float, float],
) -> list[dict]:
    """Balanced, stratified subset with train/val/test split assignments."""
    rng = random.Random(seed)
    by_emotion: dict[int, list[Path]] = {i: [] for i in range(len(EMOTIONS))}
    for c in clips:
        by_emotion[emotion_of(c)].append(c)

    rows: list[dict] = []
    train_f, val_f, _ = splits
    for emo_idx, paths in by_emotion.items():
        rng.shuffle(paths)
        chosen = paths[:per_class]
        n = len(chosen)
        n_train = int(n * train_f)
        n_val = int(n * val_f)
        for i, p in enumerate(chosen):
            split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
            rows.append(
                {
                    "path": str(p.resolve()),
                    "emotion": EMOTIONS[emo_idx],
                    "emotion_idx": emo_idx,
                    "actor": p.stem.split("-")[6],
                    "split": split,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RAVDESS and build a subset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--actors", type=int, default=10, help="Use actors 1..N.")
    parser.add_argument("--per-class", type=int, default=40, help="Clips per emotion (<=80).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    zips_dir = data_dir / "zips"
    raw_dir = data_dir / "raw"
    subset_dir = data_dir / "subset"
    for d in (zips_dir, raw_dir, subset_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.actors} actor(s) from RAVDESS...")
    for actor in range(1, args.actors + 1):
        zip_path = download_actor(actor, zips_dir)
        extract_actor(zip_path, raw_dir)

    clips = find_av_speech_clips(raw_dir)
    print(f"Found {len(clips)} audio-visual speech clips.")
    if not clips:
        raise SystemExit("No AV speech clips found — check the download.")

    test_frac = 1.0 - args.train_frac - args.val_frac
    rows = build_subset(clips, args.per_class, args.seed, (args.train_frac, args.val_frac, test_frac))

    manifest = subset_dir / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "emotion", "emotion_idx", "actor", "split"])
        writer.writeheader()
        writer.writerows(rows)

    # Report balance.
    from collections import Counter

    by_split = Counter(r["split"] for r in rows)
    by_emo = Counter(r["emotion"] for r in rows)
    print(f"Wrote manifest with {len(rows)} clips -> {manifest}")
    print(f"  splits: {dict(by_split)}")
    print(f"  per emotion: {dict(by_emo)}")


if __name__ == "__main__":
    main()
