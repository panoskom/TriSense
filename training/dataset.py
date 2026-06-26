"""Dataset utilities: cache preprocessed clip features and serve them.

Feature extraction (frame decode, audio extraction, Whisper transcription) is
the slow part, so it runs **once** and is cached to disk. Training then reads
the cached tensors. This makes both the frozen-encoder CPU path and the LoRA
GPU path cheap to iterate on.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.features.pipeline import FeatureExtractor  # noqa: E402


def read_manifest(manifest_path: Path) -> list[dict]:
    """Read the subset manifest CSV into a list of row dicts."""
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))


def build_cache(rows: list[dict], cache_dir: Path, device: str = "cpu") -> list[dict]:
    """Extract and cache preprocessed features for every clip in ``rows``.

    Returns metadata rows augmented with the on-disk cache path. Clips whose
    cache already exists are skipped.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    extractor: FeatureExtractor | None = None
    meta: list[dict] = []

    for row in tqdm(rows, desc="Caching features"):
        clip_id = Path(row["path"]).stem
        cache_path = cache_dir / f"{clip_id}.pt"
        if not cache_path.exists():
            if extractor is None:  # lazy — avoids loading models if cache is warm
                extractor = FeatureExtractor(device=device)
            try:
                bundle = extractor.extract(row["path"])
            except Exception as exc:  # noqa: BLE001
                print(f"  ! skipping {clip_id}: {exc}")
                continue
            torch.save(
                {
                    "pixel_values": bundle.pixel_values,
                    "audio_values": bundle.audio_values,
                    "audio_mask": bundle.audio_mask,
                    "text_emb": bundle.text_emb,
                    "transcript": bundle.transcript,
                },
                cache_path,
            )
        meta.append(
            {
                "clip_id": clip_id,
                "cache_path": str(cache_path),
                "path": row["path"],
                "emotion": row["emotion"],
                "label": int(row["emotion_idx"]),
                "split": row["split"],
            }
        )
    return meta


class ClipFeatureDataset(Dataset):
    """Serves cached preprocessed inputs for a given split."""

    def __init__(self, meta: list[dict], split: str) -> None:
        self.items = [m for m in meta if m["split"] == split]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        m = self.items[idx]
        cached = torch.load(m["cache_path"], weights_only=False)
        return {
            "pixel_values": cached["pixel_values"],
            "audio_values": cached["audio_values"],
            "audio_mask": cached["audio_mask"],
            "text_emb": cached["text_emb"],
            "label": torch.tensor(m["label"], dtype=torch.long),
            "clip_id": m["clip_id"],
        }


def collate(batch: list[dict]) -> dict:
    """Pad variable-length audio and stack the rest into a batch."""
    max_t = max(b["audio_values"].shape[0] for b in batch)
    audio = torch.zeros(len(batch), max_t)
    mask = torch.zeros(len(batch), max_t, dtype=torch.long)
    for i, b in enumerate(batch):
        t = b["audio_values"].shape[0]
        audio[i, :t] = b["audio_values"]
        mask[i, :t] = b["audio_mask"]
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "audio_values": audio,
        "audio_mask": mask,
        "text_emb": torch.stack([b["text_emb"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "clip_id": [b["clip_id"] for b in batch],
    }
