"""Regenerate the held-out results gallery from a committed checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.config import DEFAULT_CHECKPOINT, Settings  # noqa: E402
from backend.app.service.inference import InferenceService  # noqa: E402

from dataset import read_manifest  # noqa: E402
from metrics import build_gallery  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Build the TriSense results gallery.")
    p.add_argument("--manifest", type=Path, default=Path("data/subset/manifest.csv"))
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--size", type=int, default=8)
    args = p.parse_args()

    rows = read_manifest(args.manifest)
    test_rows = [r for r in rows if r["split"] == "test"]
    service = InferenceService(Settings(device=args.device, checkpoint_path=args.checkpoint))
    items = build_gallery(service, test_rows, max_items=args.size)
    print(f"Built gallery with {len(items)} clips.")


if __name__ == "__main__":
    main()
