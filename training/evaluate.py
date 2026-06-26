"""Evaluate a committed checkpoint and (re)write the model card + confusion matrix.

Handy after the real GPU run: regenerates ``artifacts/model_card.json`` and the
confusion matrix PNG from the served checkpoint without retraining.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.config import ARTIFACTS_DIR, DEFAULT_CHECKPOINT, EMOTIONS  # noqa: E402
from backend.app.model.checkpoint import load_checkpoint  # noqa: E402

from dataset import ClipFeatureDataset, build_cache, collate, read_manifest  # noqa: E402
from metrics import evaluate, save_confusion_matrix, write_model_card  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a TriSense checkpoint on the manifest test split.")
    p.add_argument("--manifest", type=Path, default=Path("data/subset/manifest.csv"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/subset/cache"))
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    rows = read_manifest(args.manifest)
    meta = build_cache(rows, args.cache_dir, device=args.device)
    model, ckpt_meta = load_checkpoint(args.checkpoint, device=args.device)

    test_ds = ClipFeatureDataset(meta, "test")
    loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate)
    metrics = evaluate(model, loader, args.device)
    print(
        f"TEST  acc={metrics['accuracy']:.3f}  macro_f1={metrics['macro_f1']:.3f}  "
        f"(n={metrics['n']}, per-modality={metrics['per_modality_accuracy']})"
    )

    cm_path = ARTIFACTS_DIR / "confusion_matrix.png"
    save_confusion_matrix(metrics["confusion_matrix"], cm_path)
    write_model_card(
        metrics,
        checkpoint_name=args.checkpoint.name,
        trained_on=ckpt_meta.get("trained_on", f"RAVDESS subset, {len(EMOTIONS)} emotions"),
        device=args.device,
        note=ckpt_meta.get("note", ""),
        confusion_image_url="/api/artifacts/confusion_matrix.png",
    )
    print("Wrote model card + confusion matrix.")


if __name__ == "__main__":
    main()
