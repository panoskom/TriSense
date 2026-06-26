"""Train TriSense and log everything to MLflow.

Two paths share this script:

* **Frozen-encoder CPU path** (phase one): no LoRA. Embeddings are computed
  once and only the four heads are trained — finishes in a couple of minutes.
* **LoRA GPU path** (phase two): LoRA adapters on CLIP and/or wav2vec2 are
  trained jointly with the heads, with optional mixed precision.

Either way the script logs params, metrics, loss curves, the confusion matrix,
the model card and the checkpoint to MLflow, then writes the committed
artifacts (checkpoint, model_card.json, confusion matrix, results gallery).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.config import (  # noqa: E402
    ARTIFACTS_DIR,
    DEFAULT_CHECKPOINT,
    EMOTIONS,
    Settings,
)
from backend.app.model.checkpoint import save_checkpoint  # noqa: E402
from backend.app.model.fusion import TriSenseModel  # noqa: E402
from backend.app.model.lora import LoraSettings  # noqa: E402
from backend.app.service.inference import InferenceService  # noqa: E402

from dataset import ClipFeatureDataset, build_cache, collate, read_manifest  # noqa: E402
from metrics import build_gallery, evaluate, save_confusion_matrix, write_model_card  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def multitask_loss(out, labels: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    """Fused loss + 0.5 * each modality head loss (multi-task supervision)."""
    return (
        criterion(out.fused_logits, labels)
        + 0.5 * criterion(out.video_logits, labels)
        + 0.5 * criterion(out.audio_logits, labels)
        + 0.5 * criterion(out.text_logits, labels)
    )


@torch.no_grad()
def precompute_embeddings(model: TriSenseModel, dataset, device: str, batch_size: int) -> TensorDataset:
    """Run the frozen encoders once to cache embeddings for fast head training."""
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    v, a, t, y = [], [], [], []
    model.eval()
    for batch in tqdm(loader, desc="Precomputing embeddings"):
        v.append(model.encode_video(batch["pixel_values"].to(device)).cpu())
        a.append(model.encode_audio(batch["audio_values"].to(device), batch["audio_mask"].to(device)).cpu())
        t.append(batch["text_emb"])
        y.append(batch["label"])
    return TensorDataset(torch.cat(v), torch.cat(a), torch.cat(t), torch.cat(y))


def train_frozen(model, train_ds, device, args) -> list[float]:
    """Head-only training on cached embeddings (CPU phase-one path)."""
    emb = precompute_embeddings(model, train_ds, device, args.batch_size)
    loader = DataLoader(emb, batch_size=args.batch_size, shuffle=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for v, a, t, y in loader:
            v, a, t, y = v.to(device), a.to(device), t.to(device), y.to(device)
            out = model.classify(v, a, t)
            loss = multitask_loss(out, y, criterion)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
        epoch_loss = running / len(emb)
        losses.append(epoch_loss)
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        print(f"  epoch {epoch + 1}/{args.epochs}  loss={epoch_loss:.4f}")
    return losses


def train_lora(model, train_ds, device, args) -> list[float]:
    """End-to-end LoRA training through the encoders (GPU phase-two path)."""
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    use_amp = args.amp and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    losses = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            y = batch["label"].to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                out = model(
                    batch["pixel_values"].to(device),
                    batch["audio_values"].to(device),
                    batch["audio_mask"].to(device),
                    batch["text_emb"].to(device),
                )
                loss = multitask_loss(out, y, criterion)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * len(y)
        epoch_loss = running / len(train_ds)
        losses.append(epoch_loss)
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        print(f"  epoch {epoch + 1}/{args.epochs}  loss={epoch_loss:.4f}")
    return losses


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TriSense (LoRA tri-modal emotion recognition).")
    p.add_argument("--manifest", type=Path, default=Path("data/subset/manifest.csv"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/subset/cache"))
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    # LoRA controls (the core method).
    p.add_argument("--lora-video", action="store_true", help="Apply LoRA to CLIP.")
    p.add_argument("--lora-audio", action="store_true", help="Apply LoRA to wav2vec2.")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--amp", action="store_true", help="Mixed precision (GPU only).")
    p.add_argument("--fusion-hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment", default="trisense")
    p.add_argument("--run-name", default=None)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--gallery-size", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = args.device

    rows = read_manifest(args.manifest)
    meta = build_cache(rows, args.cache_dir, device=device)

    lora = LoraSettings(
        video=args.lora_video,
        audio=args.lora_audio,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    use_lora = lora.video or lora.audio
    model = TriSenseModel(lora=lora, fusion_hidden=args.fusion_hidden).to(device)
    print(
        f"Trainable params: {model.num_trainable_params():,} "
        f"(LoRA video={lora.video}, audio={lora.audio}, frozen-path={not use_lora})"
    )

    train_ds = ClipFeatureDataset(meta, "train")
    val_ds = ClipFeatureDataset(meta, "val")
    test_ds = ClipFeatureDataset(meta, "test")

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "device": device,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "lora_video": lora.video,
                "lora_audio": lora.audio,
                "lora_rank": lora.rank,
                "lora_alpha": lora.alpha,
                "amp": args.amp,
                "fusion_hidden": args.fusion_hidden,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "n_test": len(test_ds),
                "trainable_params": model.num_trainable_params(),
            }
        )

        trainer = train_lora if use_lora else train_frozen
        trainer(model, train_ds, device, args)

        # --- Validation & test evaluation ---
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate)
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        print(
            f"VAL  acc={val_metrics['accuracy']:.3f} f1={val_metrics['macro_f1']:.3f} | "
            f"TEST acc={test_metrics['accuracy']:.3f} f1={test_metrics['macro_f1']:.3f}"
        )
        mlflow.log_metrics(
            {
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_acc_video": test_metrics["per_modality_accuracy"]["video"],
                "test_acc_audio": test_metrics["per_modality_accuracy"]["audio"],
                "test_acc_text": test_metrics["per_modality_accuracy"]["text"],
            }
        )

        # --- Save checkpoint (trainable weights only) ---
        meta_info = {
            "trained_on": f"RAVDESS subset, {len(train_ds)} train clips",
            "lora_video": lora.video,
            "lora_audio": lora.audio,
            "device": device,
        }
        save_checkpoint(model, args.checkpoint, meta=meta_info)
        print(f"Saved checkpoint -> {args.checkpoint}")

        # --- Artifacts: confusion matrix + model card ---
        cm_path = ARTIFACTS_DIR / "confusion_matrix.png"
        save_confusion_matrix(test_metrics["confusion_matrix"], cm_path)
        note = (
            "Quick CPU checkpoint trained on a tiny RAVDESS subset to make the app "
            "run end-to-end. Re-run on GPU with LoRA for stronger numbers."
            if not use_lora
            else "LoRA fine-tuned on RAVDESS subset."
        )
        trained_on = (
            f"RAVDESS speech AV subset — {len(train_ds)} train / {len(val_ds)} val / "
            f"{len(test_ds)} test clips, {len(EMOTIONS)} emotions."
        )
        card = write_model_card(
            test_metrics,
            checkpoint_name=args.checkpoint.name,
            trained_on=trained_on,
            device=device,
            note=note,
            confusion_image_url="/api/artifacts/confusion_matrix.png",
        )

        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(args.checkpoint))
        mlflow.log_dict(card, "model_card.json")

        # --- Results gallery (served read-only by the app) ---
        print("Building results gallery...")
        settings = Settings(device=device, checkpoint_path=args.checkpoint)
        service = InferenceService(settings)
        test_rows = [m for m in meta if m["split"] == "test"]
        gallery = build_gallery(service, test_rows, max_items=args.gallery_size)
        mlflow.log_artifact(str(ARTIFACTS_DIR / "gallery" / "gallery.json"))
        print(f"Gallery: {len(gallery)} clips.")

    print("Done.")


if __name__ == "__main__":
    main()
