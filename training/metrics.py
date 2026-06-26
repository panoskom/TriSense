"""Evaluation, confusion matrix, model card, and results-gallery generation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.config import EMOTIONS, MODEL_CARD_PATH, GALLERY_DIR  # noqa: E402


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str) -> dict:
    """Run the model over a loader and compute fused + per-modality metrics."""
    model.eval()
    fused_pred, v_pred, a_pred, t_pred, labels = [], [], [], [], []
    for batch in loader:
        out = model(
            batch["pixel_values"].to(device),
            batch["audio_values"].to(device),
            batch["audio_mask"].to(device),
            batch["text_emb"].to(device),
        )
        fused_pred += out.fused_logits.argmax(-1).cpu().tolist()
        v_pred += out.video_logits.argmax(-1).cpu().tolist()
        a_pred += out.audio_logits.argmax(-1).cpu().tolist()
        t_pred += out.text_logits.argmax(-1).cpu().tolist()
        labels += batch["label"].tolist()

    labels_np = np.array(labels)

    def acc(pred: list[int]) -> float:
        return float((np.array(pred) == labels_np).mean()) if labels else 0.0

    cm = confusion_matrix(labels, fused_pred, labels=list(range(len(EMOTIONS))))
    return {
        "n": len(labels),
        "accuracy": acc(fused_pred),
        "macro_f1": float(f1_score(labels, fused_pred, average="macro", zero_division=0)) if labels else 0.0,
        "per_modality_accuracy": {
            "video": acc(v_pred),
            "audio": acc(a_pred),
            "text": acc(t_pred),
        },
        "confusion_matrix": cm.tolist(),
    }


def save_confusion_matrix(cm: list[list[int]], path: Path) -> None:
    """Render a labelled confusion-matrix PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_xticklabels(EMOTIONS, rotation=45, ha="right")
    ax.set_yticklabels(EMOTIONS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("TriSense — fused confusion matrix (held-out test)")
    thresh = cm_arr.max() / 2 if cm_arr.max() else 0.5
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j, i, str(int(cm_arr[i, j])),
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_model_card(
    metrics: dict,
    checkpoint_name: str,
    trained_on: str,
    device: str,
    note: str,
    confusion_image_url: str | None,
) -> dict:
    """Write ``artifacts/model_card.json`` from evaluation metrics."""
    card = {
        "checkpoint": checkpoint_name,
        "trained_on": trained_on,
        "device": device,
        "num_test_clips": metrics["n"],
        "test_accuracy": round(metrics["accuracy"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "per_modality_accuracy": {k: round(v, 4) for k, v in metrics["per_modality_accuracy"].items()},
        "confusion_matrix": {"labels": EMOTIONS, "matrix": metrics["confusion_matrix"]},
        "confusion_matrix_image": confusion_image_url,
        "random_baseline": round(1.0 / len(EMOTIONS), 4),
        "note": note,
    }
    MODEL_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_CARD_PATH.write_text(json.dumps(card, indent=2))
    return card


def build_gallery(service, test_rows: list[dict], max_items: int = 8) -> list[dict]:
    """Run inference on a handful of held-out clips and write the gallery.

    ``service`` is an :class:`InferenceService`. ``test_rows`` come from the
    manifest (each has 'path' and 'emotion'). One representative frame per clip
    is saved as a thumbnail under ``artifacts/gallery``.
    """
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    # Pick a balanced-ish sample: first occurrence of distinct emotions, then fill.
    seen: set[str] = set()
    ordered = [r for r in test_rows if not (r["emotion"] in seen or seen.add(r["emotion"]))]
    for r in test_rows:
        if len(ordered) >= max_items:
            break
        if r not in ordered:
            ordered.append(r)
    ordered = ordered[:max_items]

    items: list[dict] = []
    for r in ordered:
        clip_id = Path(r["path"]).stem
        result = service.analyze(r["path"])

        # Save a representative frame (middle of the sampled set) as a thumbnail.
        thumb_rel = f"gallery/{clip_id}.jpg"
        _save_thumbnail(result.frames[len(result.frames) // 2], GALLERY_DIR / f"{clip_id}.jpg")

        items.append(
            {
                "clip_id": clip_id,
                "true_emotion": r["emotion"],
                "predicted_emotion": result.predicted_emotion,
                "confidence": round(result.confidence, 4),
                "correct": result.predicted_emotion == r["emotion"],
                "modality_predictions": {
                    k: v.model_dump() for k, v in result.modality_predictions.items()
                },
                "contributions": {k: round(v, 4) for k, v in result.contributions.items()},
                "transcript": result.transcript,
                "thumbnail": f"/api/artifacts/{thumb_rel}",
            }
        )

    (GALLERY_DIR / "gallery.json").write_text(json.dumps(items, indent=2))
    return items


def _save_thumbnail(data_uri: str, path: Path) -> None:
    """Decode a base64 data-URI JPEG and write it to ``path``."""
    import base64

    b64 = data_uri.split(",", 1)[1]
    path.write_bytes(base64.b64decode(b64))
