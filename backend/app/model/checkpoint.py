"""Save/load helpers for the small TriSense checkpoint.

Only trainable tensors (LoRA adapters + the four heads) are stored, alongside
the LoRA settings needed to rebuild the model. The frozen pretrained weights
are never saved — they are loaded from the baked Hugging Face cache.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..config import EMOTIONS
from .fusion import TriSenseModel
from .lora import LoraSettings

CHECKPOINT_FORMAT = 1


def save_checkpoint(model: TriSenseModel, path: str | Path, meta: dict | None = None) -> None:
    """Persist trainable weights + config to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "trainable_state_dict": model.trainable_state_dict(),
        "lora": model.lora_settings.to_dict(),
        "fusion_hidden": model.fusion_head[0].out_features,
        "emotions": EMOTIONS,
        "meta": meta or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[TriSenseModel, dict]:
    """Rebuild a :class:`TriSenseModel` and load trainable weights from ``path``.

    Returns the eval-mode model on ``device`` and the checkpoint's ``meta`` dict.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    lora = LoraSettings.from_dict(payload.get("lora", {}))
    fusion_hidden = int(payload.get("fusion_hidden", 256))

    model = TriSenseModel(lora=lora, fusion_hidden=fusion_hidden)
    _missing, unexpected = model.load_state_dict(payload["trainable_state_dict"], strict=False)
    # ``_missing`` is expected: it is every frozen base-encoder weight (not saved).
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:5]} ...")

    model.to(device)
    model.eval()
    return model, payload.get("meta", {})
