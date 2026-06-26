"""LoRA wiring — the core fine-tuning method of TriSense.

Instead of updating the millions of weights in CLIP and wav2vec2, we freeze
them and inject small low-rank adapters (LoRA) into their attention
projections. Only those adapters (plus the tiny classifier heads) are trained,
which is cheap, fast, and produces a small checkpoint.

The same target modules work for both encoders because CLIP's and wav2vec2's
attention blocks both expose ``q_proj``/``k_proj``/``v_proj``/``out_proj``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model

# Attention projections targeted by LoRA in both CLIP and wav2vec2.
LORA_TARGET_MODULES: list[str] = ["q_proj", "k_proj", "v_proj", "out_proj"]


@dataclass
class LoraSettings:
    """Serializable description of the LoRA configuration.

    Persisted inside the checkpoint so inference can rebuild the exact same
    adapter shapes before loading the trained weights.
    """

    video: bool = True   # apply LoRA to the CLIP vision tower
    audio: bool = True   # apply LoRA to wav2vec2
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(LORA_TARGET_MODULES))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LoraSettings":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def _config(settings: LoraSettings) -> LoraConfig:
    # No ``task_type``: we use the low-level injector, so the model keeps its
    # original forward signature and we avoid task-specific prep that some
    # vision towers (e.g. CLIP's projection model) don't support.
    return LoraConfig(
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=settings.target_modules,
        bias="none",
    )


def apply_lora(encoder: nn.Module, settings: LoraSettings) -> nn.Module:
    """Inject LoRA adapters into an encoder and freeze its base weights.

    Uses peft's :func:`inject_adapter_in_model` so the encoder keeps its
    original forward signature (the fusion model calls it unchanged). Every
    non-LoRA parameter is explicitly frozen so only the adapters train.
    """
    model = inject_adapter_in_model(_config(settings), encoder)
    for name, param in model.named_parameters():
        param.requires_grad_("lora_" in name)
    return model


def freeze(module: nn.Module) -> nn.Module:
    """Freeze every parameter of a module (used when LoRA is disabled)."""
    for p in module.parameters():
        p.requires_grad_(False)
    return module
