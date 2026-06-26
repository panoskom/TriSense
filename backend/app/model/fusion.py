"""TriSenseModel: the tri-modal classifier with per-modality and fusion heads.

Architecture
------------
* CLIP vision tower  -> mean over frames -> 512-d video embedding
* wav2vec2           -> mean-pooled       -> 768-d audio embedding
* (text embedding precomputed upstream by the frozen Whisper+MiniLM branch)
* Per-modality heads: a linear classifier on each embedding.
* Fusion head: concatenate the three embeddings -> small MLP -> logits.

The encoders are frozen; LoRA adapters and all four heads are the trainable
parameters. ``encode`` and ``classify`` are split so training can cache frozen
embeddings (fast CPU path) or backprop through LoRA (GPU path).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..config import (
    AUDIO_EMBED_DIM,
    FUSION_INPUT_DIM,
    NUM_CLASSES,
    TEXT_EMBED_DIM,
    VIDEO_EMBED_DIM,
)
from ..encoders.audio import load_wav2vec2, mean_pool
from ..encoders.video import load_clip_vision
from .lora import LoraSettings, apply_lora, freeze


@dataclass
class ModelOutputs:
    """Logits and embeddings from one forward pass."""

    video_logits: torch.Tensor
    audio_logits: torch.Tensor
    text_logits: torch.Tensor
    fused_logits: torch.Tensor
    video_emb: torch.Tensor
    audio_emb: torch.Tensor
    text_emb: torch.Tensor


class TriSenseModel(nn.Module):
    """Tri-modal emotion classifier with LoRA-adapted encoders."""

    def __init__(self, lora: LoraSettings, fusion_hidden: int = 256) -> None:
        super().__init__()
        self.lora_settings = lora

        # --- Encoders (frozen base; optionally LoRA-adapted) ---
        clip_vision = load_clip_vision()
        wav2vec2 = load_wav2vec2()
        self.clip_vision = apply_lora(clip_vision, lora) if lora.video else freeze(clip_vision)
        self.wav2vec2 = apply_lora(wav2vec2, lora) if lora.audio else freeze(wav2vec2)

        # --- Per-modality classifier heads ---
        self.video_head = nn.Linear(VIDEO_EMBED_DIM, NUM_CLASSES)
        self.audio_head = nn.Linear(AUDIO_EMBED_DIM, NUM_CLASSES)
        self.text_head = nn.Linear(TEXT_EMBED_DIM, NUM_CLASSES)

        # --- Fusion head: concat -> MLP ---
        self.fusion_head = nn.Sequential(
            nn.Linear(FUSION_INPUT_DIM, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden, NUM_CLASSES),
        )

    # ------------------------------------------------------------------ #
    # Encoding (heavy; runs the pretrained towers)
    # ------------------------------------------------------------------ #
    def encode_video(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``[B, F, 3, H, W]`` frames -> ``[B, 512]`` averaged video embedding."""
        b, f = pixel_values.shape[:2]
        flat = pixel_values.flatten(0, 1)  # [B*F, 3, H, W]
        embeds = self.clip_vision(pixel_values=flat).image_embeds  # [B*F, 512]
        return embeds.view(b, f, -1).mean(dim=1)  # [B, 512]

    def encode_audio(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Raw waveform -> ``[B, 768]`` mean-pooled audio embedding."""
        out = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        return mean_pool(out.last_hidden_state, attention_mask)

    # ------------------------------------------------------------------ #
    # Classification (light; just the heads)
    # ------------------------------------------------------------------ #
    def classify(
        self,
        video_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> ModelOutputs:
        """Run the four heads on precomputed embeddings."""
        fused_in = torch.cat([video_emb, audio_emb, text_emb], dim=-1)
        return ModelOutputs(
            video_logits=self.video_head(video_emb),
            audio_logits=self.audio_head(audio_emb),
            text_logits=self.text_head(text_emb),
            fused_logits=self.fusion_head(fused_in),
            video_emb=video_emb,
            audio_emb=audio_emb,
            text_emb=text_emb,
        )

    def fuse_logits(
        self,
        video_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Fusion-head logits only — used for leave-one-modality-out probing."""
        return self.fusion_head(torch.cat([video_emb, audio_emb, text_emb], dim=-1))

    def forward(
        self,
        pixel_values: torch.Tensor,
        audio_values: torch.Tensor,
        audio_mask: torch.Tensor | None,
        text_emb: torch.Tensor,
    ) -> ModelOutputs:
        """Full forward from raw inputs (used when LoRA is being trained)."""
        video_emb = self.encode_video(pixel_values)
        audio_emb = self.encode_audio(audio_values, audio_mask)
        return self.classify(video_emb, audio_emb, text_emb)

    # ------------------------------------------------------------------ #
    # Checkpointing — only trainable params (LoRA adapters + heads)
    # ------------------------------------------------------------------ #
    def trainable_parameter_names(self) -> list[str]:
        return [n for n, p in self.named_parameters() if p.requires_grad]

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """State dict containing only trainable tensors (keeps checkpoints small)."""
        wanted = set(self.trainable_parameter_names())
        return {k: v.cpu() for k, v in self.state_dict().items() if k in wanted}

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
