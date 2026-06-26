"""Audio encoder: wav2vec2-base.

The raw 16 kHz waveform is encoded with wav2vec2 and mean-pooled (over valid
time steps) into one 768-d audio embedding. wav2vec2 is the LoRA fine-tuning
target for the audio path.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from ..config import AUDIO_SAMPLE_RATE, WAV2VEC2_MODEL_ID


def load_wav2vec2_extractor() -> Wav2Vec2FeatureExtractor:
    """Load the wav2vec2 feature extractor (handles normalisation/padding)."""
    return Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_MODEL_ID)


def load_wav2vec2() -> Wav2Vec2Model:
    """Load the wav2vec2-base encoder."""
    return Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL_ID)


def preprocess_audio(
    extractor: Wav2Vec2FeatureExtractor, waveform: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a mono waveform into ``(input_values, attention_mask)`` tensors."""
    out = extractor(
        waveform,
        sampling_rate=AUDIO_SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True,
        padding=True,
    )
    mask = out.get("attention_mask")
    if mask is None:
        mask = torch.ones_like(out["input_values"], dtype=torch.long)
    return out["input_values"], mask


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool wav2vec2 hidden states over valid time steps -> ``[B, 768]``.

    The feature-extractor mask is at the *waveform* resolution, so it is
    down-sampled to the hidden-state length before pooling.
    """
    if attention_mask is None:
        return last_hidden_state.mean(dim=1)

    out_len = last_hidden_state.shape[1]
    # Down-sample the waveform-resolution mask to the hidden length.
    idx = torch.linspace(0, attention_mask.shape[1] - 1, steps=out_len, device=attention_mask.device)
    pooled_mask = attention_mask[:, idx.long()].unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * pooled_mask).sum(dim=1)
    counts = pooled_mask.sum(dim=1).clamp(min=1.0)
    return summed / counts
