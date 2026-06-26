"""Video encoder: OpenAI CLIP image tower.

A handful of frames are sampled from the clip, each encoded with the CLIP
vision tower, and the per-frame embeddings are averaged into one 512-d video
embedding. The vision tower is the LoRA fine-tuning target for the video path.
"""
from __future__ import annotations

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from ..config import CLIP_MODEL_ID


def load_clip_processor() -> CLIPImageProcessor:
    """Load the CLIP image processor used to preprocess sampled frames."""
    return CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)


def load_clip_vision() -> CLIPVisionModelWithProjection:
    """Load the CLIP vision tower (with projection head -> 512-d embeddings)."""
    return CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID)


def preprocess_frames(processor: CLIPImageProcessor, frames: list[Image.Image]) -> torch.Tensor:
    """Turn a list of PIL frames into a ``[num_frames, 3, H, W]`` pixel tensor."""
    out = processor(images=frames, return_tensors="pt")
    return out["pixel_values"]
