"""Central configuration and shared constants for TriSense.

This module is imported by *both* the FastAPI backend and the training
scripts so that label order, encoder identifiers and embedding sizes never
drift between training and inference.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Task definition
# --------------------------------------------------------------------------- #
# RAVDESS emotion order. The filename code (1..8) maps to ``EMOTIONS[code - 1]``.
EMOTIONS: list[str] = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]
NUM_CLASSES: int = len(EMOTIONS)

# Modalities, in canonical order used everywhere (embeddings, fusion, UI).
MODALITIES: list[str] = ["video", "audio", "text"]

# --------------------------------------------------------------------------- #
# Pretrained encoders (all free, no API keys)
# --------------------------------------------------------------------------- #
CLIP_MODEL_ID: str = "openai/clip-vit-base-patch32"
WAV2VEC2_MODEL_ID: str = "facebook/wav2vec2-base"
WHISPER_MODEL_ID: str = "openai/whisper-tiny"
TEXT_EMBED_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"

# Embedding dimensionality produced by each encoder (after pooling).
VIDEO_EMBED_DIM: int = 512   # CLIP get_image_features (projected)
AUDIO_EMBED_DIM: int = 768   # wav2vec2-base hidden size, mean-pooled
TEXT_EMBED_DIM: int = 384    # all-MiniLM-L6-v2
FUSION_INPUT_DIM: int = VIDEO_EMBED_DIM + AUDIO_EMBED_DIM + TEXT_EMBED_DIM

# Feature-extraction defaults.
NUM_VIDEO_FRAMES: int = 8        # frames sampled per clip
AUDIO_SAMPLE_RATE: int = 16_000  # Hz expected by wav2vec2 and whisper

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT: Path = REPO_ROOT / "checkpoints" / "trisense.pt"
ARTIFACTS_DIR: Path = REPO_ROOT / "artifacts"
GALLERY_DIR: Path = ARTIFACTS_DIR / "gallery"
MODEL_CARD_PATH: Path = ARTIFACTS_DIR / "model_card.json"


class Settings(BaseSettings):
    """Runtime settings, overridable via environment variables (``TRISENSE_*``)."""

    model_config = SettingsConfigDict(env_prefix="TRISENSE_", extra="ignore")

    device: str = "cpu"  # "cpu" or "cuda"
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    artifacts_dir: Path = ARTIFACTS_DIR

    # Upload validation.
    max_upload_mb: int = 50
    allowed_extensions: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv", ".webm")

    # Where Hugging Face weights are cached / baked. ``None`` -> default HF cache.
    hf_home: str | None = None


settings = Settings()
