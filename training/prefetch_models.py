"""Download every pretrained encoder into the Hugging Face cache.

Run once locally before training, and during the Docker build so the image is
fully offline at runtime. Honours ``HF_HOME`` if set.
"""
from __future__ import annotations

import sys

# Make the backend package importable when run from the repo root.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from backend.app.config import (  # noqa: E402
    CLIP_MODEL_ID,
    TEXT_EMBED_MODEL_ID,
    WAV2VEC2_MODEL_ID,
    WHISPER_MODEL_ID,
)


def main() -> None:
    print("Prefetching pretrained encoders (this downloads ~1.5 GB once)...")

    from transformers import (
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2Model,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    print(f"  - CLIP vision: {CLIP_MODEL_ID}")
    CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)
    CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID)

    print(f"  - wav2vec2: {WAV2VEC2_MODEL_ID}")
    Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_MODEL_ID)
    Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL_ID)

    print(f"  - whisper: {WHISPER_MODEL_ID}")
    WhisperProcessor.from_pretrained(WHISPER_MODEL_ID)
    WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_ID)

    print(f"  - text embedder: {TEXT_EMBED_MODEL_ID}")
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(TEXT_EMBED_MODEL_ID)

    print("All encoders cached.")


if __name__ == "__main__":
    main()
