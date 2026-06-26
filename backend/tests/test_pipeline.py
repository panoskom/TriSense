"""Tests for the feature-extraction pipeline using the tiny fixture clip."""
from __future__ import annotations

import numpy as np

from backend.app.config import AUDIO_SAMPLE_RATE, NUM_VIDEO_FRAMES, TEXT_EMBED_DIM
from backend.app.features.pipeline import extract_audio, sample_frames


def test_sample_frames(tiny_clip):
    frames = sample_frames(tiny_clip, num_frames=NUM_VIDEO_FRAMES)
    assert len(frames) == NUM_VIDEO_FRAMES
    assert all(f.mode == "RGB" for f in frames)
    assert frames[0].size[0] > 0 and frames[0].size[1] > 0


def test_extract_audio(tiny_clip):
    wav = extract_audio(tiny_clip, sample_rate=AUDIO_SAMPLE_RATE)
    assert isinstance(wav, np.ndarray)
    assert wav.ndim == 1
    assert wav.size > AUDIO_SAMPLE_RATE // 2  # ~1.2s clip -> plenty of samples


def test_feature_extractor_shapes(extractor, tiny_clip):
    bundle = extractor.extract(tiny_clip)
    assert bundle.pixel_values.shape[0] == NUM_VIDEO_FRAMES
    assert bundle.pixel_values.shape[1] == 3  # channels
    assert bundle.text_emb.shape == (TEXT_EMBED_DIM,)
    assert isinstance(bundle.transcript, str)
    assert bundle.audio_values.ndim == 1
    assert bundle.audio_mask.shape == bundle.audio_values.shape
