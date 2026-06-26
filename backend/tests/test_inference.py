"""Tests for the inference service and checkpoint round-trip."""
from __future__ import annotations

import math

import torch

from backend.app.config import EMOTIONS, MODALITIES, NUM_VIDEO_FRAMES
from backend.app.model.checkpoint import load_checkpoint, save_checkpoint
from backend.app.model.fusion import TriSenseModel
from backend.app.model.lora import LoraSettings


def test_analyze_contract(service, tiny_clip):
    result = service.analyze(tiny_clip)

    # Fused prediction is a valid emotion with a calibrated probability.
    assert result.predicted_emotion in EMOTIONS
    assert 0.0 <= result.confidence <= 1.0
    assert math.isclose(sum(result.probabilities.values()), 1.0, abs_tol=1e-3)
    assert set(result.probabilities) == set(EMOTIONS)

    # Per-modality predictions for all three modalities.
    assert set(result.modality_predictions) == set(MODALITIES)
    for pred in result.modality_predictions.values():
        assert pred.emotion in EMOTIONS

    # Leave-one-out contributions: one per modality.
    assert set(result.contributions) == set(MODALITIES)

    # Explainability extras.
    assert len(result.frames) == NUM_VIDEO_FRAMES
    assert all(f.startswith("data:image/jpeg;base64,") for f in result.frames)
    assert isinstance(result.transcript, str)
    assert result.inference_ms >= 0.0


def test_checkpoint_roundtrip(tmp_path):
    model = TriSenseModel(LoraSettings(video=False, audio=False)).eval()
    ckpt = tmp_path / "rt.pt"
    save_checkpoint(model, ckpt, meta={"trained_on": "roundtrip"})

    reloaded, meta = load_checkpoint(ckpt, device="cpu")
    assert meta["trained_on"] == "roundtrip"

    B, F = 1, NUM_VIDEO_FRAMES
    pix = torch.randn(B, F, 3, 224, 224)
    aud = torch.randn(B, 8000)
    mask = torch.ones(B, 8000, dtype=torch.long)
    txt = torch.randn(B, 384)
    with torch.no_grad():
        a = model(pix, aud, mask, txt).fused_logits
        b = reloaded(pix, aud, mask, txt).fused_logits
    assert torch.allclose(a, b, atol=1e-5)
