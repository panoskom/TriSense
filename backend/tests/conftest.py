"""Shared pytest fixtures.

Models are loaded once per session (they are cached locally). A throwaway,
untrained checkpoint is created in a temp dir so inference can be exercised
without depending on the committed checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.config import Settings
from backend.app.model.checkpoint import save_checkpoint
from backend.app.model.fusion import TriSenseModel
from backend.app.model.lora import LoraSettings
from backend.app.service.inference import InferenceService

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def tiny_clip() -> Path:
    """Path to the tiny committed test clip (video + audio)."""
    path = FIXTURES / "tiny.mp4"
    assert path.exists(), "tiny.mp4 fixture is missing"
    return path


@pytest.fixture(scope="session")
def service(tmp_path_factory) -> InferenceService:
    """An InferenceService backed by a fresh untrained checkpoint."""
    ckpt = tmp_path_factory.mktemp("ckpt") / "trisense_test.pt"
    model = TriSenseModel(LoraSettings(video=False, audio=False))
    save_checkpoint(model, ckpt, meta={"trained_on": "untrained test fixture"})
    return InferenceService(Settings(device="cpu", checkpoint_path=ckpt))


@pytest.fixture(scope="session")
def extractor(service: InferenceService):
    """Reuse the service's feature extractor (avoids loading models twice)."""
    return service.extractor
