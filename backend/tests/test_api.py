"""HTTP-level tests using FastAPI's TestClient with the fixture service."""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.config import EMOTIONS
from backend.app.main import app


def test_health_and_analyze(service, tiny_clip):
    with TestClient(app) as client:
        # Inject the test service (avoids depending on the committed checkpoint).
        app.state.service = service

        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        with open(tiny_clip, "rb") as f:
            resp = client.post("/api/analyze", files={"file": ("tiny.mp4", f, "video/mp4")})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["predicted_emotion"] in EMOTIONS
        assert set(body["contributions"]) == {"video", "audio", "text"}


def test_rejects_bad_extension(service):
    with TestClient(app) as client:
        app.state.service = service
        resp = client.post("/api/analyze", files={"file": ("note.txt", b"hello", "text/plain")})
        assert resp.status_code == 415
