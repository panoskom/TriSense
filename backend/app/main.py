"""TriSense FastAPI application.

Serves the JSON API under ``/api`` and the built React SPA at ``/``. The model
is loaded once at startup from the committed checkpoint; the app never trains.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router
from .config import ARTIFACTS_DIR, settings
from .service.inference import InferenceService

logger = logging.getLogger("trisense")
logging.basicConfig(level=logging.INFO)

# Directory holding the built frontend (populated by the Docker build / Vite).
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model checkpoint once on startup."""
    try:
        app.state.service = InferenceService(settings)
        logger.info(
            "TriSense model loaded from %s on %s", settings.checkpoint_path, settings.device
        )
    except Exception as exc:  # noqa: BLE001
        app.state.service = None
        logger.error("Failed to load model checkpoint: %s", exc)
    yield


app = FastAPI(
    title="TriSense",
    description="Tri-modal (video + audio + text) emotion recognition with LoRA fine-tuning.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS is convenient for the Vite dev server (port 5173) hitting the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve committed artifacts (confusion matrix PNG, gallery thumbnails) read-only.
if ARTIFACTS_DIR.exists():
    app.mount("/api/artifacts", StaticFiles(directory=ARTIFACTS_DIR), name="artifacts")


# --------------------------------------------------------------------------- #
# SPA: serve the built React app and let client-side routing handle unknown
# (non-/api) paths by falling back to index.html.
# --------------------------------------------------------------------------- #
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str) -> FileResponse:
        candidate = FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
else:

    @app.get("/")
    def placeholder() -> dict:
        return {
            "message": "TriSense API is running. Frontend build not found; "
            "see /docs for the API.",
        }
