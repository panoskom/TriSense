"""HTTP API routes for TriSense."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from ..config import MODEL_CARD_PATH, GALLERY_DIR, settings
from ..schemas import AnalyzeResponse, GalleryItem, HealthResponse, ModelCard
from ..service.inference import InferenceService

router = APIRouter(prefix="/api")


def get_service(request: Request) -> InferenceService:
    """Return the singleton inference service, or 503 if it failed to load."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train a checkpoint first.")
    return service


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Liveness/readiness probe."""
    service = getattr(request.app.state, "service", None)
    return HealthResponse(
        status="ok",
        model_loaded=service is not None,
        device=settings.device,
        checkpoint=settings.checkpoint_path.name,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    service: InferenceService = Depends(get_service),
) -> AnalyzeResponse:
    """Analyse an uploaded clip and return the explainable prediction."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(settings.allowed_extensions)}",
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large (> {settings.max_upload_mb} MB).",
                )
            tmp.write(chunk)

    if size == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty upload.")

    try:
        return service.analyze(tmp_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Could not process clip: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surface a clean 500 to the client
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/model-card", response_model=ModelCard)
def model_card() -> ModelCard:
    """Return the committed model card (held-out accuracy, confusion matrix)."""
    if not MODEL_CARD_PATH.exists():
        raise HTTPException(status_code=404, detail="Model card not found. Run training first.")
    data = json.loads(MODEL_CARD_PATH.read_text())
    return ModelCard(**data)


@router.get("/gallery", response_model=list[GalleryItem])
def gallery() -> list[GalleryItem]:
    """Return the pre-computed held-out results gallery (empty if not built)."""
    gallery_json = GALLERY_DIR / "gallery.json"
    if not gallery_json.exists():
        return []
    items = json.loads(gallery_json.read_text())
    return [GalleryItem(**item) for item in items]
