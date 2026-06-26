# syntax=docker/dockerfile:1
# ----------------------------------------------------------------------------
# TriSense — single multi-stage image: build the React SPA, then serve it plus
# the FastAPI backend on CPU. Pretrained weights + the trained checkpoint are
# baked in, so the container needs no internet and no GPU at run time.
# ----------------------------------------------------------------------------

# --- Stage 1: build the frontend ------------------------------------------- #
FROM node:20-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build           # -> /frontend/dist

# --- Stage 2: backend + baked models --------------------------------------- #
FROM python:3.11-slim AS app

# System libs: libsndfile for soundfile/librosa, libgomp for torch/sklearn.
# (Video frames use opencv-headless; audio uses the pip-packaged ffmpeg binary.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    TRISENSE_DEVICE=cpu

WORKDIR /app

# Install Python deps (CPU torch) first for better layer caching.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r backend/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# App code, committed checkpoint, artifacts, and the model-prefetch script.
COPY backend/ backend/
COPY training/prefetch_models.py training/prefetch_models.py
COPY checkpoints/ checkpoints/
COPY artifacts/ artifacts/

# Bake every pretrained encoder into the image's HF cache (offline at runtime).
RUN python training/prefetch_models.py

# Copy the built SPA from stage 1.
COPY --from=frontend /frontend/dist frontend/dist

# At runtime everything is local — no network calls to Hugging Face.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/health').status==200 else 1)"

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
