# TriSense — tri-modal emotion recognition with LoRA fine-tuning

TriSense looks at a short clip of one person speaking and predicts their **emotion**
using three modalities at once:

| Modality | What it sees | Encoder |
|----------|--------------|---------|
| 🎞️ **Video** | the face, from a few sampled frames | CLIP image tower (`openai/clip-vit-base-patch32`) |
| 🔊 **Audio** | the tone of voice, from the raw waveform | `facebook/wav2vec2-base` |
| 📝 **Text** | the spoken words | Whisper-tiny transcribes → `all-MiniLM-L6-v2` embeds |

The three embeddings are fused by a small MLP into one prediction over 8 emotions
(*neutral, calm, happy, sad, angry, fearful, disgust, surprised*). It is a portfolio
project that shows **backend, frontend, MLOps, and multimodal deep-learning** skills,
and it runs **fully in Docker with one command on a CPU laptop** — the repo ships a
trained checkpoint, so it works out of the box.

> **The core method is LoRA fine-tuning.** Instead of retraining the giant CLIP and
> wav2vec2 encoders, we freeze them and train tiny low-rank adapters injected into
> their attention layers, jointly with the classifier heads. This is cheap, fast, and
> produces a checkpoint of just a few MB. See [The method: LoRA](#the-method-lora-is-the-centerpiece).

---

## Quick start (the only command you need)

```bash
docker compose up --build
```

Then open:

- **http://localhost:8000** — the TriSense app (upload a clip, see the explained prediction)
- **http://localhost:5000** — the MLflow UI (training runs, metrics, confusion matrix)

No GPU, no API keys, no internet at run time. The pretrained weights and the trained
checkpoint are baked into the image; the app loads the committed checkpoint and **never
trains at run time**.

> RAVDESS clips are short scripted sentences. To try it, record a 1–5 s clip of a face
> speaking (mp4/mov/avi/mkv/webm, ≤ 50 MB) and upload it. You can also just look at the
> built-in **Results Gallery**, which shows held-out test clips with no upload needed.

---

## What you get in the UI

- **Fused prediction + confidence** for the uploaded clip.
- **Per-modality predictions** — what video, audio and text each think on their own.
- **Modality contribution chart** — a *leave-one-modality-out* bar chart: we drop each
  modality and measure how far the fused confidence falls. This is the explainability
  centerpiece and it is intuitive: a tall bar means that modality really mattered.
- **Transcript** from Whisper, and the **sampled frames** the model actually looked at.
- **Results Gallery** — a handful of committed held-out clips, each with true label,
  fused & per-modality predictions, and the contribution chart.
- **Model card** — held-out **accuracy**, **macro-F1**, **confusion matrix**, and a link
  to the MLflow runs.

---

## The method: LoRA is the centerpiece

LoRA (Low-Rank Adaptation) freezes a pretrained encoder and learns a small pair of
low-rank matrices added to chosen weight matrices. You train ~1% of the parameters,
get most of the benefit of fine-tuning, and the result is tiny.

In TriSense:

- **Frozen base encoders:** all of CLIP's and wav2vec2's pretrained weights.
- **LoRA adapters (trained):** injected into the attention projections
  (`q_proj`, `k_proj`, `v_proj`, `out_proj`) of **both** CLIP and wav2vec2.
- **Heads (trained):** a per-modality linear classifier for video, audio and text, plus
  the fusion MLP.
- **Text path is fully frozen:** Whisper transcribes and MiniLM embeds; only the small
  text head learns. (RAVDESS uses two scripted sentences, so the text branch mainly
  proves the pipeline rather than carrying strong emotional signal — see [Honesty](#honest-caveats).)

Everything is configurable: which encoders get LoRA, LoRA rank & alpha, learning rate,
batch size, epochs, mixed precision, and the `cuda`/`cpu` device flag.

**Two training modes:**

| | Phase one (this repo, CPU) | Phase two (your GPU) |
|---|---|---|
| Encoders | frozen (LoRA off) | **LoRA on CLIP + wav2vec2** |
| Trains | the 4 heads only | LoRA adapters + the 4 heads |
| Why | finishes in minutes so the app runs end-to-end | the real fine-tuning run |
| Command | `make train` | `make train-gpu` |

The committed checkpoint is the quick CPU one. The code path for full LoRA training is
implemented and tested; you run it on GPU in [Training](#training-phase-two-on-your-gpu).

---

## Architecture

```
        ┌─────────── frame sampling ──────────┐
clip ──►│ CLIP vision (+LoRA) → mean → 512-d   │─┐
        └─────────────────────────────────────┘ │   ┌── video head → 8
        ┌─────────── waveform ────────────────┐  ├──►│   audio head → 8
clip ──►│ wav2vec2 (+LoRA) → mean-pool → 768-d │──┤   │   text head  → 8
        └─────────────────────────────────────┘  │   └── fusion MLP(512+768+384) → 8  ← final
        ┌─ Whisper-tiny → text → MiniLM 384-d ─┐  │
clip ──►│              (frozen)                │──┘
        └─────────────────────────────────────┘
```

```
backend/app/
  encoders/      video.py (CLIP), audio.py (wav2vec2), text.py (Whisper+MiniLM)
  model/         fusion.py (TriSenseModel), lora.py (adapters), checkpoint.py (small ckpt I/O)
  features/      pipeline.py (clip → frames, audio, transcript, embeddings)
  service/       inference.py (prediction + leave-one-out explainability)
  api/           routes.py (analyze, health, model-card, gallery)
  main.py        FastAPI app; serves /api and the built SPA at /
training/        download_data.py, prefetch_models.py, train.py, evaluate.py, make_gallery.py
frontend/        React + Vite + TypeScript + Plotly single page
```

---

## API

| Method & path | Purpose |
|---|---|
| `POST /api/analyze` | Upload a clip → full explainable JSON (fused + per-modality predictions, leave-one-out contributions, transcript, frames). |
| `GET /api/health` | Liveness + whether the model is loaded. |
| `GET /api/model-card` | Held-out accuracy, macro-F1, confusion matrix. |
| `GET /api/gallery` | Pre-computed held-out results gallery. |

Interactive docs at `http://localhost:8000/docs`.

---

## Local development (without Docker)

Requires Python 3.11+ and Node 20+. [`uv`](https://github.com/astral-sh/uv) is used if present.

```bash
make install          # create .venv and install deps (CPU torch)
make prefetch         # download the pretrained encoders into the HF cache

# Frontend (for the dev server with hot reload):
cd frontend && npm install && npm run dev   # proxies /api to :8000

# Backend:
. .venv/bin/activate
uvicorn backend.app.main:app --reload        # http://localhost:8000

# Tests:
make test
```

---

## Reproducing the data + quick checkpoint

```bash
make data       # downloads a RAVDESS subset and builds data/subset/manifest.csv
make train      # caches features once, trains the heads, writes the checkpoint,
                # the model card, the confusion matrix and the results gallery,
                # logging everything to MLflow (./mlruns)
```

`make data` is configurable: `make data ACTORS=24 PER_CLASS=80` builds a bigger subset.
The **raw dataset is never committed** (only the manifest and the small checkpoint are).

### Dataset — RAVDESS

[RAVDESS](https://zenodo.org/records/1188976) (free, on Zenodo) — the speech,
audio-visual subset: 8 emotions, one actor per clip, face + voice. `download_data.py`
keeps only audio-visual speech files and builds a small **balanced** subset (40–80 clips
per emotion, configurable) with a stratified train/val/test split.

---

## Training (phase two, on your GPU)

The real LoRA fine-tuning runs **locally, outside Docker**, on your GPU
(reference hardware: **NVIDIA RTX 2080 Ti, 16 GB**).

```bash
# 1) Environment with the CUDA build of PyTorch (instead of the CPU build):
uv venv .venv --python 3.12 && . .venv/bin/activate
uv pip install -r training/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# 2) A larger RAVDESS subset (more actors / more clips per emotion):
python training/download_data.py --actors 24 --per-class 80

# 3) The real run: LoRA on both encoders, GPU, mixed precision:
make train-gpu
# == python training/train.py --device cuda --amp \
#       --lora-video --lora-audio --lora-rank 8 --lora-alpha 16 \
#       --epochs 8 --batch-size 24 --lr 2e-4 --run-name gpu-lora-both
```

**Recommended starting settings (2080 Ti, 16 GB)** — these are starting points, not gospel:

| Setting | Value |
|---|---|
| LoRA | on CLIP **and** wav2vec2 |
| LoRA rank / alpha | 8 / 16 |
| Mixed precision | on (`--amp`) |
| Batch size | 16–32 (start ~24) |
| Epochs | a few (start ~8) |
| Learning rate | ~2e-4 |

**What good looks like:**

- Fused **accuracy clearly above the 12.5% random baseline** for 8 classes.
- **Fused accuracy ≥ the best single modality** — fusion should help, not hurt.

**Troubleshooting:**

- *Out of memory* → lower `--batch-size` (e.g. 16, then 8); keep `--amp` on.
- *Underfitting / low accuracy* → train more epochs, raise `--lr`, or raise `--lora-rank`
  (e.g. 16) so the adapters have more capacity.

**After the run:**

```bash
mlflow ui            # or just keep the docker mlflow service up; inspect metrics
# happy with it? the new checkpoint is already at checkpoints/trisense.pt
python training/evaluate.py --device cuda   # refresh model card + confusion matrix
python training/make_gallery.py --device cuda
git add checkpoints/trisense.pt artifacts/ && git commit -m "Update checkpoint from GPU run"
```

The app then serves the new checkpoint automatically.

---

## MLOps

[MLflow](https://mlflow.org/) (open source) tracks every run: parameters (LoRA settings,
lr, batch size, epochs…), metrics (accuracy, macro-F1, loss curves, per-modality
accuracy), the confusion-matrix artifact, the model card, and the saved checkpoint.
Training is a single reproducible command (`make train` / `make train-gpu`) that logs to
`./mlruns`; the **MLflow UI runs as a Docker service** on port 5000 (it reads the
`./mlruns` you produce locally — runs appear there after you train).

---

## Honest caveats

- **Text branch:** RAVDESS only has two scripted sentences ("Kids are talking by the
  door" / "Dogs are sitting by the door"), so the spoken *words* barely correlate with
  emotion. The text path is real and end-to-end (Whisper → MiniLM → head), but it mainly
  **proves the tri-modal pipeline** rather than adding strong signal. The leave-one-out
  chart usually shows this honestly: text contributes little.
- **The committed checkpoint is the quick CPU one** — frozen encoders, heads only,
  trained on a tiny balanced subset so the app runs end-to-end in minutes. It clears the
  random baseline but is not the strong model; run `make train-gpu` for that.
- **One person per clip**, frontal-ish face, short clip — that is the RAVDESS setting.

---

## Tech stack

FastAPI · PyTorch · Hugging Face Transformers · PEFT (LoRA) · sentence-transformers ·
OpenCV · librosa · React · Vite · TypeScript · Plotly · MLflow · Docker. Everything is
free and open source.

## License

MIT — see [LICENSE](LICENSE).
