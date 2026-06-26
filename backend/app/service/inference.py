"""Inference service: clip in -> explainable prediction out.

Loads the committed checkpoint once and serves analyses. The explainability
centrepiece is leave-one-modality-out (LOMO): drop each modality's embedding
and measure how much the fused confidence for the predicted class falls.
"""
from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from ..config import EMOTIONS, Settings
from ..features.pipeline import FeatureBundle, FeatureExtractor
from ..model.checkpoint import load_checkpoint
from ..model.fusion import ModelOutputs
from ..schemas import AnalyzeResponse, ModalityPrediction


def _probs_dict(logits: torch.Tensor) -> dict[str, float]:
    """Softmax a ``[num_classes]`` logit vector into an emotion->prob dict."""
    probs = F.softmax(logits, dim=-1)
    return {emotion: float(probs[i]) for i, emotion in enumerate(EMOTIONS)}


def _modality_prediction(logits: torch.Tensor) -> ModalityPrediction:
    probs = _probs_dict(logits)
    top = max(probs.items(), key=lambda kv: kv[1])[0]
    return ModalityPrediction(emotion=top, confidence=probs[top], probabilities=probs)


def _frame_to_data_uri(frame: Image.Image, size: int = 224) -> str:
    """Encode a PIL frame as a base64 JPEG data URI for the UI."""
    img = frame.convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class InferenceService:
    """Holds the model + feature extractor and produces explainable results."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = settings.device
        self.extractor = FeatureExtractor(device=self.device)
        self.model, self.meta = load_checkpoint(settings.checkpoint_path, device=self.device)

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _run_model(self, bundle: FeatureBundle) -> ModelOutputs:
        pixel = bundle.pixel_values.unsqueeze(0).to(self.device)
        audio = bundle.audio_values.unsqueeze(0).to(self.device)
        mask = bundle.audio_mask.unsqueeze(0).to(self.device)
        text = bundle.text_emb.unsqueeze(0).to(self.device)
        return self.model(pixel, audio, mask, text)

    @torch.no_grad()
    def _contributions(self, out: ModelOutputs, pred_idx: int, full_conf: float) -> dict[str, float]:
        """Leave-one-modality-out: drop each embedding, measure confidence fall."""
        zero_v = torch.zeros_like(out.video_emb)
        zero_a = torch.zeros_like(out.audio_emb)
        zero_t = torch.zeros_like(out.text_emb)

        drops = {
            "video": self.model.fuse_logits(zero_v, out.audio_emb, out.text_emb),
            "audio": self.model.fuse_logits(out.video_emb, zero_a, out.text_emb),
            "text": self.model.fuse_logits(out.video_emb, out.audio_emb, zero_t),
        }
        contributions: dict[str, float] = {}
        for name, logits in drops.items():
            dropped_conf = float(F.softmax(logits, dim=-1)[0, pred_idx])
            contributions[name] = full_conf - dropped_conf
        return contributions

    def analyze(self, video_path: str | Path) -> AnalyzeResponse:
        """Full explainable analysis of one clip."""
        start = time.perf_counter()
        bundle = self.extractor.extract(video_path)
        out = self._run_model(bundle)

        fused_probs = F.softmax(out.fused_logits, dim=-1)[0]
        pred_idx = int(torch.argmax(fused_probs))
        pred_emotion = EMOTIONS[pred_idx]
        confidence = float(fused_probs[pred_idx])

        contributions = self._contributions(out, pred_idx, confidence)

        modality_predictions = {
            "video": _modality_prediction(out.video_logits[0]),
            "audio": _modality_prediction(out.audio_logits[0]),
            "text": _modality_prediction(out.text_logits[0]),
        }

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return AnalyzeResponse(
            predicted_emotion=pred_emotion,
            confidence=confidence,
            probabilities={e: float(fused_probs[i]) for i, e in enumerate(EMOTIONS)},
            modality_predictions=modality_predictions,
            contributions=contributions,
            transcript=bundle.transcript,
            frames=[_frame_to_data_uri(f) for f in bundle.frames],
            inference_ms=round(elapsed_ms, 1),
        )
