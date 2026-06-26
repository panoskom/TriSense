"""Pydantic response/request models for the TriSense API.

These types are the contract the React frontend codes against, so every field
returned by the inference service is declared here with a docstring.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ModalityPrediction(BaseModel):
    """A single modality's standalone classification of the clip."""

    emotion: str = Field(..., description="Top emotion predicted by this modality alone.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax probability of the top emotion.")
    probabilities: dict[str, float] = Field(
        ..., description="Full probability distribution over all emotions for this modality."
    )


class AnalyzeResponse(BaseModel):
    """The full explainable result returned for an analysed clip."""

    predicted_emotion: str = Field(..., description="Fused (final) emotion prediction.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Fused confidence for the top emotion.")
    probabilities: dict[str, float] = Field(..., description="Fused probability distribution over all emotions.")

    modality_predictions: dict[str, ModalityPrediction] = Field(
        ..., description="Per-modality standalone predictions, keyed by 'video', 'audio', 'text'."
    )
    contributions: dict[str, float] = Field(
        ...,
        description=(
            "Leave-one-modality-out contribution per modality: the drop in fused "
            "confidence (for the predicted class) when that modality is removed. "
            "Higher means the modality mattered more."
        ),
    )

    transcript: str = Field(..., description="Whisper-tiny transcript of the spoken audio.")
    frames: list[str] = Field(
        ..., description="Base64 data-URI JPEGs of the frames sampled from the video."
    )
    inference_ms: float = Field(..., description="Wall-clock inference time in milliseconds.")


class ConfusionMatrix(BaseModel):
    """A labelled confusion matrix."""

    labels: list[str] = Field(..., description="Class labels, row/column order.")
    matrix: list[list[int]] = Field(..., description="Counts; matrix[i][j] = true i predicted j.")


class ModelCard(BaseModel):
    """Summary of the committed checkpoint's held-out performance."""

    checkpoint: str = Field(..., description="Checkpoint file name being served.")
    trained_on: str = Field(..., description="Human description of the training data/run.")
    device: str = Field(..., description="Device the API is running inference on.")
    num_test_clips: int = Field(..., description="Number of held-out clips evaluated.")
    test_accuracy: float = Field(..., description="Overall accuracy on the held-out test set.")
    macro_f1: float = Field(..., description="Macro-averaged F1 over the 8 emotions.")
    per_modality_accuracy: dict[str, float] = Field(
        default_factory=dict, description="Accuracy of each standalone modality head."
    )
    confusion_matrix: ConfusionMatrix
    confusion_matrix_image: str | None = Field(
        None, description="Relative URL of the rendered confusion-matrix PNG, if available."
    )
    random_baseline: float = Field(0.125, description="Random-guess accuracy for 8 balanced classes.")
    note: str = Field("", description="Caveats (e.g. quick CPU checkpoint vs. full GPU run).")


class GalleryItem(BaseModel):
    """One pre-computed held-out example shown in the results gallery."""

    clip_id: str
    true_emotion: str
    predicted_emotion: str
    confidence: float
    correct: bool
    modality_predictions: dict[str, ModalityPrediction]
    contributions: dict[str, float]
    transcript: str
    thumbnail: str | None = Field(None, description="Relative URL of a representative frame.")


class HealthResponse(BaseModel):
    """Liveness/readiness payload."""

    status: str
    model_loaded: bool
    device: str
    checkpoint: str
