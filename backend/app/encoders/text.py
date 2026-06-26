"""Text branch: Whisper-tiny transcription + MiniLM sentence embedding.

The text path stays fully frozen: we transcribe the audio with Whisper-tiny,
then embed the transcript with all-MiniLM-L6-v2 into a 384-d vector. Only the
small text classifier head (in the fusion model) is trained.
"""
from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from ..config import AUDIO_SAMPLE_RATE, TEXT_EMBED_MODEL_ID, WHISPER_MODEL_ID


class TextBranch:
    """Frozen transcribe-then-embed pipeline.

    Loaded once and reused. Whisper transcribes; MiniLM embeds. Both run in
    ``torch.no_grad`` and are never updated during training.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID)
        self.whisper = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_ID)
        self.whisper.to(device)
        self.whisper.eval()
        self.embedder = SentenceTransformer(TEXT_EMBED_MODEL_ID, device=device)

    @torch.no_grad()
    def transcribe(self, waveform: np.ndarray) -> str:
        """Transcribe a 16 kHz mono waveform to text (English)."""
        inputs = self.whisper_processor(
            waveform, sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="pt"
        )
        features = inputs.input_features.to(self.device)
        forced = self.whisper_processor.get_decoder_prompt_ids(language="en", task="transcribe")
        generated = self.whisper.generate(features, forced_decoder_ids=forced, max_new_tokens=128)
        text = self.whisper_processor.batch_decode(generated, skip_special_tokens=True)[0]
        return text.strip()

    @torch.no_grad()
    def embed(self, text: str) -> torch.Tensor:
        """Embed a transcript into a 384-d tensor (``[384]``)."""
        # Empty transcripts embed to a zero vector (no signal).
        if not text:
            return torch.zeros(384)
        vec = self.embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return torch.from_numpy(np.asarray(vec, dtype=np.float32))
