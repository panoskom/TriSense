"""Feature extraction: video file -> frames, audio, transcript, embeddings.

Shared by training and inference so a clip is always turned into model inputs
the same way. Video frames are decoded with OpenCV; audio is extracted with
the pip-packaged ffmpeg binary (no system ffmpeg required).
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import soundfile as sf
import torch
from PIL import Image

from ..config import AUDIO_SAMPLE_RATE, NUM_VIDEO_FRAMES
from ..encoders.audio import load_wav2vec2_extractor, preprocess_audio
from ..encoders.text import TextBranch
from ..encoders.video import load_clip_processor, preprocess_frames

_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class FeatureBundle:
    """All model inputs and human-facing artefacts for a single clip."""

    frames: list[Image.Image]          # sampled PIL frames (for the UI)
    pixel_values: torch.Tensor         # [F, 3, H, W]
    audio_values: torch.Tensor         # [T]
    audio_mask: torch.Tensor           # [T]
    waveform: np.ndarray               # raw 16 kHz mono
    transcript: str
    text_emb: torch.Tensor             # [384]


def sample_frames(video_path: Path, num_frames: int = NUM_VIDEO_FRAMES) -> list[Image.Image]:
    """Evenly sample ``num_frames`` RGB frames across the clip."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Fallback: read sequentially.
        frames_all = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_all.append(frame)
        cap.release()
        if not frames_all:
            raise ValueError("Could not decode any video frames from the clip.")
        idxs = np.linspace(0, len(frames_all) - 1, num=num_frames).astype(int)
        return [Image.fromarray(cv2.cvtColor(frames_all[i], cv2.COLOR_BGR2RGB)) for i in idxs]

    idxs = np.linspace(0, total - 1, num=num_frames).astype(int)
    frames: list[Image.Image] = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise ValueError("Could not decode any video frames from the clip.")
    # Pad by repeating the last frame if some reads failed.
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames


def extract_audio(video_path: Path, sample_rate: int = AUDIO_SAMPLE_RATE) -> np.ndarray:
    """Extract a mono waveform at ``sample_rate`` Hz using bundled ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        cmd = [
            _FFMPEG, "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-f", "wav", wav_path,
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to extract audio: {proc.stderr.decode()[-400:]}")
        waveform, _ = sf.read(wav_path, dtype="float32")
    finally:
        Path(wav_path).unlink(missing_ok=True)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if waveform.size == 0:
        waveform = np.zeros(sample_rate, dtype=np.float32)
    return waveform.astype(np.float32)


class FeatureExtractor:
    """Stateful extractor holding the processors and the frozen text branch."""

    def __init__(self, device: str = "cpu", num_frames: int = NUM_VIDEO_FRAMES) -> None:
        self.device = device
        self.num_frames = num_frames
        self.clip_processor = load_clip_processor()
        self.audio_extractor = load_wav2vec2_extractor()
        self.text_branch = TextBranch(device=device)

    def extract(self, video_path: str | Path) -> FeatureBundle:
        """Turn a clip on disk into a :class:`FeatureBundle`."""
        path = Path(video_path)
        frames = sample_frames(path, self.num_frames)
        pixel_values = preprocess_frames(self.clip_processor, frames)  # [F, 3, H, W]

        waveform = extract_audio(path)
        audio_values, audio_mask = preprocess_audio(self.audio_extractor, waveform)

        transcript = self.text_branch.transcribe(waveform)
        text_emb = self.text_branch.embed(transcript)

        return FeatureBundle(
            frames=frames,
            pixel_values=pixel_values,
            audio_values=audio_values.squeeze(0),
            audio_mask=audio_mask.squeeze(0),
            waveform=waveform,
            transcript=transcript,
            text_emb=text_emb,
        )
