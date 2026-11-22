"""Речевые участки по silero-vad через sherpa-onnx (без torch)."""

from pathlib import Path

import numpy as np
import sherpa_onnx

from .media import SR

PAD = 0.2  # как speech_pad_ms=200 у silero-vad


def _detector(vad_dir: Path, threshold: float, min_silence: float, max_len: float):
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(Path(vad_dir) / "silero_vad.onnx")
    cfg.silero_vad.threshold = threshold
    cfg.silero_vad.min_silence_duration = min_silence
    cfg.silero_vad.min_speech_duration = 0.25
    cfg.silero_vad.max_speech_duration = max_len
    cfg.sample_rate = SR
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)


def vad_chunks(audio: np.ndarray, vad_dir: Path, max_gap=0.8, max_len=30.0, threshold=0.5,
               min_silence=0.5) -> list[list[float]]:
    """Речевые участки [start, end] в секундах: соседние склеиваются, пока пауза ≤ max_gap
    и участок не длиннее max_len."""
    det = _detector(vad_dir, threshold, min_silence, max_len)
    window = 512
    total = len(audio) / SR
    spans: list[tuple[float, float]] = []

    def drain():
        while not det.empty():
            seg = det.front
            spans.append((seg.start / SR, (seg.start + len(seg.samples)) / SR))
            det.pop()

    for i in range(0, len(audio), window):
        det.accept_waveform(audio[i:i + window])
        drain()
    det.flush()
    drain()

    chunks: list[list[float]] = []
    for a, b in spans:
        a, b = max(0.0, a - PAD), min(total, b + PAD)
        if chunks and a - chunks[-1][1] <= max_gap and b - chunks[-1][0] <= max_len:
            chunks[-1][1] = b
        else:
            chunks.append([a, b])
    return chunks
