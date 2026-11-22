"""Распознавание речи: Parakeet int8 на sherpa-onnx (CPU) по кускам от VAD, batch 1."""

import os
from pathlib import Path

import numpy as np
import sherpa_onnx

from .media import SR
from .vad import vad_chunks


def threads() -> int:
    return os.cpu_count() or 1


def load_recognizer(model_dir: Path, threads: int) -> sherpa_onnx.OfflineRecognizer:
    p = Path(model_dir)
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(p / "encoder.int8.onnx"), decoder=str(p / "decoder.int8.onnx"),
        joiner=str(p / "joiner.int8.onnx"), tokens=str(p / "tokens.txt"), num_threads=threads,
        model_type="nemo_transducer", decoding_method="greedy_search")


def sentences(result, dur: float) -> list[tuple[float, float, str]]:
    """[(start, end, text)] предложений одного куска, граница после токена с . ? !; время от начала куска.
    Предложение кончается там, где начинается следующее (последнее: в конце куска)."""
    out, words, start = [], [], None
    toks, ts = list(result.tokens), list(result.timestamps)
    for i, (tok, t) in enumerate(zip(toks, ts)):
        if start is None:
            start = t
        words.append(tok)
        if tok.rstrip().endswith((".", "?", "!")) or i == len(toks) - 1:
            end = ts[i + 1] if i + 1 < len(ts) else dur
            text = "".join(words).replace("▁", " ").strip()
            if text:
                out.append((start, end, text))
            words, start = [], None
    return out


def transcribe(rec, audio: np.ndarray, vad_dir: Path, max_len: float = 30.0) -> list[dict]:
    """Сегменты {"s","e","text"} по всему аудио; время от его начала. За раз декодируется один кусок VAD."""
    segs = []
    for s, e in vad_chunks(audio, vad_dir, max_len=max_len):
        st = rec.create_stream()
        st.accept_waveform(SR, audio[int(s * SR):int(e * SR)])
        rec.decode_streams([st])
        segs += [{"s": round(s + a, 2), "e": round(s + b, 2), "text": t}
                 for a, b, t in sentences(st.result, e - s)]
    return segs
