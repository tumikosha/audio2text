"""Распознавание речи: transducer (по умолчанию Parakeet int8) на sherpa-onnx (CPU) по кускам от VAD, batch 1."""

import os
from pathlib import Path

import numpy as np
import sherpa_onnx

from .media import SR
from .models import asr_files
from .vad import vad_chunks

PAD_S = 0.5  # тишина по краям куска в запасном пути
MIN_SPLIT_S = 2.0  # пустой кусок короче этого не повторяется; часть короче 2 * MIN_SPLIT_S не режется
WHOLE_RETRY_S = 8.0  # обе половины пусты: кусок короче этого ещё раз декодируется целиком с отступом


def threads() -> int:
    return os.cpu_count() or 1


def load_recognizer(model_dir: Path, threads: int) -> sherpa_onnx.OfflineRecognizer:
    enc, dec, join, tok = asr_files(Path(model_dir))
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(enc), decoder=str(dec), joiner=str(join), tokens=str(tok), num_threads=threads,
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


def _decode(rec, piece: np.ndarray, pad: float = 0.0) -> list[tuple[float, float, str]]:
    """sentences() одного куска; pad — тишина с обеих сторон, времена сдвинуты обратно и зажаты в кусок."""
    dur = len(piece) / SR
    if pad:
        z = np.zeros(int(pad * SR), np.float32)
        piece = np.concatenate([z, piece, z])
    st = rec.create_stream()
    st.accept_waveform(SR, piece)
    rec.decode_streams([st])
    out = []
    for a, b, t in sentences(st.result, len(piece) / SR):
        a, b = min(max(0.0, a - pad), dur), min(max(0.0, b - pad), dur)
        if b > a or not pad:  # без отступа — как sentences(); с отступом схлопнутое в отступе отбрасывается
            out.append((a, b, t))
    return out


def _quiet_cut(piece: np.ndarray) -> int:
    """Индекс разреза: середина самого тихого окна 0.1 с в средней половине куска."""
    f = int(0.1 * SR)
    lo = len(piece) // 4
    n = (3 * len(piece) // 4 - lo) // f
    energy = (piece[lo:lo + n * f].reshape(n, f) ** 2).mean(1)
    return lo + int(energy.argmin()) * f + f // 2


def _halves(rec, piece: np.ndarray, off: float = 0.0) -> list[tuple[float, float, str]]:
    """Кусок режется в тишине на две части, каждая декодируется с отступами. Пустая часть режется дальше, только если
    в соседней есть текст: обе пусты — это не речь (музыка, шум), дальше не тратим время."""
    h = _quiet_cut(piece)
    parts = [(piece[:h], off), (piece[h:], off + h / SR)]
    res = [_decode(rec, p, PAD_S) for p, _ in parts]
    if not any(res):
        return []
    out = []
    for (p, o), r in zip(parts, res):
        if r:
            out += [(o + a, o + b, t) for a, b, t in r]
        elif len(p) / SR >= 2 * MIN_SPLIT_S:
            out += _halves(rec, p, o)
    return out


def _words(r) -> int:
    return sum(len(t.split()) for _, _, t in r)


def decode_chunk(rec, piece: np.ndarray) -> list[tuple[float, float, str]]:
    """Предложения куска VAD, время от его начала. Parakeet (greedy) иногда возвращает пусто на целом куске речи,
    чаще когда кусок начат посреди слова (запись с середины фразы, разрез VAD по max_speech_duration). Тогда — запасной
    путь: кусок короче 2 * MIN_SPLIT_S декодируется с тишиной PAD_S по краям; длиннее — режется пополам (_halves) и
    декодируется целиком с отступами, берётся вариант, где больше слов; обе половины пусты — целиком с отступами,
    только если кусок короче WHOLE_RETRY_S (длинный пустой кусок — обычно музыка, шум). Отступ не ставится всегда: на обычных кусках
    он терял фразы («Me at the zoo»: 3 предложения из 4) и замедлял распознавание."""
    r = _decode(rec, piece)
    dur = len(piece) / SR
    if r or dur < MIN_SPLIT_S:
        return r
    if dur < 2 * MIN_SPLIT_S:
        return _decode(rec, piece, PAD_S)
    halves = _halves(rec, piece)
    if not halves:
        return _decode(rec, piece, PAD_S) if dur < WHOLE_RETRY_S else []
    whole = _decode(rec, piece, PAD_S)
    return halves if _words(halves) > _words(whole) else whole


def transcribe(rec, audio: np.ndarray, vad_dir: Path, max_len: float = 30.0,
               chunks: list[list[float]] | None = None, bar=None) -> list[dict]:
    """Сегменты {"s","e","text"} по всему аудио; время от его начала. За раз декодируется один кусок VAD.
    chunks — готовые куски VAD [start, end] (секунды от начала audio); None — VAD запускается здесь.
    bar — tqdm в секундах аудио: после каждого куска доводится до его конца."""
    segs = []
    for s, e in vad_chunks(audio, vad_dir, max_len=max_len) if chunks is None else chunks:
        segs += [{"s": round(s + a, 2), "e": round(s + b, 2), "text": t}
                 for a, b, t in decode_chunk(rec, audio[int(s * SR):int(e * SR)])]
        if bar is not None and e > bar.n:
            bar.update(e - bar.n)
    return segs
