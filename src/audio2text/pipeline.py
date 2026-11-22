"""Конвейер для одного файла: аудио -> VAD + распознавание -> говорящие -> doc для markdown."""

from __future__ import annotations

import time
from pathlib import Path

from . import asr, diarize, media, models, vad
from .markdown import _hms

TRANSCRIPT = "локально, parakeet-tdt-0.6b-v3"
HEAD_S = 60.0  # скорость меряется на первых ≤ HEAD_S секундах
HEAD_MIN_S = 30.0  # ... но не меньше этого, иначе замер шумный
GAP_S = 0.8  # пауза между кусками VAD шире этого (порог склейки vad_chunks): там можно резать


def head_cut(chunks: list[list[float]]) -> float | None:
    """Точка разреза в тишине между кусками VAD, в [HEAD_MIN_S, HEAD_S]: середина последней паузы шире GAP_S.
    None — такой паузы нет."""
    cut = None
    for (_, b), (c, _) in zip(chunks, chunks[1:]):
        mid = (b + c) / 2
        if c - b > GAP_S and HEAD_MIN_S <= mid <= HEAD_S:
            cut = mid
    return cut


def transcribe_file(path: Path, log=print) -> dict:
    """Файл -> {"source", "transcript", "audio_s", "speakers", "segments"}. Аудио читается до загрузки моделей:
    не медиафайл падает сразу, без скачивания 0.5 ГБ."""
    audio = media.load_audio(path)
    dur = len(audio) / media.SR
    asr_dir, vad_dir, spk_dir = models.ensure_all([models.ASR, models.VAD, models.SPEAKER], log=log)
    n = asr.threads()
    rec = asr.load_recognizer(asr_dir, n)
    extractor = diarize.load_extractor(spk_dir, n)
    t0 = time.monotonic()
    if dur <= HEAD_S:
        log(f"длительность {_hms(dur)}")
        segs = asr.transcribe(rec, audio, vad_dir)
    else:
        chunks = vad.vad_chunks(audio, vad_dir)
        vad_took = time.monotonic() - t0  # VAD всего файла: прибавляется к оценке один раз, не масштабируется
        cut = head_cut(chunks)
        head_s = cut or HEAD_S
        t1 = time.monotonic()
        if cut:
            head = asr.transcribe(rec, audio, vad_dir, chunks=[c for c in chunks if c[1] <= cut])
        else:  # нет паузы для разреза: замер на первых HEAD_S с, потом всё заново
            asr.transcribe(rec, audio[:int(HEAD_S * media.SR)], vad_dir)
        took = time.monotonic() - t1
        log(f"длительность {_hms(dur)}, оценка ~{max(1, round((took / head_s * dur + vad_took) / 60))} мин")
        if cut:
            segs = head + asr.transcribe(rec, audio, vad_dir, chunks=[c for c in chunks if c[0] >= cut])
        else:
            segs = asr.transcribe(rec, audio, vad_dir, chunks=chunks)
    log(f"распознано за {_hms(time.monotonic() - t0)}, определяю говорящих")
    segs, speakers = diarize.diarize(extractor, audio, segs)
    log(f"готово за {_hms(time.monotonic() - t0)}")
    return {"source": str(path), "transcript": TRANSCRIPT, "audio_s": round(dur, 1), "speakers": speakers,
            "segments": segs}
