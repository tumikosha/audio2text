"""Конвейер для одного файла: аудио -> VAD + распознавание -> говорящие -> doc для markdown."""

from __future__ import annotations

import time
from pathlib import Path

from tqdm import tqdm

from . import asr, diarize, media, models, vad
from .progress import spinner
from .markdown import _hms

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


def _bar(dur: float, done: float = 0.0) -> tqdm:
    """Полоса распознавания в stderr, в секундах аудио; не в терминале (CI, перенаправление) выключена."""
    return tqdm(total=round(dur, 1), initial=round(done, 1), desc="transcribing", disable=None, leave=False,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f} s [{elapsed}<{remaining}]")


def transcribe_file(path: Path, log=print, model: models.AsrChoice | None = None) -> dict:
    """Файл -> {"source", "transcript", "audio_s", "speakers", "segments"}. Аудио читается до загрузки моделей:
    не медиафайл падает сразу, без скачивания 0.5 ГБ. model — из models.resolve_asr; None — по умолчанию."""
    model = model or models.resolve_asr(None)
    with spinner("reading audio"):
        audio = media.load_audio(path)
    dur = len(audio) / media.SR
    if model.folder:
        vad_dir, spk_dir = models.ensure_all([models.VAD, models.SPEAKER], log=log)
        asr_dir = model.folder
    else:
        asr_dir, vad_dir, spk_dir = models.ensure_all([model.model, models.VAD, models.SPEAKER], log=log)
    n = asr.threads()
    with spinner("loading models"):
        rec = asr.load_recognizer(asr_dir, n)
        extractor = diarize.load_extractor(spk_dir, n)
    t0 = time.monotonic()
    if dur <= HEAD_S:
        log(f"duration {_hms(dur)}")
        with _bar(dur) as bar:
            segs = asr.transcribe(rec, audio, vad_dir, bar=bar)
            bar.update(bar.total - bar.n)  # хвост после последнего куска VAD
    else:
        with spinner("finding speech"):
            chunks = vad.vad_chunks(audio, vad_dir)
        vad_took = time.monotonic() - t0  # VAD всего файла: прибавляется к оценке один раз, не масштабируется
        cut = head_cut(chunks)
        head_s = cut or HEAD_S
        t1 = time.monotonic()
        with spinner("estimating time"):
            if cut:
                head = asr.transcribe(rec, audio, vad_dir, chunks=[c for c in chunks if c[1] <= cut])
            else:  # нет паузы для разреза: замер на первых HEAD_S с, потом всё заново
                asr.transcribe(rec, audio[:int(HEAD_S * media.SR)], vad_dir)
        took = time.monotonic() - t1
        log(f"duration {_hms(dur)}, estimate ~{max(1, round((took / head_s * dur + vad_took) / 60))} min")
        with _bar(dur, cut or 0.0) as bar:  # полоса после строки оценки: вывод не перемешивается
            if cut:
                segs = head + asr.transcribe(rec, audio, vad_dir, chunks=[c for c in chunks if c[0] >= cut], bar=bar)
            else:
                segs = asr.transcribe(rec, audio, vad_dir, chunks=chunks, bar=bar)
            bar.update(bar.total - bar.n)  # хвост после последнего куска VAD
    log(f"transcribed in {_hms(time.monotonic() - t0)}, finding speakers")
    segs, speakers = diarize.diarize(extractor, audio, segs)
    log(f"done in {_hms(time.monotonic() - t0)}")
    return {"source": str(path), "transcript": f"локально, {model.label}", "audio_s": round(dur, 1), "speakers": speakers,
            "segments": segs}
