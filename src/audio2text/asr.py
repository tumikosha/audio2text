"""Распознавание речи: transducer (по умолчанию Parakeet int8) на sherpa-onnx (CPU) по кускам от VAD, batch 1."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sherpa_onnx

from .media import SR
from .models import asr_files
from .vad import vad_chunks

PAD_S = 0.5  # тишина по краям куска в запасном пути
MIN_SPLIT_S = 2.0  # пустой кусок короче этого не повторяется; часть короче 2 * MIN_SPLIT_S не режется
WHOLE_RETRY_S = 8.0  # обе половины пусты: кусок короче этого ещё раз декодируется целиком с отступом
HOLE_S = 3.0  # дыра без токенов, где речи (по энергии) не меньше этого, декодируется отдельно
HOLE_WPS = 1.5  # текст дыры берётся, если слов в нём не меньше этого на секунду её речи (иначе музыка, шум)
PIECE_S = 10.0  # дыра декодируется частями не длиннее этого
FRAME_S = 0.05  # окно оценки энергии
HOLE_DB = 10.0  # речь в дыре: окна не тише уровня токенов первого прохода больше чем на столько (тише — фон)
MAX_HOLES = 3  # дыр на кусок не больше этого (с наибольшей речью): время на шуме ограничено


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


def _result(rec, piece: np.ndarray, pad: float = 0.0) -> SimpleNamespace:
    """Токены куска; pad — тишина с обеих сторон, времена токенов сдвинуты обратно (от начала куска)."""
    if pad:
        z = np.zeros(int(pad * SR), np.float32)
        piece = np.concatenate([z, piece, z])
    st = rec.create_stream()
    st.accept_waveform(SR, piece)
    rec.decode_streams([st])
    return SimpleNamespace(tokens=list(st.result.tokens), timestamps=[t - pad for t in st.result.timestamps])


def _sentences(res, dur: float, pad: float = 0.0) -> list[tuple[float, float, str]]:
    """sentences() результата _result; времена зажаты в кусок."""
    out = []
    for a, b, t in sentences(res, dur + pad):
        a, b = min(max(0.0, a), dur), min(max(0.0, b), dur)
        if b > a or not pad:  # без отступа — как sentences(); с отступом схлопнутое в отступе отбрасывается
            out.append((a, b, t))
    return out


def _decode(rec, piece: np.ndarray, pad: float = 0.0) -> list[tuple[float, float, str]]:
    """sentences() одного куска; pad — тишина с обеих сторон, времена сдвинуты обратно и зажаты в кусок."""
    return _sentences(_result(rec, piece, pad), len(piece) / SR, pad)


def _quietest(piece: np.ndarray, lo: int, hi: int) -> int:
    """Индекс середины самого тихого окна 0.1 с между отсчётами lo и hi; окно не помещается — середина."""
    f = int(0.1 * SR)
    n = (hi - lo) // f
    if n < 1:
        return (lo + hi) // 2
    energy = (piece[lo:lo + n * f].reshape(n, f) ** 2).mean(1)
    return lo + int(energy.argmin()) * f + f // 2


def _quiet_cut(piece: np.ndarray) -> int:
    """Индекс разреза: середина самого тихого окна 0.1 с в средней половине куска."""
    return _quietest(piece, len(piece) // 4, 3 * len(piece) // 4)


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


def _energy(piece: np.ndarray) -> np.ndarray:
    f = int(FRAME_S * SR)
    n = len(piece) // f
    return np.sqrt((piece[:n * f].reshape(n, f) ** 2).mean(1))


def _speech(piece: np.ndarray, ts: list | None = None) -> np.ndarray:
    """Окна FRAME_S с речью по энергии: громче десятой доли 90-го перцентиля куска и не тише -60 дБ. ts — времена
    токенов первого прохода: тогда ещё и не тише медианы окон с токенами больше чем на HOLE_DB (фон, чужая речь)."""
    e = _energy(piece)
    if not len(e):
        return e > 0
    mask = e > max(0.1 * np.percentile(e, 90), 1e-3)
    frames = sorted({j for t in ts or [] for j in range(int(t / FRAME_S), min(len(e), int((t + 0.2) / FRAME_S) + 1))})
    if frames:
        mask &= e >= np.median(e[frames]) * 10 ** (-HOLE_DB / 20)
    return mask


def _speech_s(mask: np.ndarray, a: float, b: float) -> float:
    return float(mask[max(0, int(a / FRAME_S)):max(0, int(b / FRAME_S))].sum()) * FRAME_S


def _nwords(toks) -> int:
    return len("".join(toks).replace("▁", " ").split())


def _hole(rec, piece: np.ndarray, mask: np.ndarray, ca: float, cb: float) -> tuple[list, list]:
    """Токены дыры [ca, cb) и их времена от начала куска: слева направо частями не длиннее PIECE_S (разрез в самом
    тихом месте второй половины части), каждая с отступами PAD_S; токены в отступе отбрасываются. Часть без речи или
    где слов меньше HOLE_WPS на секунду речи (музыка, шум), и всё после неё не берутся: на музыке это один вызов."""
    toks, ts, pos = [], [], ca
    while cb - pos > 1e-3:
        end = cb if cb - pos <= PIECE_S else _quietest(
            piece, int((pos + PIECE_S / 2) * SR), int((pos + PIECE_S) * SR)) / SR
        sp = _speech_s(mask, pos, end)
        if sp == 0:
            break
        part = piece[int(pos * SR):int(end * SR)]
        res = _result(rec, part, PAD_S)
        keep = [(k, pos + max(t, 0.0)) for k, t in zip(res.tokens, res.timestamps) if -0.2 < t < end - pos]
        if _nwords(k for k, _ in keep) < HOLE_WPS * sp:
            break
        toks += [k for k, _ in keep]
        ts += [t for _, t in keep]
        pos = end
    return toks, ts


def _seam(hk: list, ht: list, nxt: str | None) -> tuple[list, list]:
    """Вставка перед токеном первого прохода со строчной буквы не кончает предложение: финальная . ? ! снимается."""
    if not hk or nxt is None or not nxt.replace("▁", "").strip()[:1].islower():
        return hk, ht
    hk, ht = list(hk), list(ht)
    while hk and not hk[-1].replace("▁", "").strip(".?! "):
        hk.pop(), ht.pop()
    if hk:
        hk[-1] = hk[-1].rstrip(".?!")
    return hk, ht


def _fill_holes(rec, piece: np.ndarray, res) -> SimpleNamespace:
    """Parakeet (greedy) иногда обрывает текст посреди куска: до первого токена, после последнего или между токенами
    остаётся речь без слов. Дыра, где речи от HOLE_S, декодируется отдельно (_hole) и её токены вставляются; где
    слов меньше HOLE_WPS на секунду речи — музыка или шум, там текст не меняется. Речь — окна не тише токенов
    первого прохода больше чем на HOLE_DB; дыр на кусок не больше MAX_HOLES. Края дыры — в тишине рядом
    с соседними токенами: время токена — его начало, поэтому левый край ищется после него, правый перед ним."""
    dur = len(piece) / SR
    toks, ts = list(res.tokens), list(res.timestamps)
    mask = _speech(piece, ts)
    edges = [-0.5] + ts + [dur + 0.3]
    holes = [(a, b) for a, b in zip(edges, edges[1:]) if _speech_s(mask, a + 0.5, b - 0.3) >= HOLE_S]
    holes = sorted(sorted(holes, key=lambda h: -_speech_s(mask, h[0] + 0.5, h[1] - 0.3))[:MAX_HOLES])
    for a, b in holes:
        ca = _quietest(piece, int((a + 0.1) * SR), int(min(a + 0.8, b) * SR)) / SR if a >= 0 else 0.0
        cb = _quietest(piece, int(max(b - 0.4, ca) * SR), int(b * SR)) / SR if b <= dur else dur
        if cb - ca < MIN_SPLIT_S:
            continue
        i = sum(1 for t in ts if t < ca)
        hk, ht = _seam(*_hole(rec, piece, mask, ca, cb), toks[i] if i < len(toks) else None)
        toks[i:i] = hk
        ts[i:i] = ht
    return SimpleNamespace(tokens=toks, timestamps=ts)


def decode_chunk(rec, piece: np.ndarray) -> list[tuple[float, float, str]]:
    """Предложения куска VAD, время от его начала. Parakeet (greedy) иногда возвращает пусто на целом куске речи,
    чаще когда кусок начат посреди слова (запись с середины фразы, разрез VAD по max_speech_duration). Тогда — запасной
    путь: кусок короче 2 * MIN_SPLIT_S декодируется с тишиной PAD_S по краям; длиннее — режется пополам (_halves) и
    декодируется целиком с отступами, берётся вариант, где больше слов; обе половины пусты — целиком с отступами,
    только если кусок короче WHOLE_RETRY_S (длинный пустой кусок — обычно музыка, шум). Отступ не ставится всегда: на обычных кусках
    он терял фразы («Me at the zoo»: 3 предложения из 4) и замедлял распознавание. Непустой, но обрезанный результат
    дополняется (_fill_holes)."""
    dur = len(piece) / SR
    res = _result(rec, piece)
    if res.tokens:
        return _sentences(_fill_holes(rec, piece, res), dur)
    if dur < MIN_SPLIT_S:
        return []
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
