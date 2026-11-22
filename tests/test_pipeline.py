from types import SimpleNamespace

import numpy as np

from audio2text import asr, diarize, media, models, pipeline, vad
from audio2text.media import SR


def test_head_cut_last_wide_gap_in_window():
    chunks = [[0.0, 20.0], [20.5, 40.0], [42.0, 55.0], [57.0, 61.0], [65.0, 90.0]]
    assert pipeline.head_cut(chunks) == 56.0  # gap 40-42 too early to win, 61-65 mid 63 > 60, 20-20.5 too narrow


def test_head_cut_none():
    assert pipeline.head_cut([[0.0, 30.0], [30.3, 60.0], [60.2, 90.0]]) is None


def _fake(monkeypatch, dur, chunks, vad_s=0.0):
    """Fake clock: VAD over the whole file takes vad_s, ASR runs at 1 s per 1 s of audio."""
    calls = []
    logs = []
    clock = [0.0]
    monkeypatch.setattr(pipeline, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def vad_chunks(audio, vad_dir, **k):
        clock[0] += vad_s
        return chunks
    monkeypatch.setattr(media, "load_audio", lambda p: np.zeros(int(dur * SR), np.float32))
    monkeypatch.setattr(models, "ensure", lambda m, cache=None, log=print: m.name)
    monkeypatch.setattr(asr, "load_recognizer", lambda d, n: "rec")
    monkeypatch.setattr(diarize, "load_extractor", lambda d, n: "ext")
    monkeypatch.setattr(vad, "vad_chunks", vad_chunks)

    def transcribe(rec, audio, vad_dir, max_len=30.0, chunks=None):
        a, b = (chunks[0][0], chunks[-1][1]) if chunks else (0.0, len(audio) / SR)
        clock[0] += b - a
        calls.append((a, b) if chunks else round(b, 2))
        return [{"s": a, "e": b, "text": "a."}]

    monkeypatch.setattr(asr, "transcribe", transcribe)
    monkeypatch.setattr(diarize, "diarize",
                        lambda ext, audio, segs: ([{**s, "speaker": "SPEAKER_00"} for s in segs], {"SPEAKER_00": {}}))
    return calls, logs


def test_transcribe_file_split_no_double_work(monkeypatch, tmp_path):
    calls, logs = _fake(monkeypatch, 300.0, [[0.0, 50.0], [52.0, 300.0]])
    doc = pipeline.transcribe_file(tmp_path / "a.wav", log=logs.append)
    assert calls == [(0.0, 50.0), (52.0, 300.0)]  # chunks from the one VAD pass, no second VAD, no shift
    assert [(s["s"], s["e"]) for s in doc["segments"]] == [(0.0, 50.0), (52.0, 300.0)]
    assert doc["transcript"] == "локально, parakeet-tdt-0.6b-v3" and doc["audio_s"] == 300.0
    assert any(m.startswith("длительность 0:05:00, оценка ~") and m.endswith(" мин") for m in logs)


def test_transcribe_file_no_gap_measures_then_full(monkeypatch, tmp_path):
    calls, logs = _fake(monkeypatch, 300.0, [[0.0, 300.0]])
    doc = pipeline.transcribe_file(tmp_path / "a.wav", log=logs.append)
    assert calls == [60.0, (0.0, 300.0)]
    assert [(s["s"], s["e"]) for s in doc["segments"]] == [(0.0, 300.0)]


def test_transcribe_file_short_once(monkeypatch, tmp_path):
    calls, logs = _fake(monkeypatch, 45.0, [])
    pipeline.transcribe_file(tmp_path / "a.wav", log=logs.append)
    assert calls == [45.0]
    assert "длительность 0:00:45" in logs


def test_estimate_does_not_scale_full_file_vad(monkeypatch, tmp_path):
    """Head 51 s of ASR (cut at 51) takes 50 s -> 300 s of audio ~ 294 s; full-file VAD 60 s is added once, not x6."""
    calls, logs = _fake(monkeypatch, 300.0, [[0.0, 50.0], [52.0, 300.0]], vad_s=60.0)
    pipeline.transcribe_file(tmp_path / "a.wav", log=logs.append)
    assert "длительность 0:05:00, оценка ~6 мин" in logs  # (50 / 51 * 300 + 60) / 60 = 5.9; scaled VAD would be ~11
