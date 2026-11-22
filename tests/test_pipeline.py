import numpy as np

from audio2text import asr, diarize, media, models, pipeline, vad
from audio2text.media import SR


def test_head_cut_last_wide_gap_in_window():
    chunks = [[0.0, 20.0], [20.5, 40.0], [42.0, 55.0], [57.0, 61.0], [65.0, 90.0]]
    assert pipeline.head_cut(chunks) == 56.0  # gap 40-42 too early to win, 61-65 mid 63 > 60, 20-20.5 too narrow


def test_head_cut_none():
    assert pipeline.head_cut([[0.0, 30.0], [30.3, 60.0], [60.2, 90.0]]) is None


def _fake(monkeypatch, dur, chunks):
    calls = []
    logs = []
    monkeypatch.setattr(media, "load_audio", lambda p: np.zeros(int(dur * SR), np.float32))
    monkeypatch.setattr(models, "ensure", lambda m, cache=None, log=print: m.name)
    monkeypatch.setattr(asr, "load_recognizer", lambda d, n: "rec")
    monkeypatch.setattr(diarize, "load_extractor", lambda d, n: "ext")
    monkeypatch.setattr(vad, "vad_chunks", lambda audio, vad_dir, **k: chunks)

    def transcribe(rec, audio, vad_dir, max_len=30.0):
        n = len(audio) / SR
        calls.append(round(n, 2))
        return [{"s": 0.0, "e": round(n, 2), "text": "a."}]

    monkeypatch.setattr(asr, "transcribe", transcribe)
    monkeypatch.setattr(diarize, "diarize",
                        lambda ext, audio, segs: ([{**s, "speaker": "SPEAKER_00"} for s in segs], {"SPEAKER_00": {}}))
    return calls, logs


def test_transcribe_file_split_no_double_work(monkeypatch, tmp_path):
    calls, logs = _fake(monkeypatch, 300.0, [[0.0, 50.0], [52.0, 300.0]])
    doc = pipeline.transcribe_file(tmp_path / "a.wav", log=logs.append)
    assert calls == [51.0, 249.0]
    assert [(s["s"], s["e"]) for s in doc["segments"]] == [(0.0, 51.0), (51.0, 300.0)]
    assert doc["transcript"] == "локально, parakeet-tdt-0.6b-v3" and doc["audio_s"] == 300.0
    assert any(m.startswith("длительность 0:05:00, оценка ~") and m.endswith(" мин") for m in logs)


def test_transcribe_file_no_gap_measures_then_full(monkeypatch, tmp_path):
    calls, logs = _fake(monkeypatch, 300.0, [[0.0, 300.0]])
    doc = pipeline.transcribe_file(tmp_path / "a.wav", log=logs.append)
    assert calls == [60.0, 300.0]
    assert [(s["s"], s["e"]) for s in doc["segments"]] == [(0.0, 300.0)]


def test_transcribe_file_short_once(monkeypatch, tmp_path):
    calls, logs = _fake(monkeypatch, 45.0, [])
    pipeline.transcribe_file(tmp_path / "a.wav", log=logs.append)
    assert calls == [45.0]
    assert "длительность 0:00:45" in logs
