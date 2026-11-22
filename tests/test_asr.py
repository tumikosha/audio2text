from types import SimpleNamespace

import numpy as np
import pytest

from audio2text import asr, models, vad
from audio2text.media import SR

def test_sentences_split_on_punctuation():
    res = SimpleNamespace(tokens=["▁При", "вет", ".", "▁Как", "?"], timestamps=[0, .2, .4, 1.0, 1.3])
    assert asr.sentences(res, 2.0) == [(0, 1.0, "Привет."), (1.0, 2.0, "Как?")]


def test_threads():
    assert asr.threads() >= 1


class FakeStream:
    result = None

    def accept_waveform(self, sr, samples):
        self.samples = samples


class FakeRec:
    """The 'result' is a single sentence over the whole chunk."""

    def create_stream(self):
        return FakeStream()

    def decode_streams(self, streams):
        for st in streams:
            n = len(st.samples) / SR
            st.result = SimpleNamespace(tokens=["▁hi", "."], timestamps=[0.0, n / 2])


def test_transcribe_offset(monkeypatch, tmp_path):
    monkeypatch.setattr(asr, "vad_chunks", lambda audio, vad_dir, max_len=30.0: [[0.0, 2.0], [10.0, 12.0]])
    segs = asr.transcribe(FakeRec(), np.zeros(14 * SR, np.float32), tmp_path)
    assert segs == [{"s": 0.0, "e": 2.0, "text": "hi."}, {"s": 10.0, "e": 12.0, "text": "hi."}]


class PickyRec(FakeRec):
    """Empty result on samples longer than `limit` s (the whole chunk, the padded chunk), text on shorter parts;
    counts decodes."""

    def __init__(self, limit):
        self.limit, self.calls = limit, 0

    def decode_streams(self, streams):
        self.calls += 1
        super().decode_streams(streams)
        for st in streams:
            if len(st.samples) / SR > self.limit:
                st.result = SimpleNamespace(tokens=[], timestamps=[])


def test_empty_chunk_split_into_parts():
    rec = PickyRec(6.0)
    segs = asr.transcribe(rec, np.zeros(30 * SR, np.float32), None, chunks=[[10.0, 20.0]])
    assert segs and all(s["text"] == "hi." for s in segs)
    assert all(10.0 <= s["s"] < s["e"] <= 20.0 for s in segs)
    assert [s["s"] for s in segs] == sorted(s["s"] for s in segs)


def test_empty_short_chunk_retried_padded():
    rec = PickyRec(2.9)  # 3 s chunk: empty as is and padded (4 s)
    assert asr.transcribe(rec, np.zeros(10 * SR, np.float32), None, chunks=[[0.0, 3.0]]) == []
    assert rec.calls == 2  # as is, then padded; under 2 * MIN_SPLIT_S it is not split
    rec = PickyRec(1.0)
    assert asr.transcribe(rec, np.zeros(10 * SR, np.float32), None, chunks=[[0.0, 1.5]]) == []
    assert rec.calls == 1  # under MIN_SPLIT_S: no retry


def test_always_empty_stops():
    rec = PickyRec(0.0)
    assert asr.transcribe(rec, np.zeros(40 * SR, np.float32), None, chunks=[[0.0, 30.0]]) == []
    assert rec.calls == 3  # as is, then two halves: both empty, not speech, no deeper split and no padded whole


def test_text_chunk_decoded_once():
    rec = PickyRec(100.0)
    asr.transcribe(rec, np.zeros(30 * SR, np.float32), None, chunks=[[0.0, 10.0], [12.0, 20.0]])
    assert rec.calls == 2


def test_padded_times_inside_chunk():
    class TailRec(FakeRec):
        """One sentence starting in the trailing pad and one at the very start."""

        def decode_streams(self, streams):
            for st in streams:
                n = len(st.samples) / SR
                st.result = SimpleNamespace(tokens=["▁a", ".", "▁b", "."], timestamps=[0.0, 0.1, n - 0.2, n - 0.1])

    got = asr._decode(TailRec(), np.zeros(2 * SR, np.float32), 0.5)
    assert got == [(0.0, pytest.approx(2.0), "a.")]  # "b." lies in the pad: dropped, "a." clamped to the chunk


class FakeVad:
    """Stands in for sherpa_onnx.VoiceActivityDetector: yields the given (start, end) seconds after flush()."""

    def __init__(self, spans):
        self.q = []
        self.spans = spans
        self.fed = 0

    def accept_waveform(self, w):
        self.fed += len(w)

    def flush(self):
        self.q = [SimpleNamespace(start=int(a * SR), samples=np.zeros(int((b - a) * SR), np.float32))
                  for a, b in self.spans]

    def empty(self):
        return not self.q

    @property
    def front(self):
        return self.q[0]

    def pop(self):
        self.q.pop(0)


def _fake_vad(monkeypatch, spans):
    fake = FakeVad(spans)
    monkeypatch.setattr(vad, "_detector", lambda *a, **k: fake)
    return fake


def test_vad_chunks_merge_and_cap(monkeypatch, tmp_path):
    fake = _fake_vad(monkeypatch, [(0, 1), (1.5, 2), (10, 11)])
    audio = np.zeros(12 * SR, np.float32)
    got = vad.vad_chunks(audio, tmp_path)
    assert fake.fed == len(audio)
    assert sum(got, []) == pytest.approx([0, 2.2, 9.8, 11.2])
    _fake_vad(monkeypatch, [(0, 1), (1.5, 2.5)])
    assert sum(vad.vad_chunks(audio, tmp_path, max_len=2.0), []) == pytest.approx([0, 1.2, 1.3, 2.7])


def test_vad_chunks_never_overlap(monkeypatch, tmp_path):
    # forced split at max_len: the 0.1 s gap is shorter than 2*PAD, split at its midpoint
    _fake_vad(monkeypatch, [(0, 30), (30.1, 40)])
    got = vad.vad_chunks(np.zeros(41 * SR, np.float32), tmp_path, max_len=30.0)
    assert len(got) == 2
    assert got[0][1] <= got[1][0]
    assert got[0][1] == pytest.approx(30.05) and got[1][0] == pytest.approx(30.05)


def test_vad_pad_clipped_to_audio(monkeypatch, tmp_path):
    _fake_vad(monkeypatch, [(0.1, 0.9)])
    got = vad.vad_chunks(np.zeros(1 * SR, np.float32), tmp_path)
    assert sum(got, []) == pytest.approx([0.0, 1.0])


@pytest.mark.models
def test_vad_silence_empty(vad_dir):
    assert vad.vad_chunks(np.zeros(3 * SR, np.float32), vad_dir) == []


@pytest.mark.models
def test_transcribe_public_sample(vad_dir, asr_dir, sample_wav):
    from audio2text.media import load_audio
    rec = asr.load_recognizer(asr_dir, asr.threads())
    segs = asr.transcribe(rec, load_audio(sample_wav), vad_dir)
    assert segs
    assert all(s["text"] and 0 <= s["s"] < s["e"] for s in segs)


def test_transcribe_given_chunks_skips_vad(monkeypatch, tmp_path):
    def no_vad(*a, **k):
        raise AssertionError("VAD must not run")

    monkeypatch.setattr(asr, "vad_chunks", no_vad)
    segs = asr.transcribe(FakeRec(), np.zeros(14 * SR, np.float32), tmp_path, chunks=[[10.0, 12.0]])
    assert segs == [{"s": 10.0, "e": 12.0, "text": "hi."}]


def test_transcribe_bar_to_chunk_ends():
    class Bar:
        n = 0.0

        def update(self, k):
            self.n += k
            seen.append(round(self.n, 2))

    seen = []
    asr.transcribe(FakeRec(), np.zeros(14 * SR, np.float32), None, chunks=[[0.0, 2.0], [10.0, 12.0]], bar=Bar())
    assert seen == [2.0, 12.0]
