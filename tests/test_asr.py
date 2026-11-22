from types import SimpleNamespace

import numpy as np
import pytest

from audio2text import asr, vad
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
    assert len(segs) == 4 and all(s["text"] == "hi." for s in segs)
    assert all(10.0 <= s["s"] < s["e"] <= 20.0 for s in segs)
    assert [s["s"] for s in segs] == sorted(s["s"] for s in segs)
    # as is; halves 2.55 (text) + 7.45 (empty, split again: 1.91 text + 5.54 empty, split again: both text); padded
    # whole (empty, the halves win)
    assert rec.calls == 8


class PadRec(FakeRec):
    """Audio of ones, pad of zeros: empty without pad; padded samples longer than `long_s` give `long_text`,
    shorter ones `short_text` (None: empty)."""

    def __init__(self, long_s, long_text, short_text):
        self.long_s, self.long_text, self.short_text, self.calls = long_s, long_text, short_text, 0

    def decode_streams(self, streams):
        self.calls += 1
        for st in streams:
            n = len(st.samples) / SR
            text = None if st.samples[0] != 0 else self.long_text if n > self.long_s else self.short_text
            st.result = (SimpleNamespace(tokens=[], timestamps=[]) if text is None else
                         SimpleNamespace(tokens=[f"▁{w}" for w in text.split()], timestamps=[0.1 * i for i in
                                                                                         range(len(text.split()))]))


def test_padded_whole_wins_with_more_words():
    rec = PadRec(10.5, "one two three.", "hi.")  # only the padded whole (11 s) is long
    segs = asr.transcribe(rec, np.ones(12 * SR, np.float32), None, chunks=[[0.0, 10.0]])
    assert [s["text"] for s in segs] == ["one two three."] and rec.calls == 4  # as is, two halves, padded whole
    assert 0.0 <= segs[0]["s"] < segs[0]["e"] <= 10.0


def test_short_chunk_both_halves_empty_padded_whole():
    rec = PadRec(6.5, "hi there.", None)  # only the padded whole 6 s chunk (7 s) gives text
    segs = asr.transcribe(rec, np.ones(8 * SR, np.float32), None, chunks=[[1.0, 7.0]])
    assert [s["text"] for s in segs] == ["hi there."] and rec.calls == 4
    assert 1.0 <= segs[0]["s"] < segs[0]["e"] <= 7.0
    rec = PadRec(100.0, None, None)  # 6 s, always empty: as is, two halves, padded whole, nothing more
    assert asr.transcribe(rec, np.ones(8 * SR, np.float32), None, chunks=[[1.0, 7.0]]) == []
    assert rec.calls == 4


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
    assert rec.calls == 3  # as is, two halves: both empty, not speech; >= WHOLE_RETRY_S: no padded whole, no deeper split


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


def test_unpadded_keeps_zero_length_sentence():
    class SameRec(FakeRec):
        def decode_streams(self, streams):
            for st in streams:
                n = len(st.samples) / SR
                st.result = SimpleNamespace(tokens=["▁a", "."], timestamps=[n, n])

    assert asr._decode(SameRec(), np.zeros(2 * SR, np.float32)) == [(2.0, 2.0, "a.")]  # as sentences() gives it


class TruncRec(FakeRec):
    """Unpadded samples (audio, not the zero pad, at [0]): words only in the first `head` s, or everywhere when
    `head` is None. Padded samples: words every 0.3 s outside the pad, or only in the first second when `stuck`."""

    def __init__(self, head, stuck=False):
        self.head, self.stuck, self.calls = head, stuck, 0

    def decode_streams(self, streams):
        self.calls += 1
        for st in streams:
            n = len(st.samples) / SR
            padded = st.samples[0] == 0
            lo, hi = (asr.PAD_S, n - asr.PAD_S) if padded else (0.0, n)
            if (self.stuck and padded) or (not padded and self.head is not None):
                hi = min(hi, 1.0 if padded else self.head)
            ts = list(np.arange(lo + 0.1, hi, 0.3))
            st.result = SimpleNamespace(tokens=["▁w"] * len(ts), timestamps=ts)


def _speech_audio(s):
    audio = np.full(s * SR, 0.1, np.float32)
    audio[0] = 0.2  # not zero: tells the chunk from the pad
    return audio


def test_truncated_chunk_filled():
    rec = TruncRec(head=1.0)
    segs = asr.transcribe(rec, _speech_audio(10), None, chunks=[[0.0, 10.0]])
    words = " ".join(s["text"] for s in segs).split()
    assert rec.calls == 2  # as is, the tail hole (under PIECE_S) padded
    assert len(words) > 25  # 3 words before the hole, ~29 in it
    assert all(0.0 <= s["s"] <= s["e"] <= 10.0 for s in segs)


def test_normal_chunk_decoded_once():
    rec = TruncRec(head=None)
    segs = asr.transcribe(rec, _speech_audio(10), None, chunks=[[0.0, 10.0]])
    assert rec.calls == 1 and len(segs[0]["text"].split()) == 33


def test_hole_without_words_kept_and_stops():
    rec = TruncRec(head=1.0, stuck=True)  # the hole decodes to almost nothing: music, noise
    segs = asr.transcribe(rec, _speech_audio(12), None, chunks=[[0.0, 12.0]])
    assert [s["text"] for s in segs] == ["w w w"]  # first pass as is
    assert rec.calls == 2  # as is, the first piece of the 11 s hole (over PIECE_S); it has no words: stop there


def test_long_hole_in_pieces_until_no_words():
    class TwoPartRec(TruncRec):
        """As TruncRec(head=1.0), but padded samples after the first two hole pieces give nothing."""

        def decode_streams(self, streams):
            super().decode_streams(streams)
            if self.calls > 3:
                streams[0].result = SimpleNamespace(tokens=[], timestamps=[])

    rec = TwoPartRec(head=1.0)
    segs = asr.transcribe(rec, _speech_audio(30), None, chunks=[[0.0, 30.0]])
    ts = [s["s"] for s in segs]
    assert rec.calls == 4  # as is, two pieces with words, the third without: stop
    words = " ".join(s["text"] for s in segs).split()
    assert 3 + 2 * 15 <= len(words) < 3 + 3 * 30 and ts == sorted(ts) and segs[-1]["e"] <= 30.0


def test_hole_after_token_at_zero_not_duplicated():
    class ZeroRec(TruncRec):
        """Unpadded: one word at 0.0 s, then nothing; padded: TruncRec."""

        def decode_streams(self, streams):
            super().decode_streams(streams)
            if not streams[0].samples[0] == 0:
                streams[0].result = SimpleNamespace(tokens=["▁first"], timestamps=[0.0])

    segs = asr.transcribe(ZeroRec(head=None), _speech_audio(10), None, chunks=[[0.0, 10.0]])
    words = " ".join(s["text"] for s in segs).split()
    assert words[0] == "first" and words.count("first") == 1 and len(words) > 25


def test_silent_hole_not_decoded():
    rec = TruncRec(head=1.0)
    audio = _speech_audio(10)
    audio[int(1.5 * SR):] = 0.0  # nothing after the words
    asr.transcribe(rec, audio, None, chunks=[[0.0, 10.0]])
    assert rec.calls == 1


class HoleRec(FakeRec):
    """Unpadded samples: the given (token, time) pairs. Padded: a word every `step` s outside the pad, then '.'."""

    def __init__(self, first, step=0.3):
        self.first, self.step, self.calls = first, step, 0

    def decode_streams(self, streams):
        self.calls += 1
        for st in streams:
            n = len(st.samples) / SR
            if st.samples[0] != 0:
                st.result = SimpleNamespace(tokens=[k for k, _ in self.first], timestamps=[t for _, t in self.first])
                continue
            ts = list(np.arange(asr.PAD_S + 0.1, n - asr.PAD_S, self.step))
            st.result = SimpleNamespace(tokens=["▁w"] * len(ts) + ["."], timestamps=ts + [ts[-1] + 0.05])


def _text(segs):
    return " ".join(s["text"] for s in segs)


def test_head_hole_filled_before_first_token():
    rec = HoleRec([("▁tail", 3.6), (".", 3.9)])
    segs = asr.transcribe(rec, _speech_audio(5), None, chunks=[[0.0, 5.0]])
    words = _text(segs).split()
    assert rec.calls == 2 and words[0] == "w" and words[-1] == "tail." and len(words) > 8
    assert len(segs) == 1  # the '.' of the insert before a lowercase token is dropped: one sentence


def test_gap_hole_in_order_and_sentence_kept():
    rec = HoleRec([("▁a", 0.1), ("▁b", 0.4), ("▁c", 4.6), ("▁d", 4.9), (".", 5.1)])
    segs = asr.transcribe(rec, _speech_audio(6), None, chunks=[[0.0, 6.0]])
    words = _text(segs).split()
    assert rec.calls == 2 and words[:2] == ["a", "b"] and words[-2:] == ["c", "d."] and set(words[2:-2]) == {"w"}
    assert len(segs) == 1


def test_insert_before_capital_keeps_its_period():
    rec = HoleRec([("▁a", 0.1), ("▁b", 0.4), ("▁Next", 4.6), (".", 5.1)])
    segs = asr.transcribe(rec, _speech_audio(6), None, chunks=[[0.0, 6.0]])
    assert len(segs) == 2 and segs[0]["text"].endswith("w.") and segs[1]["text"] == "Next."
    assert segs[0]["e"] <= segs[1]["s"]


def test_words_per_second_threshold():
    first = [("▁a", 0.1), ("▁b", 0.4)]  # tail hole of about 9 s of speech
    rec = HoleRec(first, step=1 / (asr.HOLE_WPS + 0.2))
    assert len(_text(asr.transcribe(rec, _speech_audio(10), None, chunks=[[0.0, 10.0]])).split()) > 10
    rec = HoleRec(first, step=1 / (asr.HOLE_WPS - 0.2))
    assert _text(asr.transcribe(rec, _speech_audio(10), None, chunks=[[0.0, 10.0]])) == "a b"
    assert rec.calls == 2


def test_quiet_hole_background_not_decoded():
    audio = _speech_audio(10)
    audio[int(1.5 * SR):] = 0.02  # 14 dB under the words: background, not the speaker
    rec = HoleRec([("▁a", 0.1), ("▁b", 0.4), ("▁c", 0.7)])
    assert _text(asr.transcribe(rec, audio, None, chunks=[[0.0, 10.0]])) == "a b c" and rec.calls == 1


def test_holes_per_chunk_limited():
    first = [("▁a", t) for t in (0.1, 6.5, 13.0, 19.5, 26.0)]  # 4 gaps and the tail: 5 holes
    rec = HoleRec(first)
    asr.transcribe(rec, _speech_audio(30), None, chunks=[[0.0, 30.0]])
    assert rec.calls == 1 + asr.MAX_HOLES


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
