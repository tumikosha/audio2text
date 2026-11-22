import tracemalloc
import wave

import numpy as np
import pytest

from audio2text import diarize as D
from audio2text import models
from audio2text.diarize import choose_labels
from audio2text.media import SR

from conftest import fetch_cached

DIMS = [192, 256, 512]   # the candidate speaker models' embedding sizes
WAV_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models"
VOICE_A = ["fangjun-sr-1.wav", "fangjun-sr-2.wav", "fangjun-sr-3.wav", "fangjun-test-sr-1.wav",
           "fangjun-test-sr-2.wav"]
VOICE_B = ["leijun-sr-1.wav", "leijun-sr-2.wav", "leijun-test-sr-1.wav", "leijun-test-sr-2.wav",
           "leijun-test-sr-3.wav"]


def _blobs(n_clusters, dim, per=30, noise=0.05, seed=1):
    rng = np.random.default_rng(seed)
    noise = noise * np.sqrt(192 / dim)   # same geometry whatever the dimension
    centers = np.eye(dim)[:n_clusters]
    emb = np.concatenate([c + noise * rng.standard_normal((per, dim)) for c in centers])
    truth = np.repeat(np.arange(n_clusters), per)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True), truth


def _same_partition(lab, truth):
    return len({(int(a), int(b)) for a, b in zip(lab, truth)}) == len(set(truth.tolist())) == len(set(lab.tolist()))


@pytest.mark.parametrize("dim", DIMS)
def test_choose_labels_one_cluster(dim):
    emb, _ = _blobs(1, dim, per=40)
    assert len(set(choose_labels(emb).tolist())) == 1


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("k", [2, 3, 4])
def test_choose_labels_k_clusters(k, dim):
    emb, truth = _blobs(k, dim)
    lab = choose_labels(emb)
    assert len(set(lab.tolist())) == k
    assert _same_partition(lab, truth)


@pytest.mark.parametrize("dim", DIMS)
def test_choose_labels_caps_at_four(dim):
    emb, _ = _blobs(6, dim, per=20)
    assert 2 <= len(set(choose_labels(emb).tolist())) <= 4


@pytest.mark.parametrize("dim", DIMS)
def test_short_call_one_speaker(dim):
    emb, _ = _blobs(2, dim, per=2)
    assert choose_labels(emb[:3]).tolist() == [0, 0, 0]


def _clouds(centers, sizes, noise, seed=1):
    """Rows around the given centres (unit vectors), noise scaled to the dimension as in _blobs."""
    rng = np.random.default_rng(seed)
    noise = noise * np.sqrt(192 / centers.shape[1])
    emb = np.concatenate([c + noise * rng.standard_normal((n, len(c))) for c, n in zip(centers, sizes)])
    return emb / np.linalg.norm(emb, axis=1, keepdims=True), np.repeat(np.arange(len(sizes)), sizes)


def _pair(dim, cos):
    """Two unit vectors in the first two coordinates with the given cosine between them."""
    t = np.sqrt((1 - cos) / (1 + cos))   # cosine of e0 + t e1 and e0 - t e1 is (1 - t^2) / (1 + t^2)
    a, b = np.eye(dim)[0] + t * np.eye(dim)[1], np.eye(dim)[0] - t * np.eye(dim)[1]
    return np.stack([a, b]) / np.linalg.norm(a)


def _count(emb):
    return len(set(choose_labels(emb).tolist()))


@pytest.mark.parametrize("dim", DIMS)
def test_one_voice_two_close_clouds_merged(dim):
    """A long one-voice recording that k-means splits in two tight clouds, centroid cosine 0.9: silhouette ~0.8, so
    only the centroid merge (MERGE_COS) makes it one speaker (LibriVox chapters split this way, cosine 0.89-0.94)."""
    emb, _ = _clouds(_pair(dim, 0.9), [40, 40], noise=0.01)
    assert _count(emb) == 1


@pytest.mark.parametrize("dim", DIMS)
def test_long_recording_two_voices_at_cosine_065(dim):
    """50 windows, centroid cosine 0.65: below MERGE_COS, two speakers."""
    emb, truth = _clouds(_pair(dim, 0.65), [25, 25], noise=0.02)
    lab = choose_labels(emb)
    assert len(set(lab.tolist())) == 2 and _same_partition(lab, truth)


@pytest.mark.parametrize("dim", DIMS)
def test_short_recording_odd_windows_one_speaker(dim):
    """14 windows, 3 of them off to the side (centroid cosine ~0.3, silhouette 0.2-0.25): one voice on a short
    recording, as on the 19-s "Me at the zoo"; with SIL_MIN alone it was two."""
    o = np.eye(dim)[0] + 2 * np.eye(dim)[1]
    emb, _ = _clouds(np.stack([np.eye(dim)[0], o / np.linalg.norm(o)]), [11, 3], noise=0.1)
    assert _count(emb) == 1


@pytest.mark.parametrize("dim", DIMS)
def test_short_recording_two_voices(dim):
    emb, truth = _clouds(np.eye(dim)[:2], [8, 8], noise=0.05)
    lab = choose_labels(emb)
    assert len(set(lab.tolist())) == 2 and _same_partition(lab, truth)


@pytest.mark.parametrize("dim", DIMS)
def test_short_recording_two_voices_moderate_silhouette(dim):
    """18 windows, distinct clouds with silhouette 0.43-0.47: above SHORT_SIL_MIN, two speakers."""
    emb, truth = _clouds(np.eye(dim)[:2], [9, 9], noise=0.08)
    lab = choose_labels(emb)
    assert len(set(lab.tolist())) == 2 and _same_partition(lab, truth)


@pytest.mark.parametrize("dim", DIMS)
def test_half_minute_one_voice_merged_at_lower_cosine(dim):
    """30 windows (~25 s): a one-voice split at centroid cosine 0.65 is merged (MID_MERGE_COS), though on a long
    recording it would stay two (MERGE_COS)."""
    emb, _ = _clouds(_pair(dim, 0.65), [15, 15], noise=0.02)
    assert _count(emb) == 1


@pytest.mark.parametrize("dim", DIMS)
def test_half_minute_two_voices_just_below_threshold(dim):
    """30 windows, centroid cosine 0.52 (the most two-person calls of 20-39 windows reach): two speakers."""
    emb, truth = _clouds(_pair(dim, 0.52), [15, 15], noise=0.02)
    lab = choose_labels(emb)
    assert len(set(lab.tolist())) == 2 and _same_partition(lab, truth)


class FakeStream:
    def accept_waveform(self, sr, samples):
        self.samples = np.asarray(samples, np.float32)

    def input_finished(self):
        pass


class FakeExtractor:
    """Embedding: the window's mean level in the first coordinate, 1 in the second (never a zero vector)."""

    def __init__(self, dim=256):
        self.dim = dim
        self.lengths = []

    def create_stream(self):
        return FakeStream()

    def compute(self, st):
        self.lengths.append(len(st.samples))
        v = np.zeros(self.dim, np.float32)
        v[0], v[1] = float(np.abs(st.samples).mean()), 1.0
        return list(v)


def test_diarize_no_speech_segments():
    segs = [{"s": 0.0, "e": 2.0, "text": ""}, {"s": 2.0, "e": 4.0, "text": "  "}]
    out, spk = D.diarize(FakeExtractor(), np.zeros(SR * 5, np.float32), segs)
    assert spk == {}
    assert [(s["speaker"], s["speaker_flag"]) for s in out] == [(None, None)] * 2


def test_diarize_silence_or_tiny_audio():
    assert D.diarize(FakeExtractor(), np.zeros(2 * SR, np.float32), []) == ([], {})
    for audio in (np.zeros(2 * SR, np.float32), np.zeros(int(0.3 * SR), np.float32), np.zeros(0, np.float32)):
        out, spk = D.diarize(FakeExtractor(), audio, [{"s": 0.0, "e": 0.3, "text": "да"}])
        assert [s["speaker"] for s in out] == ["SPEAKER_00"]
        assert list(spk) == ["SPEAKER_00"]
        assert np.isfinite(spk["SPEAKER_00"]["embedding"]).all()


def test_embed_one_padded_window_per_row():
    ext = FakeExtractor(dim=192)
    audio = np.ones(int(2.0 * SR), np.float32)
    wins = [(0, 0.0, 1.5), (0, 1.0, 2.5)]   # the second runs past the end of the audio
    e = D.embed(ext, audio, wins)
    assert e.shape == (2, 192) and e.dtype == np.float32
    assert np.allclose(np.linalg.norm(e, axis=1), 1)
    assert ext.lengths == [int(D.WIN_S * SR)] * 2


def test_embed_zero_vector_stays_finite():
    class Zero(FakeExtractor):
        def compute(self, st):
            return [0.0] * self.dim
    e = D.embed(Zero(), np.zeros(SR * 2, np.float32), [(0, 0.0, 1.5)])
    assert np.isfinite(e).all()


def test_embed_rows_from_streams_not_batch():
    """embed holds one stream at a time: no raw-audio batch array of all windows."""
    ext = FakeExtractor()
    wins = [(0, float(s), float(s) + 1.5) for s in range(0, 600)]
    audio = np.zeros(SR * 602, np.float32)
    tracemalloc.start()
    try:
        D.embed(ext, audio, wins)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 600 * int(D.WIN_S * SR) * 4 / 4, f"peak {peak / 2**20:.1f} MB"   # far below a 600-window batch


def test_silhouette_large_call_bounded_memory():
    """A 2-h call has ~8k windows: the silhouette must not build an n x n distance matrix."""
    emb, truth = _blobs(3, 256, per=3334)
    emb = emb.astype(np.float32)          # n = 10002; a full n x n float32 matrix alone is ~400 MB
    tracemalloc.start()
    try:
        lab = choose_labels(emb)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 150 * 2**20, f"peak {peak / 2**20:.0f} MB"
    assert len(lab) == len(emb) and _same_partition(lab, truth)  # every window labelled


def test_silhouette_subsample_same_choice(monkeypatch):
    emb, _ = _blobs(3, 256, per=1000, noise=0.6, seed=3)   # noisy: the choice is not trivial
    sub = choose_labels(emb)
    monkeypatch.setattr(D, "SIL_MAX_N", 10**9)       # full computation
    assert choose_labels(emb).tolist() == sub.tolist()


# ---- real model ----

@pytest.fixture(scope="session")
def extractor():
    assert models.SPEAKER is not None
    return D.load_extractor(models.ensure(models.SPEAKER), 2)


def _fetch(request, name):
    path = fetch_cached(request, name, f"{WAV_BASE}/{name}")
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def _voice(request, names, seconds=10.0):
    """seconds of one speaker: different recordings joined, repeated if they are too short."""
    x = np.concatenate([_fetch(request, n) for n in names])
    n = int(seconds * SR)
    return np.tile(x, n // len(x) + 1)[:n]


@pytest.mark.models
def test_diarize_two_voices_public(request, extractor):
    audio = np.concatenate([_voice(request, VOICE_A), _voice(request, VOICE_B)])
    segs = [{"s": 0.0, "e": 10.0, "text": "one"}, {"s": 10.0, "e": 20.0, "text": "two"}]
    out, spk = D.diarize(extractor, audio, segs)
    assert [s["speaker"] for s in out] == ["SPEAKER_00", "SPEAKER_01"]
    assert sorted(spk) == ["SPEAKER_00", "SPEAKER_01"]
    assert all(len(v["embedding"]) == extractor.dim for v in spk.values())


@pytest.mark.models
def test_real_extractor_silence_and_tiny(extractor):
    for audio in (np.zeros(2 * SR, np.float32), np.zeros(int(0.3 * SR), np.float32)):
        out, spk = D.diarize(extractor, audio, [{"s": 0.0, "e": 0.3, "text": "да"}])
        assert [s["speaker"] for s in out] == ["SPEAKER_00"]
        assert np.isfinite(spk["SPEAKER_00"]["embedding"]).all()
    assert D.diarize(extractor, np.zeros(2 * SR, np.float32), []) == ([], {})
