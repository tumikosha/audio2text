"""Диаризация: эмбеддинги голоса ONNX-моделью sherpa-onnx по окнам 1.5 с, k-means, число говорящих 1-4 по
косинусному силуэту, затем слияние кластеров с близкими центроидами. Кластеризация и разметка — на numpy."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import sherpa_onnx

from .media import SR

WIN_S, HOP_S = 1.5, 0.75
FRAME_S = 0.25
MIN_RUN_S = 0.5
MIN_WINDOWS = 4
NEAREST_S = 2.0
MIXED_SHARE = 0.8
SIL_MIN = 0.1   # best silhouette below this: one speaker
# One voice still splits: k-means cuts a long one-voice recording in two clouds whose centroids are close
# (cosine 0.89-0.94 on five LibriVox readers), with a silhouette up to 0.24. Two different people stay far apart:
# at most 0.69 over 145 two-person phone calls, at most 0.21 on 10 pairs of LibriVox readers. Merged above this.
MERGE_COS = 0.8
# Short recordings (under SHORT_WINDOWS windows, ~15 s of speech): each phrase can form its own cluster, since
# neighbouring windows overlap by half. One voice: best silhouette up to 0.28 on the 19-s "Me at the zoo"; up to 0.36
# on 15-s LibriVox excerpts, but their centroids at cosine 0.59 and up, so the merge takes them; two voices (7 s + 7 s of two readers): silhouette 0.59 and up,
# cosine 0.18 at most; the two such phone calls: silhouette 0.38 and 0.49, and merging at cosine 0.4 already
# made one of them a single speaker.
SHORT_WINDOWS = 20
SHORT_SIL_MIN = 0.33
SHORT_MERGE_COS = 0.5
SIL_MAX_N = 2000   # silhouette on at most this many windows (even stride): a 2-h call has ~8k, n x n would be ~0.5 GB


def load_extractor(model_dir: Path, threads: int) -> sherpa_onnx.SpeakerEmbeddingExtractor:
    """Speaker embedding model: the single .onnx file in model_dir."""
    onnx = sorted(Path(model_dir).glob("*.onnx"))
    if len(onnx) != 1:
        raise SystemExit(f"в папке модели спикеров должен быть один .onnx: {model_dir}")
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(onnx[0]), num_threads=threads, provider="cpu")
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


def windows(spans, total_s):
    """[(segment index, start, end)] of the 1.5 s windows over the live segments."""
    out = []
    for i, (a, b) in spans.items():
        if b - a < WIN_S:
            mid = (a + b) / 2
            s = min(max(0.0, mid - WIN_S / 2), max(0.0, total_s - WIN_S))
            out.append((i, s, s + WIN_S))
            continue
        s = a
        while s + WIN_S < b:
            out.append((i, s, s + WIN_S))
            s += HOP_S
        out.append((i, b - WIN_S, b))
    return out


def embed(extractor, audio, wins) -> np.ndarray:
    """L2-normalised embeddings, one row per window; one stream at a time, each window zero-padded to WIN_S."""
    n = int(WIN_S * SR)
    out = np.zeros((len(wins), extractor.dim), np.float32)
    chunk = np.zeros(n, np.float32)
    for k, (_, s, _) in enumerate(wins):
        a = max(0, int(s * SR))
        piece = audio[a:a + n]
        chunk[:] = 0.0
        chunk[:len(piece)] = piece
        st = extractor.create_stream()
        st.accept_waveform(SR, chunk)
        st.input_finished()
        out[k] = extractor.compute(st)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out


def cluster(e, k, restarts=20):
    """Cluster index per row: cosine k-means, the best of `restarts` random starts (fixed seed). Average-linkage
    clustering split off a few odd windows ("секундочку", noise) and left both voices in one cluster on 5 calls of 10."""
    if len(e) <= k:
        return np.arange(len(e))
    rng = np.random.default_rng(0)
    best = None
    for _ in range(restarts):
        cent = e[rng.choice(len(e), k, replace=False)]
        for _ in range(50):
            lab = (e @ cent.T).argmax(1)
            new = np.stack([e[lab == c].mean(0) if (lab == c).any() else cent[c] for c in range(k)])
            new /= np.linalg.norm(new, axis=1, keepdims=True)
            if np.allclose(new, cent):
                break
            cent = new
        score = (e * cent[lab]).sum()
        if best is None or score > best[0]:
            best = (score, lab)
    return best[1]


def silhouette(e: np.ndarray, lab: np.ndarray) -> float:
    """Mean silhouette with cosine distance; -1 when fewer than two non-empty clusters.
    Over SIL_MAX_N rows it is computed on an even-stride subsample (deterministic); the labels are not touched."""
    if len(lab) > SIL_MAX_N:
        idx = np.linspace(0, len(lab) - 1, SIL_MAX_N).round().astype(int)
        e, lab = e[idx], lab[idx]
    ids = np.unique(lab)
    if len(ids) < 2:
        return -1.0
    d = 1.0 - e @ e.T
    sizes = {c: int((lab == c).sum()) for c in ids}
    mean_to = np.stack([d[:, lab == c].sum(1) for c in ids], axis=1)        # (n, clusters): sum of distances
    own = np.searchsorted(ids, lab)
    n_own = np.array([sizes[c] for c in lab])
    a = mean_to[np.arange(len(lab)), own] / np.maximum(n_own - 1, 1)
    other = mean_to / np.array([sizes[c] for c in ids])
    other[np.arange(len(lab)), own] = np.inf
    b = other.min(1)
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
    s[n_own == 1] = 0.0
    return float(s.mean())


def merge_close(e: np.ndarray, lab: np.ndarray, cos_min: float) -> np.ndarray:
    """Labels 0..m-1 after merging the closest pair of clusters while their centroid cosine is >= cos_min."""
    lab = lab.copy()
    while True:
        ids = np.unique(lab)
        if len(ids) < 2:
            break
        c = np.stack([e[lab == i].mean(0) for i in ids])
        c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
        s = c @ c.T
        np.fill_diagonal(s, -np.inf)
        a, b = np.unravel_index(s.argmax(), s.shape)
        if s[a, b] < cos_min:
            break
        lab[lab == ids[b]] = ids[a]
    return np.unique(lab, return_inverse=True)[1]


def choose_labels(emb: np.ndarray, max_k: int = 4) -> np.ndarray:
    """Cluster label per row; k in 2..max_k by best mean cosine silhouette, k = 1 if that is below SIL_MIN
    (SHORT_SIL_MIN under SHORT_WINDOWS rows), then clusters with centroid cosine >= MERGE_COS (SHORT_MERGE_COS)
    merged."""
    if len(emb) < MIN_WINDOWS:
        return np.zeros(len(emb), int)
    e = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    best_sil, best = -np.inf, None
    for k in range(2, max_k + 1):
        lab = cluster(e, k)
        sil = silhouette(e, lab)
        if sil > best_sil:
            best_sil, best = sil, lab
    short = len(e) < SHORT_WINDOWS
    if best is None or best_sil < (SHORT_SIL_MIN if short else SIL_MIN):
        return np.zeros(len(emb), int)
    return merge_close(e, best, SHORT_MERGE_COS if short else MERGE_COS)


def join(pieces):
    """[[start, end, cluster]] with neighbours of the same cluster joined."""
    out = []
    for s, e, c in pieces:
        if out and out[-1][2] == c:
            out[-1][1] = e
        else:
            out.append([s, e, c])
    return out


def runs(spans, wins, lab):
    """[(start, end, cluster, segment index)]: per 0.25 s frame the majority cluster of the covering windows,
    the shortest run under 0.5 s joined to the previous one (the next, at the start of a segment) until none is left."""
    out = []
    for i, (a, b) in spans.items():
        mine = [(s, e, c) for (j, s, e), c in zip(wins, lab) if j == i]
        frames = []
        t = a
        while t < b - 1e-6:
            mid = min(t + FRAME_S / 2, b)
            votes = [c for s, e, c in mine if s <= mid < e]
            if not votes:  # a short segment's window is centred, still pick it
                votes = [min(mine, key=lambda w: abs((w[0] + w[1]) / 2 - mid))[2]]
            frames.append((t, min(t + FRAME_S, b), max(set(votes), key=votes.count)))
            t += FRAME_S
        seg = join(frames)
        while len(seg) > 1:
            k = min(range(len(seg)), key=lambda k: seg[k][1] - seg[k][0])
            if seg[k][1] - seg[k][0] >= MIN_RUN_S:
                break
            j = k - 1 if k else 1  # the previous run takes it, the next one at the start
            seg[j] = [min(seg[j][0], seg[k][0]), max(seg[j][1], seg[k][1]), seg[j][2]]
            del seg[k]
            seg = join(seg)
        out += [(s, e, int(c), i) for s, e, c in seg]
    return out


def assign(segments: list[dict], turns: list[tuple[float, float, str]]) -> dict[int, tuple[str | None, str | None]]:
    """{segment index: (turn speaker or None, flag or None)} for the live segments with text."""
    out = {}
    for i, seg in enumerate(segments):
        if not seg["text"].strip():
            continue
        t = (seg["s"], seg["e"])
        overlap = Counter()
        for a, b, k in turns:
            o = min(t[1], b) - max(t[0], a)
            if o > 0:
                overlap[k] += o
        if overlap:
            k, top = overlap.most_common(1)[0]
            out[i] = (k, None if top >= MIXED_SHARE * sum(overlap.values()) else "speaker_mixed")
            continue
        gap, k = min(((max(a - t[1], t[0] - b), k) for a, b, k in turns), default=(None, None))
        out[i] = (k, "speaker_nearest") if gap is not None and gap <= NEAREST_S else (None, "speaker_unknown")
    return out


def first_speech_names(labels: dict[int, tuple[str | None, str | None]], segments: list[dict]) -> dict[str, str]:
    """Turn speaker -> SPEAKER_00, SPEAKER_01 ... in order of first speech."""
    order: dict[str, str] = {}
    for i in sorted(labels, key=lambda i: segments[i]["s"]):
        k = labels[i][0]
        if k is not None and k not in order:
            order[k] = f"SPEAKER_{len(order):02d}"
    return order


def diarize(extractor, audio: np.ndarray, segments: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    """Segments with `speaker` / `speaker_flag`, and {SPEAKER_xx: {speech_s, embedding}}; the speaker count is picked
    by silhouette and centroid merge (choose_labels). No segments with text: ([], {}) or the segments unlabelled."""
    spans = {i: (s["s"], s["e"]) for i, s in enumerate(segments) if s["text"].strip()}
    wins = windows(spans, len(audio) / SR)
    emb = embed(extractor, audio, wins) if wins else np.zeros((0, extractor.dim), np.float32)
    if len(wins) < MIN_WINDOWS:  # too short to tell voices apart: one voice
        lab = np.zeros(len(wins), int)
    else:
        lab = choose_labels(emb)
    raw = [(s, e, f"c{c}") for s, e, c, _ in runs(spans, wins, lab)]
    labels = assign(segments, raw)
    names = first_speech_names(labels, segments)
    turns = [(round(s, 2), round(e, 2), names[c]) for s, e, c in raw if c in names]
    new = []
    for i, seg in enumerate(segments):
        k, flag = labels.get(i, (None, None))
        new.append({**seg, "speaker": names.get(k) if k is not None else None, "speaker_flag": flag})
    speech = Counter()
    for s, e, spk in turns:
        speech[spk] += e - s
    speakers = {}
    for spk, v in sorted(speech.items()):
        rows = [emb[w] for w, c in enumerate(lab) if names.get(f"c{c}") == spk]
        mean = np.mean(rows, axis=0)
        mean = mean / max(float(np.linalg.norm(mean)), 1e-12)
        speakers[spk] = {"speech_s": round(v, 1), "embedding": [round(float(x), 5) for x in mean]}
    return new, speakers
