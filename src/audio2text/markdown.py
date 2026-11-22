"""Транскрипт в Markdown: заголовок, источник, абзацы по говорящим."""

from __future__ import annotations

from pathlib import Path

PARA_S = 60  # абзац без смены говорящего закрывается после этого срока на конце предложения
PARA_MAX_S = 120  # ... и на любом сегменте после этого срока


def _hms(s: float | None) -> str:
    if s is None:
        return "?"
    s = int(round(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _clock(s: float) -> str:
    s = int(s)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def to_markdown(doc: dict) -> str:
    """Заголовок, источник, затем абзацы: **SPEAKER_xx** `hh:mm:ss` текст. Новый абзац на каждой смене говорящего
    (смена в субтитрах YouTube, ">>", без имени: `—`) и в длинных монологах на конце предложения."""
    title = doc.get("title") or Path(str(doc["source"])).stem
    meta = [f"- Источник: {doc['source']}"]
    if doc.get("transcript"):
        meta.append(f"- Транскрипт: {doc['transcript']}")
    if doc.get("audio_s"):
        meta.append(f"- Длительность: {_hms(float(doc['audio_s']))}")
    paras, cur = [], None
    for s in doc["segments"]:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        who = s.get("speaker")
        long_ = cur and (s["s"] - cur["t"] >= PARA_MAX_S
                         or (s["s"] - cur["t"] >= PARA_S and cur["text"][-1][-1:] in ".!?…"))
        if cur is None or s.get("turn") or who != cur["who"] or long_:
            cur = {"who": who, "t": s["s"], "text": [], "turn": bool(s.get("turn"))}
            paras.append(cur)
        cur["text"].append(text)
    body = []
    for p in paras:
        head = f"**{p['who']}** " if p["who"] else ("— " if p["turn"] else "")
        body.append(f"{head}`{_clock(p['t'])}` {' '.join(p['text'])}")
    return f"# {title}\n\n" + "\n".join(meta) + "\n\n" + "\n\n".join(body) + "\n"


def write(doc: dict, out: Path) -> None:
    """Markdown в out, UTF-8, атомарно: сначала .tmp рядом, затем замена."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(to_markdown(doc), encoding="utf-8", newline="\n")
    tmp.replace(out)
