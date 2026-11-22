"""Источники: ссылки YouTube (метаданные, субтитры, скачивание аудио) через yt-dlp.

yt-dlp запускается через `uvx --quiet --with deno "yt-dlp[default]@latest"`: свежий yt-dlp и JS-рантайм deno
из колеса, ничего не нужно ставить в систему. Переменная окружения AUDIO2TEXT_YTDLP_NO_DENO=1 переключает на
запасной путь: deno из PATH, иначе node из PATH (`--js-runtimes node`), иначе понятная ошибка. Это выход для ОС,
где колесо deno не работает (Windows не проверялась).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from audio2text.media import ffmpeg_exe

_UV_HINT = "нужен uv: https://docs.astral.sh/uv/"
_JS_HINT = ("для ссылок YouTube нужен JS-рантайм: установите deno (https://deno.com) или node "
            "(https://nodejs.org) либо уберите AUDIO2TEXT_YTDLP_NO_DENO")
_BASE = ["uvx", "--quiet"]
_PKG = "yt-dlp[default]@latest"


def is_url(s: str) -> bool:
    return bool(re.match(r"^https?://", s))


def fallback_cmd() -> list[str]:
    """Команда без колеса deno: JS-рантайм берётся из PATH."""
    cmd = [*_BASE, _PKG]
    if shutil.which("deno"):
        return cmd
    if shutil.which("node"):
        return [*cmd, "--js-runtimes", "node"]
    raise SystemExit(_JS_HINT)


def ytdlp() -> list[str]:
    if not shutil.which("uvx"):
        raise SystemExit(_UV_HINT)
    if os.environ.get("AUDIO2TEXT_YTDLP_NO_DENO") == "1":
        return fallback_cmd()
    return [*_BASE, "--with", "deno", _PKG]


def _run(args: list[str]) -> str:
    """yt-dlp с аргументами args; stdout. Ошибка -> SystemExit с последней непустой строкой stderr."""
    r = subprocess.run([*ytdlp(), *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    if r.returncode != 0:
        lines = [ln for ln in r.stderr.splitlines() if ln.strip()]
        last = lines[-1].strip() if lines else f"код {r.returncode}"
        raise SystemExit(f"yt-dlp: {last}{'; повторите запуск' if 'HTTP Error 403' in last else ''}")
    return r.stdout


def youtube_info(url: str, work: Path) -> dict:
    """yt-dlp -J для одного видео (не плейлиста), кэш в <work>/info.json."""
    p = work / "info.json"
    if not p.exists():
        work.mkdir(parents=True, exist_ok=True)
        p.write_text(_run(["-J", "--no-playlist", "--playlist-items", "1", url]), encoding="utf-8")
    info = json.loads(p.read_text(encoding="utf-8"))
    if info.get("_type") == "playlist" or "entries" in info:
        p.unlink(missing_ok=True)
        raise SystemExit("ссылка на плейлист или канал: укажите ссылку на одно видео")
    return info


def caption_tracks(info: dict) -> list[tuple[str, str]]:
    """[(дорожка, вид)] субтитров на языке оригинала, лучшие первыми: авторские, затем автоматические
    <lang>-orig. Переводы не берутся; язык неизвестен и неоднозначен -> пусто."""
    lang = info.get("language")
    subs = [k for k in (info.get("subtitles") or {}) if k != "live_chat"]
    auto = info.get("automatic_captions") or {}
    if lang:
        out = [(k, "subtitles") for k in sorted(subs, key=len) if k == lang or k.startswith(lang + "-")]
        if f"{lang}-orig" in auto:
            out.append((f"{lang}-orig", "auto"))
        return out
    orig = [k for k in auto if k.endswith("-orig")]
    return [(orig[0], "auto")] if len(orig) == 1 else []


def parse_json3(data: dict) -> list[dict]:
    """json3 субтитры YouTube -> [{s, e, text, turn}]; turn — смена говорящего (">>" в дорожке)."""
    out = []
    for ev in data.get("events") or []:
        if ev.get("aAppend") or not ev.get("segs"):
            continue
        text = "".join(s.get("utf8", "") for s in ev["segs"]).replace("\n", " ").strip()
        if not text:
            continue
        turn = text.startswith(">>") or any(s.get("isSpeakerChange") for s in ev["segs"])
        text = text.lstrip("> ").strip()
        s = ev.get("tStartMs", 0) / 1000
        out.append({"s": round(s, 2), "e": round(s + ev.get("dDurationMs", 0) / 1000, 2), "text": text, "turn": turn})
    return out


def captions_doc(url: str, info: dict, track: str, kind: str, work: Path) -> dict | None:
    """Транскрипт одной дорожки; None, если YouTube отдал её без текста (бывает с авторскими)."""
    d = work / "captions" / f"{kind}-{track}"
    d.mkdir(parents=True, exist_ok=True)
    _run(["--skip-download", "--no-playlist", "--sub-format", "json3", "--sub-langs", track,
          "--write-subs" if kind == "subtitles" else "--write-auto-subs", "-o", str(d / "sub.%(ext)s"), url])
    files = sorted(d.glob("*.json3"))
    try:
        segs = parse_json3(json.loads(files[0].read_text(encoding="utf-8"))) if files else []
    except ValueError:
        segs = []
    if not segs:
        print(f"субтитры YouTube {track} ({kind}) пусты, пропускаю", flush=True)
        return None
    print(f"субтитры YouTube {track} ({kind}), распознавание не запускалось", flush=True)
    return {"source": url, "title": info.get("title"), "transcript": f"субтитры YouTube {track} ({kind})",
            "audio_s": info.get("duration"), "speakers": {}, "segments": segs}


def _finished(p: Path) -> bool:
    """Готовый файл загрузки, а не фрагмент или служебный файл yt-dlp (.part, .part-Frag3, .ytdl, .temp)."""
    return p.is_file() and ".part" not in p.name and not p.name.endswith((".ytdl", ".temp"))


def download(url: str, work: Path) -> Path:
    """Лучшее аудио <url> в <work>/download/; прежняя загрузка переиспользуется."""
    d = work / "download"
    done = [p for p in sorted(d.glob("*")) if _finished(p)] if d.exists() else []
    if done:
        return done[0]
    d.mkdir(parents=True, exist_ok=True)
    print(f"скачиваю {url}", flush=True)
    _run(["-f", "bestaudio/best", "--no-playlist", "--ffmpeg-location", ffmpeg_exe(),
          "-o", str(d / "src.%(ext)s"), url])
    files = [p for p in sorted(d.glob("src.*")) if _finished(p)]
    if not files:
        raise SystemExit(f"yt-dlp ничего не скачал: {url}")
    return files[0]


def download_video(url: str, work: Path) -> Path:
    """Видео <url> в <work>/video/: лучшее качество, которое играют обычные плееры (h264 + aac в mp4); дорожки
    склеивает ffmpeg из imageio-ffmpeg. Прежняя загрузка переиспользуется."""
    p = work / "video" / "src.mp4"  # только итог: src.f137.mp4 (одна дорожка) и src.temp.mp4 (склейка) — нет
    if p.is_file():
        return p
    d = p.parent
    d.mkdir(parents=True, exist_ok=True)
    print(f"скачиваю видео {url}", flush=True)
    _run(["-f", "bv*+ba/b", "-S", "vcodec:h264,res,acodec:aac", "--merge-output-format", "mp4",
          "--remux-video", "mp4", "--no-playlist",
          "--ffmpeg-location", ffmpeg_exe(), "-o", str(d / "src.%(ext)s"), url])
    if not p.is_file():
        raise SystemExit(f"yt-dlp ничего не скачал: {url}")
    return p


_RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\.|$)", re.IGNORECASE)


def title_name(title: str) -> str:
    """Имя файла из названия видео: без символов путей и shell, до 120 знаков; зарезервированные
    в Windows имена (con, nul, com1, lpt1...) получают "_" после имени: "con" -> "con_", "nul.txt" -> "nul_.txt"
    (Windows проверяет часть до первой точки, поэтому "_" в конце "nul.txt_" не помог бы)."""
    name = re.sub(r'[\\/:*?"<>|\s]+', " ", title).strip(" .")[:120].strip() or "transcript"
    m = _RESERVED.match(name)
    return f"{m.group(1)}_{name[m.end(1):]}" if m else name
