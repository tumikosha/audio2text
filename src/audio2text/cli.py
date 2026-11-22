"""audio2text: аудиофайл или ссылка YouTube -> Markdown с говорящими, локально.

Ссылка: по умолчанию берутся субтитры на языке оригинала (без говорящих); нет таких субтитров или -tr —
скачивается аудио и распознаётся с говорящими."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import traceback
from pathlib import Path

import platformdirs

from . import pipeline, sources
from .markdown import write


def work_root() -> Path:
    return Path(platformdirs.user_cache_dir("audio2text")) / "work"


def work_dir(source: str, out: Path | None) -> Path:
    """Папка работы для пары (источник, выход): повтор той же команды переиспользует скачанное."""
    key = (source if sources.is_url(source) else str(Path(source).resolve())) + "|" + (
        str(out.resolve()) if out else "")
    return work_root() / hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _args(argv):
    ap = argparse.ArgumentParser(prog="audio2text", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="аудио- или видеофайл либо ссылка YouTube")
    ap.add_argument("out", type=Path, nargs="?",
                    help="файл .md; по умолчанию ./<название ролика или имя файла>.md")
    ap.add_argument("-tr", "--transcribe", action="store_true",
                    help="ссылка: не брать субтитры, распознавать аудио с говорящими")
    ap.add_argument("--keep", action="store_true", help="не удалять рабочую папку после успеха")
    return ap.parse_args(argv)


def run(a, work: Path) -> int:
    url = sources.is_url(a.source)
    if not url and not Path(a.source).is_file():
        raise SystemExit(f"файл не найден: {a.source}")
    if url:
        info = sources.youtube_info(a.source, work)
        out = a.out or Path(f"{sources.title_name(info.get('title') or 'transcript')}.md")
        doc = None
        if not a.transcribe:
            for track, kind in sources.caption_tracks(info):
                if doc := sources.captions_doc(a.source, info, track, kind, work):
                    break
        if doc is None:
            doc = pipeline.transcribe_file(sources.download(a.source, work))
            doc["source"], doc["title"] = a.source, info.get("title")
    else:
        out = a.out or Path(f"{Path(a.source).stem}.md")
        doc = pipeline.transcribe_file(Path(a.source))
    write(doc, out)
    spk = f", говорящих: {len(doc['speakers'])}" if doc.get("speakers") else ""
    print(f"{out}: сегментов {len(doc['segments'])}{spk}", flush=True)
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # консоль Windows в cp1251/cp866: не падать на символах из названий
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    a = _args(argv)
    work = None
    try:
        work = work_dir(a.source, a.out)
        return run(a, work)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, flush=True)
            return 2
        raise
    except KeyboardInterrupt:
        print("прервано", file=sys.stderr, flush=True)
        return 130
    except Exception as e:
        work = work or work_root()
        work.mkdir(parents=True, exist_ok=True)
        log = work / "error.log"
        log.write_text(traceback.format_exc(), encoding="utf-8")
        text = " ".join(str(e).split())
        print(f"ошибка: {type(e).__name__}: {text} (подробности: {log})", file=sys.stderr, flush=True)
        return 1
