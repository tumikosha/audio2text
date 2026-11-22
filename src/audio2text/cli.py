"""audio2text: аудиофайл или ссылка YouTube -> Markdown с говорящими, локально.

Ссылка: по умолчанию берутся субтитры на языке оригинала (без говорящих); нет таких субтитров или -tr —
скачивается аудио и распознаётся с говорящими."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import traceback
from pathlib import Path

from . import media, models, pipeline, sources
from .markdown import write


def work_root() -> Path:
    return models.cache_root() / "work"


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
    ap.add_argument("--model", metavar="M",
                    help=f"модель распознавания: {', '.join(models.ASR_MODELS)} "
                         f"(по умолчанию {models.ASR_DEFAULT}) или путь к папке с моделью")
    ap.add_argument("--video", action="store_true",
                    help="ссылка: сохранить и видео рядом с текстом, <выход>.mp4")
    ap.add_argument("--mp3", action="store_true", help="сохранить и звук рядом с текстом, <выход>.mp3")
    ap.add_argument("--keep", action="store_true", help="не удалять рабочую папку после успеха")
    return ap.parse_args(argv)


def _same(a: Path, b: Path) -> bool:
    return a.exists() and b.exists() and os.path.samefile(a, b)  # samefile: на macOS/Windows регистр не важен


def _check_free(a, out: Path, src: Path | None = None) -> None:
    """Готовые .mp3/.mp4 не перезаписываются: ошибка до скачивания и распознавания. Исключение — mp3, который
    и есть исходный файл."""
    for on, ext in ((a.mp3, ".mp3"), (a.video, ".mp4")):
        p = out.with_suffix(ext)
        if on and p.exists() and not (src and _same(src, p)):
            raise SystemExit(f"файл уже есть: {p}; удалите его или укажите другой выход")


def _move(src: Path, dst: Path) -> None:
    """src -> dst через временное имя рядом с dst: рабочая папка может быть на другом диске."""
    tmp = dst.with_name(f".{dst.stem}.tmp{dst.suffix}")
    try:
        tmp.unlink(missing_ok=True)  # остаток оборванного запуска: на Windows переименование на него падает
        shutil.move(src, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def run(a, work: Path) -> int:
    asr_model = models.resolve_asr(a.model)  # неверный --model падает до чтения файла и сети
    url = sources.is_url(a.source)
    if a.video and not url:
        raise SystemExit("--video работает только для ссылки")
    if not url and not Path(a.source).is_file():
        raise SystemExit(f"файл не найден: {a.source}")
    if url:
        info = sources.youtube_info(a.source, work)
        out = a.out or Path(f"{sources.title_name(info.get('title') or 'transcript')}.md")
        _check_free(a, out)
        doc = None
        if not a.transcribe:
            for track, kind in sources.caption_tracks(info):
                if doc := sources.captions_doc(a.source, info, track, kind, work):
                    break
        # с --video звук для распознавания и mp3 берётся из видео: второй загрузки нет
        src = sources.download_video(a.source, work) if a.video else (
            sources.download(a.source, work) if doc is None or a.mp3 else None)
        if doc is None:
            doc = pipeline.transcribe_file(src, model=asr_model)
            doc["source"], doc["title"] = a.source, info.get("title")
    else:
        src = Path(a.source)
        out = a.out or Path(f"{src.stem}.md")
        _check_free(a, out, src)
        doc = pipeline.transcribe_file(src, model=asr_model)
    write(doc, out)
    spk = f", говорящих: {len(doc['speakers'])}" if doc.get("speakers") else ""
    print(f"{out}: сегментов {len(doc['segments'])}{spk}", flush=True)
    saved = []
    if a.mp3:
        mp3 = out.with_suffix(".mp3")
        if not url and _same(src, mp3):
            print(f"звук уже в mp3: {mp3}", flush=True)
        else:
            media.to_mp3(src, mp3)
            saved.append(mp3)
    if a.video:  # после mp3: он делается из этого же файла
        _move(src, out.with_suffix(".mp4"))
        saved.append(out.with_suffix(".mp4"))
    for p in saved:
        print(f"сохранено: {p}", flush=True)
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
        text = f"ошибка: {type(e).__name__}: {' '.join(str(e).split())}"
        try:
            work = work or work_root()
            work.mkdir(parents=True, exist_ok=True)
            log = work / "error.log"
            log.write_text(traceback.format_exc(), encoding="utf-8")
            text += f" (подробности: {log})"
        except Exception:  # журнал не записать (кэш недоступен): одна строка без пути
            pass
        print(text, file=sys.stderr, flush=True)
        return 1
