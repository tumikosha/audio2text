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


_DESCRIPTION = """\
Превращает аудио- или видеофайл либо ссылку на ролик YouTube в текст (файл .md): кто говорил, когда и что.
Всё считается на вашем компьютере. По ссылке сначала берутся субтитры ролика (быстро, без говорящих)."""

_EPILOG = """\
примеры:
  audio2text запись.mp3
  audio2text интервью.mp4 итог.md
  audio2text "https://www.youtube.com/watch?v=jNQXAC9IVRw"
  audio2text "https://www.youtube.com/watch?v=jNQXAC9IVRw" -tr
  audio2text "https://www.youtube.com/watch?v=jNQXAC9IVRw" --video --mp3
  audio2text english_talk.m4a --model parakeet-tdt-0.6b-v2"""


def _args(argv):
    ap = argparse.ArgumentParser(prog="audio2text", description=_DESCRIPTION, epilog=_EPILOG,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="аудио- или видеофайл с диска либо ссылка на один ролик YouTube")
    ap.add_argument("out", type=Path, nargs="?",
                    help="куда сохранить текст; без него — <имя файла>.md или <название ролика>.md в текущей папке")
    ap.add_argument("-tr", "--transcribe", action="store_true",
                    help="для ссылки: не брать субтитры, а распознать звук с разделением на говорящих")
    ap.add_argument("--video", action="store_true", help="только для ссылки: сохранить ролик в mp4 рядом с текстом")
    ap.add_argument("--mp3", action="store_true", help="сохранить звук в mp3 рядом с текстом (ссылка или файл)")
    ap.add_argument("--model", metavar="M",
                    help=f"модель распознавания: {', '.join(models.ASR_MODELS)} "
                         f"(по умолчанию {models.ASR_DEFAULT}) или путь к папке с моделью")
    ap.add_argument("--keep", action="store_true",
                    help="не удалять временные файлы (скачанный звук) после успешной работы")
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


def _save(jobs: list[tuple[str, Path, Path]]) -> None:
    """jobs: [(вид, источник, цель)], вид "mp3" или "видео". Всё сначала готовится во временные файлы рядом
    с целями (рабочая папка может быть на другом диске), затем os.replace каждого. При любой ошибке временные
    файлы удаляются, уже появившиеся цели тоже: ни одной цели, повтор той же команды не упрётся в «файл уже есть»."""
    tmps = [dst.with_name(f".{dst.stem}.part{dst.suffix}") for _, _, dst in jobs]
    placed, moved = [], None
    try:
        for stage in ("prepare", "replace"):
            for (kind, src, dst), tmp in zip(jobs, tmps):
                try:
                    if stage == "replace":
                        os.replace(tmp, dst)
                        placed.append(dst)
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    tmp.unlink(missing_ok=True)  # остаток оборванного запуска: на Windows перенос на него падает
                    if kind == "mp3":
                        media.to_mp3(src, tmp)
                    else:
                        shutil.move(src, tmp)
                        moved = (tmp, src)
                except OSError as e:
                    raise SystemExit(f"не удалось сохранить {kind}: {dst}: {e.strerror or e}")
    except BaseException:
        for p in placed:
            p.unlink(missing_ok=True)
        if moved and moved[0].exists():  # вернуть ролик в рабочую папку: повтор не скачает его заново
            try:
                shutil.move(*moved)
            except OSError:
                pass
        raise
    finally:
        for tmp in tmps:
            tmp.unlink(missing_ok=True)


def run(a, work: Path) -> int:
    asr_model = models.resolve_asr(a.model)  # неверный --model падает до чтения файла и сети
    url = sources.is_url(a.source)
    for on, ext in ((a.mp3, ".mp3"), (a.video, ".mp4")):
        if on and a.out and a.out.suffix.lower() == ext:
            raise SystemExit(f"файл текста и сохраняемый файл совпадают: {a.out}; укажите итог с расширением .md")
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
    jobs = []
    if a.mp3:
        mp3 = out.with_suffix(".mp3")
        if not url and _same(src, mp3):
            print(f"звук уже в mp3: {mp3}", flush=True)
        else:
            jobs.append(("mp3", src, mp3))
    if a.video:  # после mp3: он делается из этого же файла
        jobs.append(("видео", src, out.with_suffix(".mp4")))
    _save(jobs)
    for _, _, p in jobs:
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
