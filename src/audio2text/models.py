"""Загрузка и кэш моделей (sha256, докачка, атомарная запись)."""

import hashlib
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import platformdirs

TIMEOUT = 30
RETRIES = 3
RETRY_DELAY = 1.0
_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


@dataclass(frozen=True)
class Model:
    name: str
    url: str
    sha256: str
    size: int
    files: tuple[str, ...]
    archive: bool


ASR = Model(
    name="parakeet-tdt-0.6b-v3-int8",
    url=f"{_BASE}/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
    sha256="1495e91b59d7fc53ddca91f3431758ec99f8fa253fdb0e1e851be0f72b14b0f9",
    size=487211443,
    files=("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"),
    archive=True,
)

VAD = Model(
    name="silero-vad",
    url=f"{_BASE}/asr-models/silero_vad.onnx",
    sha256="9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
    size=643854,
    files=("silero_vad.onnx",),
    archive=False,
)

SPEAKER = Model(
    name="nemo-titanet-small",
    url=f"{_BASE}/speaker-recongition-models/nemo_en_titanet_small.onnx",
    sha256="ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e",
    size=40257283,
    files=("nemo_en_titanet_small.onnx",),
    archive=False,
)


def cache_root() -> Path:
    """Корень кэша (модели и рабочие папки); AUDIO2TEXT_CACHE переопределяет папку пользователя."""
    return Path(os.environ.get("AUDIO2TEXT_CACHE") or platformdirs.user_cache_dir("audio2text"))


def cache_dir() -> Path:
    return cache_root() / "models"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _progress(done: int, total: int) -> None:
    sys.stderr.write(f"\r  {done / 1e6:.0f} МБ из {total / 1e6:.0f} МБ")
    sys.stderr.flush()


def _fetch(url: str, part: Path, total: int) -> None:
    """Качает url в part, докачивая с позиции, где остановились."""
    last: Exception | None = None
    for attempt in range(RETRIES):
        if attempt:
            time.sleep(RETRY_DELAY * attempt)
        have = part.stat().st_size if part.exists() else 0
        if total and have == total:
            return
        if total and have > total:
            part.unlink()
            have = 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                if have and resp.status != 206:
                    have = 0  # сервер не умеет Range: начинаем заново
                with part.open("ab" if have else "wb") as f:
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        _progress(have, total)
            sys.stderr.write("\n")
            if total and have < total:
                raise OSError("соединение оборвано")
            return
        except urllib.error.HTTPError as e:
            if e.code == 416 and part.exists():
                part.unlink()  # битый .part: начнём сначала
            last = e
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last = e
    reason = getattr(last, "reason", last)
    if str(last) == "соединение оборвано":
        raise SystemExit(
            f"загрузка прервана: {url}: соединение оборвано; "
            "повторите запуск, докачка продолжится")
    if isinstance(last, urllib.error.HTTPError):
        raise SystemExit(f"не удалось скачать модель: {url}: HTTP {last.code}")
    raise SystemExit(f"нет сети или сервер недоступен: {url}: {reason}")


def _extract(archive: Path, dest: Path, files: tuple[str, ...]) -> None:
    """Достаёт files из архива прямо в dest, отбрасывая верхнюю папку."""
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        with tarfile.open(archive) as tar:
            for member in tar:
                parts = member.name.split("/", 1)
                rel = parts[1] if len(parts) == 2 else parts[0]
                if member.isfile() and rel in files:
                    member.name = rel
                    tar.extract(member, tmp, filter="data")
        missing = [f for f in files if not (tmp / f).is_file()]
        if missing:
            raise SystemExit(f"в архиве модели нет файлов: {', '.join(missing)}")
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            os.replace(tmp / f, dest / f)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def ensure(model: Model, cache: Path | None = None, log=print) -> Path:
    """Папка с файлами модели; качает, если их нет."""
    folder = (cache or cache_dir()) / model.name
    if all((folder / f).is_file() for f in model.files):
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    log(f"скачиваю модели: {model.size / 1e9:.1f} ГБ, один раз")
    fname = Path(urlparse(model.url).path).name
    final = folder / fname
    part = folder / (fname + ".part")
    _fetch(model.url, part, model.size)
    if _sha256(part) != model.sha256:
        part.unlink(missing_ok=True)
        raise SystemExit(
            f"контрольная сумма модели не совпала: {model.url}; повторите запуск")
    if model.archive:
        try:
            _extract(part, folder, model.files)
        finally:
            part.unlink(missing_ok=True)
    else:
        os.replace(part, final)
        if fname not in model.files:
            raise SystemExit(f"неверное описание модели: {model.name}")
    return folder
