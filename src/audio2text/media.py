"""Работа с ffmpeg и декодирование аудио."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SR = 16000


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        found = shutil.which("ffmpeg")
        if found:
            return found
        raise SystemExit("ffmpeg не найден: установите ffmpeg или пакет imageio-ffmpeg")


def load_audio(path: Path) -> np.ndarray:
    """Файл любого формата -> float32 mono 16 кГц. Выход ffmpeg читается кусками в один растущий буфер,
    массив смотрит в него без копии: в памяти одна копия сэмплов (3 ч ~ 0.7 ГБ)."""
    cmd = [ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    with tempfile.TemporaryFile() as err:  # stderr в файл: второй канал мог бы заполниться и встать
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        except OSError as e:
            raise SystemExit(f"не удалось прочитать аудио: {path}: {e.strerror or e}")
        buf = bytearray()
        with proc:
            while chunk := proc.stdout.read(1 << 20):
                buf += chunk
        del buf[len(buf) // 4 * 4:]
        if proc.returncode != 0 or not buf:
            err.seek(0)
            lines = err.read().decode("utf-8", "replace").strip().splitlines()
            last = lines[-1] if lines else "нет аудиодорожки"
            raise SystemExit(f"не удалось прочитать аудио: {path}: {last}")
    return np.frombuffer(buf, dtype=np.float32)
