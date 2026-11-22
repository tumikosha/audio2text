"""Работа с ffmpeg и декодирование аудио."""

import shutil
import subprocess
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
    """Файл любого формата -> float32 mono 16 кГц."""
    cmd = [ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    data = proc.stdout[: len(proc.stdout) // 4 * 4]
    if proc.returncode != 0 or not data:
        lines = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        last = lines[-1] if lines else "нет аудиодорожки"
        raise SystemExit(f"не удалось прочитать аудио: {path}: {last}")
    return np.frombuffer(data, dtype=np.float32).copy()
