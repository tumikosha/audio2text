import subprocess
from pathlib import Path

import numpy as np
import pytest

from audio2text.media import SR, ffmpeg_exe, load_audio


def test_load_audio_reads_wav(tmp_path: Path):
    p = tmp_path / "t.wav"
    subprocess.run([ffmpeg_exe(), "-v", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=1", "-ar", "44100", str(p)], check=True)
    a = load_audio(p)
    assert SR == 16000
    assert a.dtype == np.float32
    assert len(a) == 16000


def test_load_audio_not_media(tmp_path: Path):
    p = tmp_path / "x.pdf"
    p.write_text("это не аудио")
    with pytest.raises(SystemExit) as e:
        load_audio(p)
    assert str(e.value).startswith("не удалось прочитать аудио")


def test_load_audio_ffmpeg_cannot_start(tmp_path: Path, monkeypatch):
    from audio2text import media

    p = tmp_path / "t.wav"
    p.write_bytes(b"x")
    monkeypatch.setattr(media, "ffmpeg_exe", lambda: str(tmp_path / "нет-ffmpeg"))
    with pytest.raises(SystemExit) as e:
        load_audio(p)
    assert str(e.value).startswith("не удалось прочитать аудио")
