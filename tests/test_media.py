import subprocess
from pathlib import Path

import numpy as np
import pytest

from audio2text.media import SR, ffmpeg_exe, load_audio, to_mp3


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
    assert str(e.value).startswith("cannot read audio")


def test_load_audio_ffmpeg_cannot_start(tmp_path: Path, monkeypatch):
    from audio2text import media

    p = tmp_path / "t.wav"
    p.write_bytes(b"x")
    monkeypatch.setattr(media, "ffmpeg_exe", lambda: str(tmp_path / "нет-ffmpeg"))
    with pytest.raises(SystemExit) as e:
        load_audio(p)
    assert str(e.value).startswith("cannot read audio")


def test_to_mp3_real_ffmpeg(tone_wav, tmp_path: Path):
    dst = tmp_path / "итог" / "звук.mp3"
    to_mp3(tone_wav, dst)
    assert abs(len(load_audio(dst)) / SR - 2.0) <= 0.1
    assert [p.name for p in dst.parent.iterdir()] == ["звук.mp3"]  # temp file gone


def test_to_mp3_error_one_line_and_no_temp(tmp_path: Path):
    src = tmp_path / "x.pdf"
    src.write_text("это не аудио")
    dst = tmp_path / "out.mp3"
    with pytest.raises(SystemExit) as e:
        to_mp3(src, dst)
    msg = str(e.value)
    assert msg.startswith("cannot save mp3: ") and "\n" not in msg
    assert list(tmp_path.iterdir()) == [src]


def test_ffmpeg_has_libmp3lame():
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-encoders"], capture_output=True, text=True, check=True)
    assert "libmp3lame" in r.stdout


def test_load_audio_removes_dc_offset(tone_wav, tmp_path: Path):
    p = tmp_path / "dc.wav"
    subprocess.run([ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(tone_wav),
                    "-af", "volume=0.5,aeval=val(0)+0.05", "-ar", "44100", str(p)], check=True)
    raw = subprocess.run([ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(p), "-f", "f32le", "-ac", "1",
                          "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    assert abs(np.frombuffer(raw, dtype=np.float32).mean()) > 0.04  # смещение в файле есть
    assert abs(load_audio(p).mean()) < 1e-3


@pytest.mark.models
def test_vad_same_chunks_with_dc_offset(sample_wav, vad_dir, tmp_path: Path):
    from audio2text.vad import vad_chunks

    p = tmp_path / "dc.wav"
    subprocess.run([ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(sample_wav),
                    "-af", "aeval=val(0)+0.05", "-ar", "16000", str(p)], check=True)
    base = vad_chunks(load_audio(sample_wav), vad_dir)
    assert base and len(vad_chunks(load_audio(p), vad_dir)) == len(base)
