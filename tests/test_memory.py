import subprocess
import tracemalloc
from pathlib import Path

from audio2text.media import ffmpeg_exe, load_audio


def test_load_audio_peak_close_to_one_array(tmp_path: Path):
    """Decoding keeps about one copy of the samples: no bytes + slice + copy (3x on long recordings)."""
    p = tmp_path / "t.flac"
    subprocess.run([ffmpeg_exe(), "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=300",
                    "-ar", "16000", "-ac", "1", str(p)], check=True)
    tracemalloc.start()
    try:
        a = load_audio(p)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(a) == 300 * 16000
    assert peak <= 1.3 * a.nbytes, f"peak {peak / 1e6:.0f} MB for a {a.nbytes / 1e6:.0f} MB array"
    a[0] = 0.5  # writable, like before
