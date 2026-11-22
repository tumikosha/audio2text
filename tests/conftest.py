import os
import urllib.request

import pytest

from audio2text import models

WAV_URL = ("https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
           "/resolve/main/test_wavs/en.wav")


def fetch_cached(request, name: str, url: str):
    """Public test file in the pytest cache; written atomically (tmp + replace)."""
    path = request.config.cache.mkdir("audio2text") / name
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        tmp = path.with_name(f"{name}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    return path


@pytest.fixture(scope="session")
def vad_dir():
    return models.ensure(models.VAD)


@pytest.fixture(scope="session")
def asr_dir():
    return models.ensure(models.ASR)


@pytest.fixture(scope="session")
def sample_wav(request):
    return fetch_cached(request, "en.wav", WAV_URL)
