import hashlib
import io
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from audio2text import models
from audio2text.models import Model, ensure


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(models, "RETRY_DELAY", 0)


class Server:
    def __init__(self, data: bytes, truncate: str | None = None):
        self.data = data
        self.truncate = truncate  # None | "first" | "always"
        self.requests: list[str | None] = []  # значения заголовка Range
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                rng = self.headers.get("Range")
                outer.requests.append(rng)
                body, code = outer.data, 200
                if rng:
                    start = int(rng.split("=")[1].rstrip("-"))
                    body, code = outer.data[start:], 206
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if outer.truncate == "always" or (
                        outer.truncate == "first" and len(outer.requests) == 1):
                    body = body[: len(body) // 2]
                    self.close_connection = True
                self.wfile.write(body)

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/m.bin"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def serve():
    servers = []

    def make(data: bytes, truncate: str | None = None) -> Server:
        s = Server(data, truncate)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


def _model(url: str, data: bytes, **kw) -> Model:
    return Model(name="m", url=url, sha256=hashlib.sha256(data).hexdigest(),
                 size=len(data), files=kw.get("files", ("m.bin",)),
                 archive=kw.get("archive", False))


def test_ensure_downloads_and_verifies(tmp_path: Path, serve):
    data = b"x" * 5000
    s = serve(data)
    m = _model(s.url, data)
    logs = []
    folder = ensure(m, tmp_path, log=logs.append)
    assert (folder / "m.bin").read_bytes() == data
    assert logs and "скачиваю модели" in logs[0]
    n = len(s.requests)
    ensure(m, tmp_path, log=logs.append)
    assert len(s.requests) == n


def test_ensure_bad_sha_rejected(tmp_path: Path, serve):
    s = serve(b"other bytes!")
    m = _model(s.url, b"expected")
    m = Model(m.name, m.url, m.sha256, 12, m.files, False)
    with pytest.raises(SystemExit) as e:
        ensure(m, tmp_path, log=lambda *_: None)
    assert "контрольная сумма" in str(e.value)
    assert not (tmp_path / "m" / "m.bin").exists()


def test_ensure_resumes_partial(tmp_path: Path, serve):
    data = bytes(range(256)) * 20
    s = serve(data)
    m = _model(s.url, data)
    half = len(data) // 2
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "m.bin.part").write_bytes(data[:half])
    ensure(m, tmp_path, log=lambda *_: None)
    assert s.requests == [f"bytes={half}-"]
    assert (folder / "m.bin").read_bytes() == data


def test_ensure_offline_message(tmp_path: Path):
    url = "http://127.0.0.1:1/m.bin"
    m = Model("m", url, "0" * 64, 10, ("m.bin",), False)
    with pytest.raises(SystemExit) as e:
        ensure(m, tmp_path, log=lambda *_: None)
    assert "нет сети" in str(e.value) and url in str(e.value)


def test_ensure_archive_extracted(tmp_path: Path, serve):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        for name, body in (("top/a.txt", b"aaa"), ("top/b.onnx", b"bbb"),
                           ("top/extra/c.wav", b"ccc")):
            ti = tarfile.TarInfo(name)
            ti.size = len(body)
            tar.addfile(ti, io.BytesIO(body))
    data = buf.getvalue()
    s = serve(data)
    m = _model(s.url, data, files=("a.txt", "b.onnx"), archive=True)
    folder = ensure(m, tmp_path, log=lambda *_: None)
    assert (folder / "a.txt").read_bytes() == b"aaa"
    assert (folder / "b.onnx").read_bytes() == b"bbb"
    assert not (folder / "m.bin").exists() and not (folder / "m.bin.part").exists()


def test_ensure_resumes_after_truncated_body(tmp_path: Path, serve):
    data = bytes(range(256)) * 400
    s = serve(data, truncate="first")
    m = _model(s.url, data)
    folder = ensure(m, tmp_path, log=lambda *_: None)
    assert (folder / "m.bin").read_bytes() == data
    assert s.requests[0] is None and s.requests[1].startswith("bytes=")


def test_ensure_truncated_always_keeps_part(tmp_path: Path, serve):
    data = bytes(range(256)) * 400
    s = serve(data, truncate="always")
    m = _model(s.url, data)
    with pytest.raises(SystemExit) as e:
        ensure(m, tmp_path, log=lambda *_: None)
    assert "прервана" in str(e.value)
    part = tmp_path / "m" / "m.bin.part"
    assert part.exists() and 0 < part.stat().st_size < len(data)


def test_cache_env_override(tmp_path: Path, monkeypatch):
    from audio2text import cli

    monkeypatch.setenv("AUDIO2TEXT_CACHE", str(tmp_path / "c"))
    assert models.cache_dir() == tmp_path / "c" / "models"
    assert cli.work_root() == tmp_path / "c" / "work"
    monkeypatch.delenv("AUDIO2TEXT_CACHE")
    assert models.cache_dir().name == "models" and "audio2text" in str(models.cache_dir())
