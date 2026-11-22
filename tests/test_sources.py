import json
import subprocess
from pathlib import Path

import pytest

from audio2text import sources


def _which(*present):
    return lambda name: f"/bin/{name}" if name in present else None


def test_caption_tracks_original_language_only_best_first():
    auto = {"en-orig": [1], "ru-orig": [1], "en": [1], "de": [1]}
    assert sources.caption_tracks({"language": "ru", "automatic_captions": auto}) == [("ru-orig", "auto")]
    assert sources.caption_tracks({"language": "ru", "subtitles": {"ru": [1], "en": [1]},
                                   "automatic_captions": auto}) == [("ru", "subtitles"), ("ru-orig", "auto")]
    assert sources.caption_tracks({"language": "ro", "automatic_captions": {"ru": [1], "en": [1]}}) == []
    assert sources.caption_tracks({"language": None, "automatic_captions": {"fr-orig": [1], "en": [1]}}) \
        == [("fr-orig", "auto")]


def test_empty_track_skipped(tmp_path, monkeypatch):
    def fake_run(args):
        d = Path(args[args.index("-o") + 1]).parent
        (d / "sub.ru.json3").write_text('{"wireMagic": "pb3", "pens": [{}]}')
        return ""
    monkeypatch.setattr(sources, "_run", fake_run)
    assert sources.captions_doc("u", {}, "ru", "subtitles", tmp_path) is None


def test_captions_doc_shape(tmp_path, monkeypatch, capsys):
    def fake_run(args):
        d = Path(args[args.index("-o") + 1]).parent
        (d / "sub.ru.json3").write_text(json.dumps(
            {"events": [{"tStartMs": 1000, "dDurationMs": 500, "segs": [{"utf8": "Привет"}]}]}))
        return ""
    monkeypatch.setattr(sources, "_run", fake_run)
    doc = sources.captions_doc("u", {"title": "T", "duration": 5}, "ru-orig", "auto", tmp_path)
    assert doc == {"source": "u", "title": "T", "transcript": "субтитры YouTube ru-orig (auto)", "audio_s": 5,
                   "speakers": {}, "segments": [{"s": 1.0, "e": 1.5, "text": "Привет", "turn": False}]}
    assert "субтитры YouTube ru-orig (auto), распознавание не запускалось" in capsys.readouterr().out


def test_json3_parsed_with_turns():
    data = {"events": [{"tStartMs": 0, "dDurationMs": 9, "id": 1},
                       {"tStartMs": 9825, "dDurationMs": 2020, "segs": [{"utf8": "[музыка]"}]},
                       {"tStartMs": 16365, "aAppend": 1, "segs": [{"utf8": "\n"}]},
                       {"tStartMs": 16700, "dDurationMs": 3500,
                        "segs": [{"utf8": ">> Екатерина", "isSpeakerChange": 1}, {"utf8": " Михайловна,"}]}]}
    assert sources.parse_json3(data) == [{"s": 9.82, "e": 11.84, "text": "[музыка]", "turn": False},
                                         {"s": 16.7, "e": 20.2, "text": "Екатерина Михайловна,", "turn": True}]


def test_title_name():
    assert sources.title_name("T: a/b?") == "T a b"
    assert sources.title_name("???") == "transcript"


def test_title_name_windows_reserved():
    for t in ("con", "NUL", "Com1", "lpt1", "prn"):
        assert sources.title_name(t) == t + "_"
    assert sources.title_name("con.txt") == "con_.txt"
    assert sources.title_name("AUX.tar.gz") == "AUX_.tar.gz"
    assert sources.title_name("console") == "console"
    assert sources.title_name("com10") == "com10"


def test_is_url():
    assert sources.is_url("https://youtu.be/x") and not sources.is_url("a.mp3")


def test_ytdlp_default_uses_deno_wheel(monkeypatch):
    monkeypatch.setattr(sources.shutil, "which", _which("uvx"))
    monkeypatch.delenv("AUDIO2TEXT_YTDLP_NO_DENO", raising=False)
    assert sources.ytdlp() == ["uvx", "--quiet", "--with", "deno", "yt-dlp[default]@latest"]


def test_ytdlp_without_uvx(monkeypatch):
    monkeypatch.setattr(sources.shutil, "which", lambda _: None)
    with pytest.raises(SystemExit, match="нужен uv"):
        sources.ytdlp()


def test_ytdlp_js_runtime_fallback(monkeypatch):
    monkeypatch.setattr(sources.shutil, "which", _which("node"))
    assert sources.fallback_cmd()[-2:] == ["--js-runtimes", "node"]
    monkeypatch.setattr(sources.shutil, "which", _which("deno", "node"))
    assert "--js-runtimes" not in sources.fallback_cmd()
    monkeypatch.setattr(sources.shutil, "which", lambda _: None)
    with pytest.raises(SystemExit, match="JS-рантайм"):
        sources.fallback_cmd()


def test_env_switches_to_fallback(monkeypatch):
    monkeypatch.setattr(sources.shutil, "which", _which("uvx", "node"))
    monkeypatch.setenv("AUDIO2TEXT_YTDLP_NO_DENO", "1")
    cmd = sources.ytdlp()
    assert "--with" not in cmd and cmd[-2:] == ["--js-runtimes", "node"]


def test_ytdlp_error_last_line(monkeypatch):
    monkeypatch.setattr(sources.shutil, "which", _which("uvx"))
    monkeypatch.setattr(sources.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 1, "", "WARNING: x\nERROR: Private video\n\n"))
    with pytest.raises(SystemExit) as e:
        sources._run(["-J", "u"])
    assert str(e.value) == "yt-dlp: ERROR: Private video"


def test_info_no_playlist(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(sources, "_run", lambda args: seen.append(args) or '{"id": "x"}')
    assert sources.youtube_info("https://youtu.be/x", tmp_path) == {"id": "x"}
    assert "--no-playlist" in seen[0]
    sources.youtube_info("https://youtu.be/x", tmp_path)  # cached
    assert len(seen) == 1


def test_download_passes_ffmpeg_and_no_playlist(tmp_path, monkeypatch):
    seen = []

    def fake(args):
        seen.append(args)
        Path(args[args.index("-o") + 1].replace("%(ext)s", "m4a")).write_bytes(b"x")
        return ""
    monkeypatch.setattr(sources, "_run", fake)
    monkeypatch.setattr(sources, "ffmpeg_exe", lambda: "/f/ffmpeg")
    p = sources.download("u", tmp_path)
    assert p.name == "src.m4a" and "--no-playlist" in seen[0]
    assert seen[0][seen[0].index("--ffmpeg-location") + 1] == "/f/ffmpeg"
    assert sources.download("u", tmp_path) == p and len(seen) == 1


def test_download_ignores_unfinished_fragments(tmp_path, monkeypatch):
    d = tmp_path / "download"
    d.mkdir()
    for n in ("src.webm.part-Frag3", "src.webm.part", "src.webm.ytdl", "src.webm.temp"):
        (d / n).write_bytes(b"x")
    seen = []

    def fake(args):
        seen.append(args)
        (d / "src.webm").write_bytes(b"x")
        return ""
    monkeypatch.setattr(sources, "_run", fake)
    monkeypatch.setattr(sources, "ffmpeg_exe", lambda: "/f/ffmpeg")
    assert sources.download("u", tmp_path).name == "src.webm"
    assert len(seen) == 1


def test_run_forces_utf8(monkeypatch):
    monkeypatch.setattr(sources.shutil, "which", _which("uvx"))
    seen = {}

    def fake(*a, **k):
        seen.update(k)
        return subprocess.CompletedProcess(a[0], 0, "ok", "")
    monkeypatch.setattr(sources.subprocess, "run", fake)
    assert sources._run(["-J"]) == "ok"
    assert seen["encoding"] == "utf-8" and seen["errors"] == "replace"
    assert seen["env"]["PYTHONUTF8"] == "1" and seen["env"]["PYTHONIOENCODING"] == "utf-8"


def test_info_rejects_playlist_and_cleans_cache(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(sources, "_run", lambda args: seen.append(args) or '{"_type": "playlist", "entries": []}')
    with pytest.raises(SystemExit, match="плейлист или канал"):
        sources.youtube_info("https://youtube.com/playlist?list=L", tmp_path)
    assert not (tmp_path / "info.json").exists()
    assert "--playlist-items" in seen[0] and "--no-playlist" in seen[0]


def test_corrupt_json3_is_empty_track(tmp_path, monkeypatch):
    def fake_run(args):
        (Path(args[args.index("-o") + 1]).parent / "sub.ru.json3").write_text("{oops")
        return ""
    monkeypatch.setattr(sources, "_run", fake_run)
    assert sources.captions_doc("u", {}, "ru", "subtitles", tmp_path) is None
