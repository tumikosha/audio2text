from pathlib import Path

import pytest

from audio2text import cli, media, pipeline, sources

URL = "https://www.youtube.com/watch?v=abc"
INFO = {"title": "Лекция: тема/1", "duration": 75}


@pytest.fixture(autouse=True)
def iso(tmp_path, monkeypatch):
    """Work folders under tmp_path, outputs in a separate cwd: nothing touches the user cache or the repo."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(cli, "work_root", lambda: tmp_path / "work")
    return cwd


def _doc(source):
    return {"source": str(source), "transcript": "локально, parakeet-tdt-0.6b-v3", "audio_s": 75.0,
            "speakers": {"SPEAKER_00": {"speech_s": 1.0, "embedding": []}},
            "segments": [{"s": 1.0, "e": 2.0, "text": "Привет.", "speaker": "SPEAKER_00"}]}


def _fake_url(monkeypatch, calls, tracks=(("ru", "subtitles"),)):
    monkeypatch.setattr(sources, "youtube_info", lambda url, work: INFO)
    monkeypatch.setattr(sources, "caption_tracks", lambda info: list(tracks))

    def captions_doc(url, info, track, kind, work):
        calls.append("captions")
        return {"source": url, "title": info["title"], "transcript": f"субтитры YouTube {track} ({kind})",
                "audio_s": 75, "speakers": {}, "segments": [{"s": 0.0, "e": 1.0, "text": "Здравствуйте.",
                                                             "turn": True}]}

    def download(url, work):
        calls.append("download")
        work.mkdir(parents=True, exist_ok=True)
        p = work / "download" / "src.webm"
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"x")
        return p

    def download_video(url, work):
        calls.append("download_video")
        p = work / "video" / "src.mp4"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"video")
        return p

    def transcribe_file(path, log=print, model=None):
        calls.append(("transcribe", Path(path).name))
        return _doc(path)

    monkeypatch.setattr(sources, "captions_doc", captions_doc)
    monkeypatch.setattr(sources, "download", download)
    monkeypatch.setattr(sources, "download_video", download_video)
    monkeypatch.setattr(pipeline, "transcribe_file", transcribe_file)


def test_cli_missing_file(capsys):
    assert cli.main(["нет такого.mp3"]) == 2
    assert "файл не найден" in capsys.readouterr().err


def test_cli_unicode_path(tmp_path, monkeypatch, iso):
    src = tmp_path / "Записи" / "встреча 1.m4a"
    src.parent.mkdir()
    src.write_bytes(b"x")
    monkeypatch.setattr(pipeline, "transcribe_file", lambda path, log=print, model=None: _doc(path))
    assert cli.main([str(src)]) == 0
    out = iso / "встреча 1.md"
    text = out.read_bytes().decode("utf-8")
    assert text.startswith("# встреча 1\n")
    assert "**SPEAKER_00** `00:00:01` Привет." in text
    assert not (tmp_path / "work").exists() or not any((tmp_path / "work").iterdir())


def test_cli_explicit_out(tmp_path, monkeypatch, iso):
    src = tmp_path / "a.wav"
    src.write_bytes(b"x")
    monkeypatch.setattr(pipeline, "transcribe_file", lambda path, log=print, model=None: _doc(path))
    assert cli.main([str(src), str(tmp_path / "итог" / "x.txt")]) == 0
    assert (tmp_path / "итог" / "x.txt").read_text(encoding="utf-8").startswith("# a\n")


def test_cli_url_captions_default(monkeypatch, iso):
    calls = []
    _fake_url(monkeypatch, calls)
    assert cli.main([URL]) == 0
    assert calls == ["captions"]
    text = (iso / "Лекция тема 1.md").read_text(encoding="utf-8")
    assert text.startswith("# Лекция: тема/1\n")
    assert "— `00:00:00` Здравствуйте." in text


def test_cli_url_transcribe_flag(monkeypatch, iso):
    calls = []
    _fake_url(monkeypatch, calls)
    assert cli.main([URL, "-tr"]) == 0
    assert calls == ["download", ("transcribe", "src.webm")]
    text = (iso / "Лекция тема 1.md").read_text(encoding="utf-8")
    assert text.startswith("# Лекция: тема/1\n")
    assert f"- Источник: {URL}\n" in text
    assert "- Транскрипт: локально, parakeet-tdt-0.6b-v3\n" in text


def test_cli_url_no_captions_falls_back(monkeypatch, iso):
    calls = []
    _fake_url(monkeypatch, calls, tracks=())
    assert cli.main([URL]) == 0
    assert calls == ["download", ("transcribe", "src.webm")]


def test_cli_unexpected_error_one_line(tmp_path, monkeypatch, capsys):
    src = tmp_path / "a.wav"
    src.write_bytes(b"x")

    def boom(path, log=print, model=None):
        raise RuntimeError("x\ny")

    monkeypatch.setattr(pipeline, "transcribe_file", boom)
    assert cli.main([str(src)]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1
    assert err[0].startswith("ошибка: RuntimeError: x y (подробности: ")
    log = err[0].rsplit("(подробности: ", 1)[1].rstrip(")")
    assert log.endswith("error.log")
    assert "RuntimeError" in open(log, encoding="utf-8").read()


def test_cli_error_log_unwritable(tmp_path, monkeypatch, capsys):
    src = tmp_path / "a.wav"
    src.write_bytes(b"x")
    blocker = tmp_path / "file-not-dir"
    blocker.write_text("")
    monkeypatch.setattr(cli, "work_root", lambda: blocker / "work")  # mkdir under a file fails

    def boom(path, log=print, model=None):
        raise RuntimeError("x")

    monkeypatch.setattr(pipeline, "transcribe_file", boom)
    assert cli.main([str(src)]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert err == ["ошибка: RuntimeError: x"]


def test_cli_not_media(tmp_path, capsys):
    """A text file fails at decoding, before any model download."""
    src = tmp_path / "заметки.txt"
    src.write_text("не аудио", encoding="utf-8")
    assert cli.main([str(src)]) == 2
    assert "не удалось прочитать аудио" in capsys.readouterr().err


def test_cli_help_exit_zero():
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0


@pytest.mark.models
def test_end_to_end_public_sample(sample_wav, iso):
    assert cli.main([str(sample_wav)]) == 0
    text = (iso / "en.md").read_text(encoding="utf-8")
    assert "- Транскрипт: локально, parakeet-tdt-0.6b-v3\n" in text
    assert "**SPEAKER_00**" in text


def test_cli_unknown_model_fails_before_work(tmp_path, monkeypatch, capsys):
    def touched(*a, **k):
        raise AssertionError("no work before the model is resolved")

    monkeypatch.setattr(pipeline, "transcribe_file", touched)
    monkeypatch.setattr(sources, "youtube_info", touched)
    src = tmp_path / "a.wav"
    src.write_bytes(b"x")
    for source in (str(src), URL):
        assert cli.main([source, "--model", "nope"]) == 2
        err = capsys.readouterr().err.strip().splitlines()
        assert err == ["модель не найдена: nope; доступные: parakeet-tdt-0.6b-v3, parakeet-tdt-0.6b-v2; "
                       "или путь к папке с моделью"]


def test_cli_model_passed_to_pipeline(tmp_path, monkeypatch, iso):
    src = tmp_path / "a.wav"
    src.write_bytes(b"x")
    got = []
    monkeypatch.setattr(pipeline, "transcribe_file", lambda path, log=print, model=None: got.append(model) or
                        _doc(path))
    assert cli.main([str(src), "--model", "parakeet-tdt-0.6b-v2"]) == 0
    assert got[0].label == "parakeet-tdt-0.6b-v2" and got[0].model.name == "parakeet-tdt-0.6b-v2-int8"


def _fake_mp3(monkeypatch, calls):
    def to_mp3(src, dst):
        assert src.exists()  # made before the video is moved away
        calls.append(("mp3", src.name))
        dst.write_bytes(b"mp3 of " + src.read_bytes())
    monkeypatch.setattr(media, "to_mp3", to_mp3)


def _touched(*a, **k):
    raise AssertionError("no work expected")


def test_cli_video_needs_url(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "transcribe_file", _touched)
    src = tmp_path / "a.wav"
    src.write_bytes(b"x")
    assert cli.main([str(src), "--video"]) == 2
    assert capsys.readouterr().err.strip() == "--video работает только для ссылки"


def test_cli_existing_media_fails_before_work_file(tmp_path, monkeypatch, capsys, iso):
    monkeypatch.setattr(pipeline, "transcribe_file", _touched)
    monkeypatch.setattr(media, "to_mp3", _touched)
    src = tmp_path / "a.wav"
    src.write_bytes(b"x")
    (iso / "a.mp3").write_bytes(b"old")
    assert cli.main([str(src), "--mp3"]) == 2
    assert capsys.readouterr().err.strip() == "файл уже есть: a.mp3; удалите его или укажите другой выход"
    assert (iso / "a.mp3").read_bytes() == b"old" and not (iso / "a.md").exists()


def test_cli_existing_media_fails_after_info_before_download(monkeypatch, capsys, iso):
    calls = []
    _fake_url(monkeypatch, calls)
    for name in ("captions_doc", "download", "download_video"):
        monkeypatch.setattr(sources, name, _touched)
    monkeypatch.setattr(pipeline, "transcribe_file", _touched)
    for flag, ext in (("--video", ".mp4"), ("--mp3", ".mp3")):
        p = iso / f"Лекция тема 1{ext}"
        p.write_bytes(b"old")
        assert cli.main([URL, flag]) == 2
        assert capsys.readouterr().err.strip() == f"файл уже есть: {p.name}; удалите его или укажите другой выход"
        assert p.read_bytes() == b"old"
        p.unlink()
    assert not (iso / "Лекция тема 1.md").exists()


def test_cli_url_video_and_mp3_from_one_download(monkeypatch, capsys, iso):
    calls = []
    _fake_url(monkeypatch, calls)
    _fake_mp3(monkeypatch, calls)
    assert cli.main([URL, "-tr", "--video", "--mp3"]) == 0
    assert calls == ["download_video", ("transcribe", "src.mp4"), ("mp3", "src.mp4")]
    assert (iso / "Лекция тема 1.mp4").read_bytes() == b"video"
    assert (iso / "Лекция тема 1.mp3").read_bytes() == b"mp3 of video"
    assert sorted(p.name for p in iso.iterdir()) == ["Лекция тема 1.md", "Лекция тема 1.mp3", "Лекция тема 1.mp4"]
    out = capsys.readouterr().out
    assert "сохранено: Лекция тема 1.mp3\n" in out and "сохранено: Лекция тема 1.mp4\n" in out


def test_cli_url_captions_with_mp3(monkeypatch, iso):
    calls = []
    _fake_url(monkeypatch, calls)
    _fake_mp3(monkeypatch, calls)
    assert cli.main([URL, "--mp3"]) == 0
    assert calls == ["captions", "download", ("mp3", "src.webm")]
    assert "— `00:00:00` Здравствуйте." in (iso / "Лекция тема 1.md").read_text(encoding="utf-8")
    assert (iso / "Лекция тема 1.mp3").exists() and not (iso / "Лекция тема 1.mp4").exists()


def test_cli_url_captions_with_video(monkeypatch, iso):
    calls = []
    _fake_url(monkeypatch, calls)
    _fake_mp3(monkeypatch, calls)
    assert cli.main([URL, "--video"]) == 0
    assert calls == ["captions", "download_video"]
    assert (iso / "Лекция тема 1.mp4").read_bytes() == b"video"


def test_cli_file_mp3_is_source(tmp_path, monkeypatch, capsys):
    d = tmp_path / "rec"
    d.mkdir()
    src = d / "запись.mp3"
    src.write_bytes(b"orig")
    monkeypatch.setattr(pipeline, "transcribe_file", lambda path, log=print, model=None: _doc(path))
    monkeypatch.setattr(media, "to_mp3", _touched)
    assert cli.main([str(src), str(d / "запись.md"), "--mp3"]) == 0
    assert src.read_bytes() == b"orig"
    out = capsys.readouterr().out
    assert f"звук уже в mp3: {d / 'запись.mp3'}" in out and "сохранено" not in out


def test_cli_file_mp3(tmp_path, monkeypatch, iso):
    calls = []
    src = tmp_path / "встреча.m4a"
    src.write_bytes(b"m4a")
    monkeypatch.setattr(pipeline, "transcribe_file", lambda path, log=print, model=None: _doc(path))
    _fake_mp3(monkeypatch, calls)
    assert cli.main([str(src), "--mp3"]) == 0
    assert (iso / "встреча.mp3").read_bytes() == b"mp3 of m4a" and src.read_bytes() == b"m4a"
