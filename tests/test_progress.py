import io
import time

from audio2text import progress


class Tty(io.StringIO):
    def isatty(self):
        return True


def test_spinner_silent_when_not_a_terminal():
    s = io.StringIO()
    with progress.spinner("loading models", s):
        pass
    assert s.getvalue() == ""


def test_spinner_shows_message_then_erases_line(monkeypatch):
    monkeypatch.setattr(progress, "TICK_S", 0.01)
    s = Tty()
    with progress.spinner("loading models", s):
        time.sleep(0.05)
    out = s.getvalue()
    assert "| loading models" in out
    assert "loading models 0s" in out  # elapsed seconds after the first tick
    last = out.rsplit("\r", 2)
    assert last[-1] == "" and last[-2].strip() == ""  # final write blanks the line and returns the cursor


def test_spinner_erases_line_on_error():
    s = Tty()
    try:
        with progress.spinner("x", s):
            raise ValueError
    except ValueError:
        pass
    assert s.getvalue().endswith("\r")
