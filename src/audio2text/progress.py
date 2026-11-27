"""Spinner in stderr for long steps that have no progress bar: "| loading models 3s".

Like the tqdm bars it shows only in a terminal: piped or redirected output (CI, tests) gets nothing. The line is
erased when the step ends, so the next message or bar starts on a clean line."""

from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager

FRAMES = "|/-\\"
TICK_S = 0.1


@contextmanager
def spinner(msg: str, stream=None):
    stream = stream or sys.stderr
    if not (hasattr(stream, "isatty") and stream.isatty()):
        yield
        return
    stop = threading.Event()
    t0 = time.monotonic()
    width = 0

    def show(line: str) -> None:
        nonlocal width
        width = max(width, len(line))
        stream.write(f"\r{line}")
        stream.flush()

    def spin():
        k = 0
        while not stop.wait(TICK_S):
            k += 1
            show(f"{FRAMES[k % len(FRAMES)]} {msg} {time.monotonic() - t0:.0f}s")

    show(f"{FRAMES[0]} {msg}")
    th = threading.Thread(target=spin, daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()
        th.join()
        stream.write("\r" + " " * width + "\r")  # spaces, not ESC[K: old Windows consoles print it raw
        stream.flush()
