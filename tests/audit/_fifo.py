"""A named pipe for tests, with a writer that comes along if a read blocks on it."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import threading
import time
from typing import Any

import pytest


def make_fifo(path: Path, *, release_after: float = 5.0) -> Path:
    """Create a FIFO at ``path``; a reader that blocks on it is let go after ``release_after`` s.

    Opening a FIFO for reading blocks until someone opens it for writing. A scan that wrongly
    reads one would hang the whole suite, so a daemon thread opens (and closes) the write end
    after a delay: a correct scan has refused long before, a broken one fails instead of hanging.
    """
    os.mkfifo(path)

    def release() -> None:
        deadline = time.monotonic() + release_after
        while time.monotonic() < deadline:
            time.sleep(0.05)
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_NONBLOCK))
        except OSError:
            pass  # nobody is reading (the usual case): nothing is blocked

    threading.Thread(target=release, daemon=True).start()
    return path


def refuse_a_blocking_open_of_a_pipe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail at once on any ``os.open`` of a FIFO that could block (no ``O_NONBLOCK``).

    The writer thread of :func:`make_fifo` lets a blocked reader go after a few seconds, so a
    reader that wrongly blocks would still end in a refusal, only late. This makes the
    mistake itself the failure, whatever the clock says.
    """
    real_open = os.open

    def guarded(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if kwargs.get("dir_fd") is None and not flags & os.O_NONBLOCK:
            try:
                is_pipe = stat.S_ISFIFO(os.stat(path).st_mode)
            except OSError:
                is_pipe = False  # not there yet: a file about to be created
            if is_pipe:
                raise AssertionError(f"{path} is a pipe and was opened in a way that can block")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded)
