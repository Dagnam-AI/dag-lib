"""``install()`` into a directory that cannot be used: the recorder that works is never replaced."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import threading
from typing import Any, Callable, Dict

from _api_support import finish_lines, record
import pytest

import dagnam_record
from dagnam_record import _state

pytestmark = pytest.mark.usefixtures("fresh_state")
POSIX = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes and symlinks")


def writer_threads() -> int:
    return sum(t.name == "dagnam-record" and t.is_alive() for t in threading.enumerate())


def regular_file(tmp_path: Path) -> Path:
    path = tmp_path / "afile"
    path.write_text("x", encoding="utf-8")
    return path


def symlinked(tmp_path: Path) -> Path:
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    return tmp_path / "link"


def read_only(tmp_path: Path) -> Path:
    path = tmp_path / "ro"
    path.mkdir()
    path.chmod(0o500)
    return path


BAD: Dict[str, Callable[[Path], Path]] = {
    "a regular file": regular_file,
    "a symlinked directory": symlinked,
    "a read-only directory": read_only,
}
needs_permissions = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can write anywhere"
)


@POSIX
@needs_permissions
@pytest.mark.parametrize("kind", list(BAD))
def test_reconfiguring_into_an_unusable_directory_keeps_the_running_recorder(
    kind: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    good = tmp_path / "good"
    before = writer_threads()
    assert dagnam_record.install(dir=str(good)) is True
    record(request={"prompt": "one"})
    running = _state.STATE.installed
    bad = BAD[kind](tmp_path)
    try:
        with caplog.at_level(logging.WARNING, logger="dagnam_record"):
            assert dagnam_record.install(dir=str(bad), sample=0.5) is True
        assert _state.STATE.installed is running
        assert caplog.text.count("cannot be used") == 1
        assert "the recorder that is running is kept" in caplog.text
        assert writer_threads() == before + 1  # the refused writer was stopped, not leaked
        record(request={"prompt": "two"})
        lines = finish_lines(good)
        assert [line["request"]["prompt"] for line in lines] == ["one", "two"]
    finally:
        bad.chmod(0o700)


@POSIX
def test_a_refused_reconfiguration_returns_what_the_running_install_returned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("openai", "anthropic"):
        for module in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            monkeypatch.setitem(sys.modules, module, None)
        monkeypatch.setitem(sys.modules, name, None)
    assert dagnam_record.install(dir=str(tmp_path / "good")) is False  # no SDK: nothing patched
    running = _state.STATE.installed
    assert dagnam_record.install(dir=str(regular_file(tmp_path))) is False
    assert _state.STATE.installed is running


@POSIX
@needs_permissions
@pytest.mark.parametrize("kind", list(BAD))
def test_a_first_install_into_an_unusable_directory_stays_active_counts_drops_and_warns_once(
    kind: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bad = BAD[kind](tmp_path)
    try:
        with caplog.at_level(logging.WARNING, logger="dagnam_record"):
            assert dagnam_record.install(dir=str(bad)) is True
            for _ in range(3):
                record()
        assert caplog.text.count("cannot be used") == 1
        assert "dropped and counted" in caplog.text
        installed: Any = _state.STATE.installed
        assert installed is not None
        assert dagnam_record.is_installed()
        assert installed.writer.usable is False
        dagnam_record.uninstall()
        assert installed.writer.counts["dropped_error"] >= 3
    finally:
        bad.chmod(0o700)


def test_a_usable_directory_is_reported_usable(tmp_path: Path) -> None:
    dagnam_record.install(dir=str(tmp_path / "good"))
    installed: Any = _state.STATE.installed
    assert installed.writer.usable is True
