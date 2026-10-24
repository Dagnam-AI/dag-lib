"""The recorder's promise to the program it watches: it fails open, whatever goes wrong.

Every case runs a real SDK call against the mock server twice, once with nothing installed and
once with the recorder broken in one way, and the caller must see exactly the same thing: the same
data, or the same exception. Each case also checks the break was really hit (a drop was counted).
"""

from __future__ import annotations

import errno
import os
from pathlib import Path
import shutil
import stat
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, NoReturn, Optional, Tuple

from _guarantee_support import KINDS, flood, run_call
import pytest

import dagnam_record
from dagnam_record import _anthropic_body, _openai_body, _recorder, _state
from dagnam_record._writer import Writer

if TYPE_CHECKING:
    from _mock_server import MockServer

pytestmark = pytest.mark.usefixtures("fresh_state")
POSIX = sys.platform != "win32"
PATIENCE = 20.0
"""Seconds a call may take, with a recorder that is broken or stuck: far above any honest run."""


def boom(*args: Any, **kwargs: Any) -> NoReturn:
    raise RuntimeError("the recorder has a bug")


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Never touch the real ``~/.dagnam``."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    return tmp_path / "home"


GATE = threading.Event()
"""While it is clear, a stuck writer thread waits on it (see ``stick``)."""


@pytest.fixture(autouse=True)
def gate(fresh_state: None) -> Iterator[None]:
    """A clear gate for the test; opened afterwards so no writer thread is left waiting."""
    GATE.clear()
    yield
    GATE.set()


def stick(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every writer thread block on its first line, until ``GATE`` opens."""

    def blocked(self: Writer, line: str) -> None:
        GATE.wait(PATIENCE)

    monkeypatch.setattr(Writer, "_write", blocked)


def installed_writer() -> Writer:
    current = _state.STATE.installed
    assert current is not None
    return current.writer


def break_submit(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    dagnam_record.install(dir=str(directory))
    monkeypatch.setattr(installed_writer(), "submit", boom)


def break_build_line(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    monkeypatch.setattr(_recorder, "build_line", boom)
    dagnam_record.install(dir=str(directory))


def break_serializer(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    monkeypatch.setattr(_openai_body, "to_plain", boom)
    monkeypatch.setattr(_anthropic_body, "to_plain", boom)
    dagnam_record.install(dir=str(directory))


def break_redact(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    dagnam_record.install(dir=str(directory), redact=boom)


def break_disk(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    real = os.write

    def full(descriptor: int, data: Any) -> int:
        if isinstance(data, bytes) and b'"trace_id"' in data:  # the recorder's line, nothing else
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(descriptor, data)

    monkeypatch.setattr(os, "write", full)
    dagnam_record.install(dir=str(directory))


def break_directory_not_a_directory(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.write_text("a file where the directory should be", encoding="utf-8")
    dagnam_record.install(dir=str(directory))


def break_directory_removed(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    dagnam_record.install(dir=str(directory))
    shutil.rmtree(directory, ignore_errors=True)


def break_directory_read_only(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    directory.mkdir(parents=True)
    directory.chmod(0o500)
    dagnam_record.install(dir=str(directory))


def break_queue_full(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    stick(monkeypatch)
    dagnam_record.install(dir=str(directory))
    flood(1200)  # the writer thread is stuck: the queue (1000) fills up


Breaker = Callable[[pytest.MonkeyPatch, Path], None]
BREAKS: Dict[str, Tuple[Breaker, Optional[str]]] = {
    "the writer's submit raises": (break_submit, "dropped_error"),
    "build_line raises": (break_build_line, "dropped_error"),
    "the serializer raises": (break_serializer, "dropped_error"),
    "redact raises": (break_redact, "dropped_error"),
    "the disk is full": (break_disk, "dropped_error"),
    "the directory is a file": (break_directory_not_a_directory, "dropped_error"),
    "the directory is removed": (break_directory_removed, None),
    "the queue is full and the writer is stuck": (break_queue_full, "dropped_full"),
}
if POSIX and hasattr(os, "geteuid") and os.geteuid() != 0:  # root can write anywhere
    BREAKS["the directory is read-only"] = (break_directory_read_only, "dropped_error")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", list(BREAKS))
def test_the_caller_sees_the_same_thing_whatever_breaks_in_the_recorder(
    name: str, kind: str, tmp_path: Path, mock_server: MockServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = run_call(kind, mock_server)
    directory = tmp_path / "traces"
    breaker, counted = BREAKS[name]
    try:
        breaker(monkeypatch, directory)
        began = time.perf_counter()
        seen = run_call(kind, mock_server)
        assert time.perf_counter() - began < PATIENCE
        assert seen == expected
        writer = installed_writer()
        GATE.set()
        dagnam_record.uninstall()  # writes what is queued (the disk-full case fails here)
        assert counted is None or writer.counts[counted] >= 1
    finally:
        if directory.is_dir():
            directory.chmod(0o700)


@pytest.mark.parametrize("kind", ["openai ok", "anthropic ok"])
def test_a_stuck_writer_does_not_slow_an_sdk_call(
    kind: str, tmp_path: Path, mock_server: MockServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = run_call(kind, mock_server)
    stick(monkeypatch)
    dagnam_record.install(dir=str(tmp_path / "traces"))
    flood(1100)
    began = time.perf_counter()
    assert run_call(kind, mock_server) == expected
    assert time.perf_counter() - began < PATIENCE
    assert installed_writer().counts["dropped_full"] > 0


def test_nothing_else_is_raised_for_a_bad_record_call() -> None:
    dagnam_record.install()
    dagnam_record.record_call(provider=None, request=None, response=object())  # type: ignore[arg-type]
    dagnam_record.record_call(provider="other", request={"x": float("nan")}, response={})
    writer = installed_writer()
    dagnam_record.uninstall()
    assert writer.counts["dropped_error"] == 2
    assert writer.counts["written"] == 0


def test_the_writer_is_a_daemon_with_a_stable_name(tmp_path: Path) -> None:
    dagnam_record.install(dir=str(tmp_path / "traces"))
    (thread,) = [t for t in threading.enumerate() if t.name == "dagnam-record"]
    assert thread.daemon is True


def test_the_home_directory_is_untouched_when_a_directory_is_given(
    tmp_path: Path, mock_server: MockServer
) -> None:
    home = Path(os.environ["HOME"])
    dagnam_record.install(dir=str(tmp_path / "traces"))
    run_call("openai ok", mock_server)
    dagnam_record.record_call(provider="other", request={"a": 1}, response={"b": 2})
    dagnam_record.uninstall()
    assert not home.exists()  # the default ~/.dagnam/traces was never made


@pytest.mark.skipif(not POSIX, reason="POSIX file modes")
def test_the_directory_is_private_and_the_files_are_owner_only_even_under_a_loose_umask(
    tmp_path: Path, mock_server: MockServer
) -> None:
    previous = os.umask(0)
    try:
        directory = tmp_path / "traces"
        dagnam_record.install(dir=str(directory))
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700  # from the moment it is made
        run_call("openai ok", mock_server)
        dagnam_record.uninstall()
    finally:
        os.umask(previous)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    files = sorted(directory.iterdir())
    assert [f.suffix for f in files] == [".json", ".jsonl"]  # .status.json, then the day file
    assert {stat.S_IMODE(f.stat().st_mode) for f in files} == {0o600}  # the day file and status
