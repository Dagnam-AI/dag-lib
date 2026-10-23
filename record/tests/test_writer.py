"""The writer: private daily files, whole lines, bounded and fail-open, with saved counts."""

from __future__ import annotations

import atexit
import errno
import json
import logging
import os
from pathlib import Path
import queue
import stat
import sys
import threading
from typing import Any, Callable, Iterator, List, Optional

import pytest

from dagnam_record._writer import Writer

T0 = 1_791_797_400.0  # 2026-10-12T09:30:00Z
DAY = 86_400
POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")


class Clock:
    """A clock the test moves by hand."""

    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        """The current time."""
        return self.now


class Probe(Writer):
    """A writer the tests can look inside, from a subclass, so no test touches a private name."""

    def drain(self) -> None:
        """Write everything queued."""
        self._drain_for_test()

    def loop(self) -> None:
        """Run the thread's loop in the calling thread."""
        self._run()

    def fd(self) -> Optional[int]:
        """The open descriptor, if any."""
        return self._fd

    def alive(self) -> bool:
        """Whether the writer thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def swap_queue(self, replacement: Any) -> None:
        """Use another queue."""
        self._queue = replacement

    def real_write(self) -> Callable[[str], None]:
        """The writer's own write, before a test replaces it."""
        return self._write


class ScriptedQueue:
    """A queue whose ``get`` plays back a script: ``queue.Empty`` means an idle timeout."""

    def __init__(self, script: List[Any]) -> None:
        self.script = script

    def get(self, timeout: Optional[float] = None) -> Optional[str]:
        """The next scripted item."""
        item = self.script.pop(0)
        if item is queue.Empty:
            raise queue.Empty
        return item

    def get_nowait(self) -> Optional[str]:
        """Nothing is left once the script has run."""
        raise queue.Empty


@pytest.fixture
def make(tmp_path: Path) -> Iterator[Callable[..., Probe]]:
    created: List[Probe] = []

    def factory(directory: Optional[Path] = None, **kwargs: Any) -> Probe:
        kwargs.setdefault("start", False)
        kwargs.setdefault("clock", Clock())
        writer = Probe(str(directory or tmp_path / "traces"), **kwargs)
        created.append(writer)
        return writer

    yield factory
    for writer in created:
        writer.close(timeout=5)


def files(directory: Path) -> List[str]:
    return sorted(p.name for p in directory.iterdir() if p.name != ".status.json")


def lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def status(directory: Path) -> Any:
    return json.loads((directory / ".status.json").read_text(encoding="utf-8"))


def fail_writes(
    monkeypatch: pytest.MonkeyPatch, writer: Probe, count: int, error: Optional[OSError] = None
) -> None:
    """Make the next ``count`` writes to the writer's descriptor raise, or write 1 byte only."""
    real = os.write
    left = [count]

    def fake(descriptor: int, data: bytes) -> int:
        if descriptor == writer.fd() and left[0] > 0:
            left[0] -= 1
            if error is None:
                return real(descriptor, data[:1])
            raise error
        return real(descriptor, data)

    monkeypatch.setattr(os, "write", fake)


def test_lines_are_appended_whole_to_a_file_named_for_the_utc_date(
    make: Callable[..., Probe], tmp_path: Path
) -> None:
    writer = make()
    writer.submit('{"a":1}')
    writer.submit('{"a":2}')
    writer.drain()
    assert files(tmp_path / "traces") == ["2026-10-12.jsonl"]
    assert (tmp_path / "traces" / "2026-10-12.jsonl").read_bytes() == b'{"a":1}\n{"a":2}\n'
    assert writer.counts == {
        "written": 2,
        "dropped_full": 0,
        "dropped_large": 0,
        "dropped_error": 0,
        "dropped_expired": 0,
    }


@POSIX_ONLY
def test_the_directory_is_0700_and_the_file_0600(
    make: Callable[..., Probe], tmp_path: Path
) -> None:
    writer = make(tmp_path / "a" / "b" / "traces")
    writer.submit("{}")
    writer.drain()
    directory = tmp_path / "a" / "b" / "traces"
    assert stat.S_IMODE(os.stat(directory).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(directory / "2026-10-12.jsonl").st_mode) == 0o600


def test_an_existing_directory_is_left_as_it_is(make: Callable[..., Probe], tmp_path: Path) -> None:
    (tmp_path / "traces").mkdir()
    writer = make()
    writer.submit("{}")
    writer.drain()
    assert files(tmp_path / "traces") == ["2026-10-12.jsonl"]


def test_files_rotate_at_the_utc_midnight(make: Callable[..., Probe], tmp_path: Path) -> None:
    clock = Clock(T0)
    writer = make(clock=clock, window_days=30)
    writer.submit("1")
    writer.drain()
    clock.now = T0 + 14.5 * 3600 - 1  # 23:59:59 UTC
    writer.submit("2")
    writer.drain()
    clock.now += 2  # 00:00:01 the next day
    writer.submit("3")
    writer.drain()
    directory = tmp_path / "traces"
    assert files(directory) == ["2026-10-12.jsonl", "2026-10-13.jsonl"]
    assert lines(directory / "2026-10-12.jsonl") == ["1", "2"]
    assert lines(directory / "2026-10-13.jsonl") == ["3"]


def test_each_line_is_one_write_of_the_whole_line(
    make: Callable[..., Probe], monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = make()
    real = os.write
    calls: List[bytes] = []

    def spy(descriptor: int, data: bytes) -> int:
        if descriptor == writer.fd():
            calls.append(data)
        return real(descriptor, data)

    monkeypatch.setattr(os, "write", spy)
    for text in ("one", "two", "three"):
        writer.submit(text)
    writer.drain()
    assert calls == [b"one\n", b"two\n", b"three\n"]


def test_two_writers_on_one_directory_never_split_a_line(
    make: Callable[..., Probe], tmp_path: Path
) -> None:
    first, second = make(), make()
    for number in range(20):
        (first if number % 2 else second).submit(json.dumps({"n": number}))
        first.drain()
        second.drain()
    found = lines(tmp_path / "traces" / "2026-10-12.jsonl")
    assert sorted(json.loads(line)["n"] for line in found) == list(range(20))


def test_a_line_over_the_limit_is_dropped_never_truncated(
    make: Callable[..., Probe], tmp_path: Path
) -> None:
    writer = make(max_line_bytes=10)
    for text in ("x" * 10, "x" * 11, "é" * 5, "é" * 6):
        writer.submit(text)
    writer.drain()
    assert lines(tmp_path / "traces" / "2026-10-12.jsonl") == ["x" * 10, "é" * 5]
    assert writer.counts["written"] == 2
    assert writer.counts["dropped_large"] == 2


def test_a_full_queue_drops_and_counts_without_waiting(
    make: Callable[..., Probe], tmp_path: Path
) -> None:
    writer = make(max_queue=2)
    for number in range(5):
        writer.submit(str(number))
    assert writer.counts["dropped_full"] == 3
    writer.drain()
    assert lines(tmp_path / "traces" / "2026-10-12.jsonl") == ["0", "1"]


def test_a_full_queue_with_a_busy_writer_thread_does_not_block_submit(
    make: Callable[..., Probe], monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = make(max_queue=2, start=True)
    real = writer.real_write()
    entered, release = threading.Event(), threading.Event()

    def blocked(line: str) -> None:
        entered.set()
        assert release.wait(5)
        real(line)

    monkeypatch.setattr(writer, "_write", blocked)
    try:
        writer.submit("a")
        assert entered.wait(5)  # the thread holds "a" and is stuck writing it
        writer.submit("b")
        writer.submit("c")
        writer.submit("d")  # the queue holds b and c: this one is dropped at once
        assert writer.counts["dropped_full"] == 1
    finally:
        release.set()
    writer.close()
    assert writer.counts["written"] == 3


def test_close_gives_up_on_a_stuck_thread_and_stops_it_once_free(
    make: Callable[..., Probe], monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = make(max_queue=1, start=True)
    real = writer.real_write()
    entered, release = threading.Event(), threading.Event()

    def blocked(line: str) -> None:
        entered.set()
        assert release.wait(5)
        real(line)

    monkeypatch.setattr(writer, "_write", blocked)
    writer.submit("a")
    assert entered.wait(5)
    writer.submit("b")  # the queue is full: close cannot even queue its stop signal
    writer.close(timeout=0.05)
    assert writer.alive()
    release.set()
    writer.close(timeout=5)
    assert not writer.alive()
    assert writer.counts["written"] == 2


def test_close_drains_everything_queued_and_stops_the_thread(
    make: Callable[..., Probe], tmp_path: Path
) -> None:
    writer = make(start=True)
    for number in range(50):
        writer.submit(json.dumps({"n": number}))
    writer.close()
    assert not writer.alive()
    found = lines(tmp_path / "traces" / "2026-10-12.jsonl")
    assert [json.loads(line)["n"] for line in found] == list(range(50))
    writer.close()  # closing again is harmless
    assert writer.counts["written"] == 50


def test_close_writes_the_queue_of_a_writer_without_a_thread(
    make: Callable[..., Probe], tmp_path: Path
) -> None:
    writer = make()
    writer.submit("x")
    writer.close()
    assert lines(tmp_path / "traces" / "2026-10-12.jsonl") == ["x"]


def test_many_threads_submitting_lose_and_split_nothing(
    make: Callable[..., Probe], tmp_path: Path
) -> None:
    writer = make(start=True, max_queue=10_000)

    def produce(worker: int) -> None:
        for number in range(100):
            writer.submit(json.dumps({"w": worker, "n": number}))

    producers = [threading.Thread(target=produce, args=(worker,)) for worker in range(4)]
    for producer in producers:
        producer.start()
    for producer in producers:
        producer.join(10)
    writer.close()
    found = [json.loads(x) for x in lines(tmp_path / "traces" / "2026-10-12.jsonl")]
    assert len(found) == 400
    assert {(x["w"], x["n"]) for x in found} == {(w, n) for w in range(4) for n in range(100)}
    assert writer.counts["written"] == 400


def test_the_recording_window_ends_once_and_later_lines_are_not_written(
    make: Callable[..., Probe], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    clock = Clock(T0)
    writer = make(clock=clock, window_days=14)
    clock.now = T0 + 14 * DAY - 1
    writer.submit("last")
    writer.drain()
    assert not writer.expired
    clock.now = T0 + 14 * DAY
    with caplog.at_level(logging.INFO, logger="dagnam_record"):
        for text in ("a", "b", "c"):
            writer.submit(text)
        writer.drain()
    assert writer.expired
    assert writer.counts["written"] == 1
    assert writer.counts["dropped_expired"] == 3
    assert [r.levelno for r in caplog.records if r.name == "dagnam_record"] == [logging.INFO]
    (path,) = (tmp_path / "traces").glob("*.jsonl")
    assert lines(path) == ["last"]


def test_the_window_is_counted_from_the_first_run_across_restarts(
    make: Callable[..., Probe], tmp_path: Path
) -> None:
    clock = Clock(T0)
    first = make(clock=clock, window_days=1)
    first.submit("a")
    first.drain()
    first.close()
    clock.now = T0 + DAY  # a new process, a day later, in the same directory
    second = make(clock=clock, window_days=1)
    second.submit("b")
    second.drain()
    assert second.counts["written"] == 1  # the first run's count carries over
    assert second.counts["dropped_expired"] == 1
    assert lines(next((tmp_path / "traces").glob("*.jsonl"))) == ["a"]


@pytest.mark.parametrize(
    "content",
    ["not json", "[1]", '{"counts": 3, "started": true}', '{"counts": {"written": -4}}'],
)
def test_an_unusable_status_file_is_ignored(
    make: Callable[..., Probe], tmp_path: Path, content: str
) -> None:
    directory = tmp_path / "traces"
    directory.mkdir()
    (directory / ".status.json").write_text(content, encoding="utf-8")
    writer = make(window_days=1)
    writer.submit("a")
    writer.drain()
    assert writer.counts["written"] == 1
    assert not writer.expired


def test_an_unwritable_directory_counts_errors_and_never_raises(
    make: Callable[..., Probe], tmp_path: Path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory", encoding="utf-8")
    writer = make(blocker / "traces")
    writer.submit("a")
    writer.submit("b")
    writer.drain()
    writer.close()
    assert writer.counts["dropped_error"] == 2
    assert writer.counts["written"] == 0


def test_a_full_disk_counts_the_error_and_the_thread_carries_on(
    make: Callable[..., Probe], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    writer = make(start=True)
    fail_writes(monkeypatch, writer, 2, OSError(errno.ENOSPC, "No space left on device"))
    for text in ("a", "b", "c"):
        writer.submit(text)
    writer.close()
    assert writer.counts["dropped_error"] == 2
    assert writer.counts["written"] == 1
    assert lines(tmp_path / "traces" / "2026-10-12.jsonl") == ["c"]


def test_a_closed_descriptor_counts_the_error_and_the_next_line_reopens(
    make: Callable[..., Probe], tmp_path: Path
) -> None:
    writer = make()
    writer.submit("a")
    writer.drain()
    descriptor = writer.fd()
    assert descriptor is not None
    os.close(descriptor)  # something else closed it under the writer
    writer.submit("b")
    writer.submit("c")
    writer.drain()
    assert writer.counts["dropped_error"] == 1
    assert writer.counts["written"] == 2
    assert lines(tmp_path / "traces" / "2026-10-12.jsonl") == ["a", "c"]


def test_a_short_write_is_an_error(
    make: Callable[..., Probe], monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = make()
    writer.submit("warm")
    writer.drain()
    fail_writes(monkeypatch, writer, 1)
    writer.submit("half")
    writer.drain()
    assert writer.counts["dropped_error"] == 1
    assert writer.counts["written"] == 1


def test_a_line_that_cannot_be_encoded_is_an_error_not_a_crash(
    make: Callable[..., Probe], tmp_path: Path
) -> None:
    writer = make()
    writer.submit('{"a":"\ud800"}')
    writer.submit("ok")
    writer.drain()
    assert writer.counts["dropped_error"] == 1
    assert lines(tmp_path / "traces" / "2026-10-12.jsonl") == ["ok"]


def test_counts_are_saved_every_30_seconds_when_changed_and_at_close(
    make: Callable[..., Probe], tmp_path: Path
) -> None:
    clock = Clock(T0)
    directory = tmp_path / "traces"
    writer = make(clock=clock)
    writer.submit("a")
    writer.drain()
    assert not (directory / ".status.json").exists()  # under 30 s: not yet
    clock.now = T0 + 31
    writer.submit("b")
    writer.drain()
    saved = status(directory)
    assert saved["counts"]["written"] == 2
    assert saved["started"] == T0
    assert saved["expired"] is False
    assert saved["directory"] == str(directory)
    clock.now += 5
    writer.submit("c")
    writer.drain()
    assert status(directory)["counts"]["written"] == 2  # changed, but under 30 s since the save
    writer.close()
    assert status(directory)["counts"]["written"] == 3
    assert [p for p in directory.iterdir() if p.name.endswith(".tmp")] == []


@POSIX_ONLY
def test_the_status_file_is_private(make: Callable[..., Probe], tmp_path: Path) -> None:
    writer = make()
    writer.submit("a")
    writer.close()
    assert stat.S_IMODE(os.stat(tmp_path / "traces" / ".status.json").st_mode) == 0o600


def test_an_unchanged_writer_saves_nothing_more(make: Callable[..., Probe], tmp_path: Path) -> None:
    clock = Clock(T0)
    writer = make(clock=clock)
    writer.close()  # the start time is the one thing to save
    saved = (tmp_path / "traces" / ".status.json").read_bytes()
    clock.now += 100
    writer.close()
    assert (tmp_path / "traces" / ".status.json").read_bytes() == saved


def test_the_idle_loop_saves_changed_counts_and_stops_on_the_signal(
    make: Callable[..., Probe], tmp_path: Path
) -> None:
    clock = Clock(T0)
    writer = make(clock=clock)
    writer.swap_queue(ScriptedQueue([queue.Empty, "line", None]))
    clock.now = T0 + 31
    writer.loop()
    assert status(tmp_path / "traces")["counts"]["written"] == 0  # saved while idle
    assert lines(tmp_path / "traces" / "2026-10-12.jsonl") == ["line"]


def test_the_expiry_is_saved_with_the_counts(make: Callable[..., Probe], tmp_path: Path) -> None:
    clock = Clock(T0)
    writer = make(clock=clock, window_days=1)
    clock.now = T0 + DAY
    writer.submit("a")
    writer.drain()
    writer.close()
    saved = status(tmp_path / "traces")
    assert saved["expired"] is True
    assert saved["counts"]["dropped_expired"] == 1


def test_a_status_file_that_cannot_be_saved_is_not_an_error(
    make: Callable[..., Probe], tmp_path: Path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    writer = make(blocker / "traces")
    writer.close()
    assert not (blocker / "traces").exists()


def test_close_is_registered_for_exit_and_unregistered_once_run(
    make: Callable[..., Probe], monkeypatch: pytest.MonkeyPatch
) -> None:
    registered: List[Any] = []
    unregistered: List[Any] = []
    monkeypatch.setattr(atexit, "register", registered.append)
    monkeypatch.setattr(atexit, "unregister", unregistered.append)
    writer = make()
    assert registered == [writer.close]
    writer.close()
    assert unregistered == [writer.close]
