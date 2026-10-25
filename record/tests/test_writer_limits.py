"""The writer's limits and failures: bytes in memory, bytes on disk, no thread, a stuck disk."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterator

import pytest

import dagnam_record
from dagnam_record._writer import MAX_QUEUE_BYTES, MAX_TOTAL_BYTES

T0 = 1_791_797_400.0  # 2026-10-12T09:30:00Z
DAY = 86_400
HUNDRED = "x" * 100


def data_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.glob("*.jsonl"))


def test_the_queue_is_bounded_by_bytes_as_well_as_by_lines(make: Any) -> None:
    writer = make(max_queue_bytes=250)  # no thread: a disk that never answers
    for _ in range(5):
        writer.submit(HUNDRED)
    assert writer.pending().qsize() == 2
    assert writer.counts["dropped_full"] == 3
    writer.drain()
    assert writer.counts["written"] == 2
    writer.submit(HUNDRED)  # what was written made room again
    writer.submit(HUNDRED)
    assert writer.pending().qsize() == 2
    assert writer.counts["dropped_full"] == 3


def test_one_line_over_the_whole_byte_budget_is_dropped_not_queued(make: Any) -> None:
    writer = make(max_queue_bytes=50)
    writer.submit(HUNDRED)
    assert writer.pending().qsize() == 0
    assert writer.counts["dropped_full"] == 1


def test_the_default_budgets_are_32_mib_in_memory_and_1_gib_on_disk() -> None:
    assert (MAX_QUEUE_BYTES, MAX_TOTAL_BYTES) == (32 * 1024 * 1024, 1024**3)


def test_the_directory_is_capped_in_total_and_the_rest_is_counted(
    make: Any, tmp_path: Path
) -> None:
    writer = make(max_total_bytes=450)
    for _ in range(8):
        writer.submit(HUNDRED)
    writer.drain()
    assert writer.counts["written"] == 4  # 4 x 101 bytes fit in 450, the fifth does not
    assert writer.counts["dropped_full"] == 4
    assert data_bytes(tmp_path / "traces") == 404


def test_what_is_already_in_the_directory_counts_against_the_cap(make: Any, tmp_path: Path) -> None:
    directory = tmp_path / "traces"
    directory.mkdir(mode=0o700)
    (directory / "2026-10-01.jsonl").write_bytes(b"o" * 350)
    writer = make(max_total_bytes=450)
    for _ in range(3):
        writer.submit(HUNDRED)
    writer.drain()
    assert writer.counts["written"] == 0  # 350 + 101 is over 450
    assert writer.counts["dropped_full"] == 3


def test_a_purge_makes_room_again_and_so_does_a_new_day(
    make: Any, new_clock: Any, tmp_path: Path
) -> None:
    clock = new_clock(T0)
    writer = make(clock=clock, max_total_bytes=250)
    for _ in range(3):
        writer.submit(HUNDRED)
    writer.drain()
    assert (writer.counts["written"], writer.counts["dropped_full"]) == (2, 1)
    for path in (tmp_path / "traces").glob("*.jsonl"):
        path.unlink()  # `python -m dagnam_record purge` in another terminal
    writer.submit(HUNDRED)
    writer.drain()
    assert writer.counts["written"] == 3
    clock.now += DAY  # another process may have filled the directory meanwhile
    (tmp_path / "traces" / "2026-10-12.jsonl").write_bytes(b"o" * 300)
    writer.submit(HUNDRED)
    writer.drain()
    assert writer.counts["dropped_full"] == 2


def test_no_thread_to_start_is_a_drop_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    def refuse(self: threading.Thread) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    directory = tmp_path / "t"
    dagnam_record.install(dir=str(directory))  # must not raise
    dagnam_record.record_call(provider="other", request={"a": 1}, response={"b": 2})
    dagnam_record.uninstall()
    monkeypatch.undo()
    counts = json.loads((directory / ".status.json").read_text(encoding="utf-8"))["counts"]
    assert counts["written"] == 0
    assert counts["dropped_error"] >= 1
    assert "cannot join thread" not in capfd.readouterr().err


@pytest.fixture
def stuck_disk(monkeypatch: pytest.MonkeyPatch) -> Iterator[threading.Event]:
    """Every data write blocks until the test lets go: a hung network mount."""
    release = threading.Event()
    real = os.write

    def blocked(descriptor: int, data: bytes) -> int:
        if data.endswith(b"\n") and not release.wait(20):
            raise OSError("test gave up")
        return real(descriptor, data)

    monkeypatch.setattr(os, "write", blocked)
    yield release
    release.set()


def test_close_does_not_touch_the_disk_again_when_the_writer_thread_is_stuck_on_it(
    make: Any, tmp_path: Path, stuck_disk: threading.Event
) -> None:
    writer = make(start=True)
    writer.submit(HUNDRED)
    deadline = time.monotonic() + 5
    while writer.pending().qsize() and time.monotonic() < deadline:
        time.sleep(0.01)  # until the thread has taken the line and is stuck writing it
    writer.mark_dirty()
    started = time.monotonic()
    writer.close(timeout=0.2)
    assert time.monotonic() - started < 3
    assert not (tmp_path / "traces" / ".status.json").exists()  # no save from the main thread
    stuck_disk.set()
    thread = writer.thread()
    assert thread is not None
    thread.join(5)
    assert not writer.alive()
