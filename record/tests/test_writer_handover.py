"""A writer handed over to its successor loses nothing, and a purged file is replaced."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import threading
from typing import Any, List

import pytest

from dagnam_record._writer import Writer

T0 = 1_791_797_400.0  # 2026-10-12T09:30:00Z
DAY = 86_400


def lines(directory: Path) -> List[str]:
    found: List[str] = []
    for path in sorted(directory.glob("*.jsonl")):
        found += path.read_text(encoding="utf-8").splitlines()
    return found


def saved(directory: Path) -> Any:
    return json.loads((directory / ".status.json").read_text(encoding="utf-8"))


def test_lines_and_counts_that_arrive_after_the_hand_over_go_to_the_successor(
    make: Any, new_clock: Any, tmp_path: Path
) -> None:
    old = make(clock=new_clock(T0))
    new = make(clock=new_clock(T0 + 5))
    old.submit("before")
    old.hand_over(new)
    old.submit("late")
    old.count("dropped_partial")
    old.count("dropped_partial")
    new.drain()
    assert lines(tmp_path / "traces") == ["before", "late"]
    assert new.counts["written"] == 2
    assert new.counts["dropped_partial"] == 2
    assert new.counts["dropped_error"] == 0
    assert old.counts["written"] == 1  # what the old writer did itself stays on record


def test_the_successor_in_the_same_directory_keeps_the_window_and_the_counts(
    make: Any, new_clock: Any, tmp_path: Path
) -> None:
    old = make(clock=new_clock(T0))
    old.count("dropped_full")
    old.submit("a")
    new = make(clock=new_clock(T0 + 3 * DAY))
    old.hand_over(new)
    new.submit("b")
    new.close()
    status = saved(tmp_path / "traces")
    assert status["started"] == T0
    assert status["counts"]["written"] == 2
    assert status["counts"]["dropped_full"] == 1


def test_the_window_start_is_carried_and_the_new_window_length_decides(
    make: Any, new_clock: Any, tmp_path: Path
) -> None:
    clock = new_clock(T0)
    old = make(clock=clock, window_days=1)
    clock.now = T0 + 2 * DAY
    old.submit("x")
    old.drain()
    assert old.expired
    same = make(clock=clock, window_days=1)
    old.hand_over(same)
    same.submit("y")
    same.drain()
    assert same.expired
    assert same.counts["dropped_expired"] == 2
    longer = make(clock=clock, window_days=5)
    same.hand_over(longer)
    longer.submit("z")
    longer.drain()
    assert not longer.expired
    assert lines(tmp_path / "traces") == ["z"]


def test_a_successor_in_another_directory_starts_its_own_counts(
    make: Any, new_clock: Any, tmp_path: Path
) -> None:
    old = make(tmp_path / "a", clock=new_clock(T0))
    new = make(tmp_path / "b", clock=new_clock(T0))
    old.submit("one")
    old.hand_over(new)
    old.submit("late")
    new.drain()
    assert lines(tmp_path / "a") == ["one"]
    assert lines(tmp_path / "b") == ["late"]
    assert saved(tmp_path / "a")["counts"]["written"] == 1
    assert new.counts["written"] == 1


def test_a_closed_writer_without_a_successor_still_counts_a_late_line_as_an_error(
    make: Any,
) -> None:
    writer = make()
    writer.close()
    writer.submit("late")
    assert writer.counts["dropped_error"] == 1


def test_a_writer_notices_that_its_file_was_deleted_and_starts_a_new_one(
    make: Any, tmp_path: Path
) -> None:
    writer = make()
    writer.submit("first")
    writer.drain()
    (path,) = (tmp_path / "traces").glob("*.jsonl")
    os.unlink(path)
    writer.submit("second")
    writer.drain()
    assert lines(tmp_path / "traces") == ["second"]
    assert writer.counts["written"] == 2


def test_a_writer_keeps_appending_to_a_file_that_still_exists(make: Any, tmp_path: Path) -> None:
    writer = make()
    for text in ("a", "b", "c"):
        writer.submit(text)
        writer.drain()
    assert lines(tmp_path / "traces") == ["a", "b", "c"]


def total(directories: List[Path], key: str) -> int:
    return sum(saved(directory)["counts"][key] for directory in directories)


@pytest.mark.parametrize("same_directory", [False, True])
def test_a_count_that_lands_just_after_the_old_writers_save_is_not_lost(
    same_directory: bool, make: Any, new_clock: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "a"
    second = first if same_directory else tmp_path / "b"
    old = make(first, clock=new_clock(T0))
    new = make(second, clock=new_clock(T0))
    real = old.real_persist()

    def persist() -> None:
        real()
        old.count("dropped_full")  # a call that began with the old recorder ends right now

    monkeypatch.setattr(old, "_persist", persist)
    old.hand_over(new)
    monkeypatch.undo()
    new.close()
    assert total(sorted({first, second}), "dropped_full") == 1


class Blocked:
    """Holds the writer thread inside its first write until released."""

    def __init__(self, writer: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        real = writer.real_write()

        def slow(line: str) -> None:
            self.entered.set()
            self.release.wait(10)
            real(line)

        monkeypatch.setattr(writer, "_write", slow)


def test_a_thread_that_outlives_the_hand_over_still_has_its_writes_counted(
    make: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = make(tmp_path / "a", start=True)
    new = make(tmp_path / "b")
    blocked = Blocked(old, monkeypatch)
    for number in range(5):
        old.submit(f"line {number}")
    assert blocked.entered.wait(5)
    old.hand_over(new, timeout=0.05)  # the thread is stuck, so close() gives up waiting for it
    blocked.release.set()
    thread = old.thread()
    assert thread is not None
    thread.join(5)
    assert not thread.is_alive()
    assert len(lines(tmp_path / "a")) == 5
    assert new.counts["written"] == 5  # written after the hand-over: counted by the successor
    assert old.counts["written"] == 0
    assert saved(tmp_path / "a")["counts"]["written"] == 0  # what the old writer had done by then


def test_a_closed_writer_stops_even_when_its_queue_was_full(
    make: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = make(start=True, max_queue=2)
    blocked = Blocked(writer, monkeypatch)
    writer.submit("one")
    assert blocked.entered.wait(5)
    writer.submit("two")
    writer.submit("three")  # the queue is full: the stop marker has no room
    writer.close(timeout=0.05)
    blocked.release.set()
    thread = writer.thread()
    assert thread is not None
    thread.join(5)
    assert not thread.is_alive()
    assert lines(tmp_path / "traces") == ["one", "two", "three"]


def test_a_long_chain_of_hand_overs_is_followed_without_recursion_and_flattened(
    make: Any, tmp_path: Path
) -> None:
    writers = [make() for _ in range(1500)]
    for earlier, later in zip(writers, writers[1:]):
        earlier.hand_over(later)
    first, last = writers[0], writers[-1]
    first.submit("late")
    first.count("dropped_partial")
    last.drain()
    assert lines(tmp_path / "traces") == ["late"]
    assert last.counts["dropped_partial"] == 1
    assert first.successor() is last
    assert first.moved_to() is last


class Raising:
    """A queue that cannot take a line."""

    def put_nowait(self, item: object) -> None:
        """Always fails."""
        raise RecursionError("too deep")

    def get_nowait(self) -> None:
        """Nothing to drain."""
        raise queue.Empty


def test_nothing_escapes_submit_when_a_forward_fails(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, new = make(), make()
    old.hand_over(new)
    new.swap_queue(Raising())
    old.submit("one")  # forwarded, and the successor's queue breaks
    assert new.counts["dropped_error"] == 1

    def broken(self: Writer, line: str) -> None:
        raise RecursionError("too deep")

    monkeypatch.setattr(Writer, "_offer", broken)  # the first step of every submit
    old.submit("two")
    assert new.counts["dropped_error"] == 2


def test_a_handed_over_writer_never_saves_over_its_successors_file(
    make: Any, new_clock: Any, tmp_path: Path
) -> None:
    old = make(clock=new_clock(T0))
    new = make(clock=new_clock(T0))
    old.hand_over(new)
    new.count("dropped_full")
    new.close()
    before = (tmp_path / "traces" / ".status.json").read_text(encoding="utf-8")
    assert saved(tmp_path / "traces")["counts"]["dropped_full"] == 1
    old.mark_dirty()  # a thread of the old writer that was still running reaches its save
    old.real_persist()()
    assert (tmp_path / "traces" / ".status.json").read_text(encoding="utf-8") == before
