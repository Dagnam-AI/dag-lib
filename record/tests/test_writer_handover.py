"""A writer handed over to its successor loses nothing, and a purged file is replaced."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List

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
