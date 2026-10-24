"""The writer: whole lines in daily files, bounded, fail-open, with a recording window."""

from __future__ import annotations

import errno
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, List, Optional

import pytest

T0 = 1_791_797_400.0  # 2026-10-12T09:30:00Z
DAY = 86_400


def lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def fail_writes(
    monkeypatch: pytest.MonkeyPatch, writer: Any, count: int, error: Optional[OSError] = None
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
    make: Any, tmp_path: Path
) -> None:
    writer = make()
    writer.submit('{"a":1}')
    writer.submit('{"a":2}')
    writer.drain()
    assert sorted(p.name for p in (tmp_path / "traces").iterdir()) == ["2026-10-12.jsonl"]
    assert (tmp_path / "traces" / "2026-10-12.jsonl").read_bytes() == b'{"a":1}\n{"a":2}\n'
    assert writer.counts == {
        "written": 2,
        "dropped_full": 0,
        "dropped_large": 0,
        "dropped_error": 0,
        "dropped_expired": 0,
        "dropped_partial": 0,
    }


def test_files_rotate_at_the_utc_midnight(make: Any, new_clock: Any, tmp_path: Path) -> None:
    clock = new_clock(T0)
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
    assert sorted(p.name for p in directory.glob("*.jsonl")) == [
        "2026-10-12.jsonl",
        "2026-10-13.jsonl",
    ]
    assert lines(directory / "2026-10-12.jsonl") == ["1", "2"]
    assert lines(directory / "2026-10-13.jsonl") == ["3"]


def test_each_line_is_one_write_of_the_whole_line(
    make: Any, monkeypatch: pytest.MonkeyPatch
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


def test_two_writers_on_one_directory_never_split_a_line(make: Any, tmp_path: Path) -> None:
    first, second = make(), make()
    for number in range(20):
        (first if number % 2 else second).submit(json.dumps({"n": number}))
        first.drain()
        second.drain()
    found = lines(tmp_path / "traces" / "2026-10-12.jsonl")
    assert sorted(json.loads(line)["n"] for line in found) == list(range(20))


def test_a_line_over_the_limit_is_dropped_never_truncated(make: Any, tmp_path: Path) -> None:
    writer = make(max_line_bytes=10)
    for text in ("x" * 10, "x" * 11, "é" * 5, "é" * 6):
        writer.submit(text)
    writer.drain()
    assert lines(tmp_path / "traces" / "2026-10-12.jsonl") == ["x" * 10, "é" * 5]
    assert writer.counts["written"] == 2
    assert writer.counts["dropped_large"] == 2


def test_a_line_too_long_even_in_characters_never_enters_the_queue(make: Any) -> None:
    writer = make(max_line_bytes=10)
    writer.submit("x" * 11)  # more characters than the limit has bytes: no need to queue it
    assert writer.counts["dropped_large"] == 1
    assert writer.pending().qsize() == 0
    writer.submit("é" * 6)  # 6 characters but 12 bytes: the writer thread catches it
    assert writer.pending().qsize() == 1
    writer.drain()
    assert writer.counts["dropped_large"] == 2


def test_submit_never_raises_even_for_something_that_is_not_a_line(make: Any) -> None:
    writer = make()
    writer.submit(5)
    writer.submit(None)
    assert writer.counts["dropped_error"] == 2


def test_a_full_queue_drops_and_counts_without_waiting(make: Any, tmp_path: Path) -> None:
    writer = make(max_queue=2)
    for number in range(5):
        writer.submit(str(number))
    assert writer.counts["dropped_full"] == 3
    writer.drain()
    assert lines(tmp_path / "traces" / "2026-10-12.jsonl") == ["0", "1"]


def test_a_full_queue_with_a_busy_writer_thread_does_not_block_submit(
    make: Any, monkeypatch: pytest.MonkeyPatch
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
    make: Any, monkeypatch: pytest.MonkeyPatch
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


def test_close_drains_everything_queued_and_stops_the_thread(make: Any, tmp_path: Path) -> None:
    writer = make(start=True)
    for number in range(50):
        writer.submit(json.dumps({"n": number}))
    writer.close()
    assert not writer.alive()
    found = lines(tmp_path / "traces" / "2026-10-12.jsonl")
    assert [json.loads(line)["n"] for line in found] == list(range(50))
    writer.close()  # closing again is harmless
    assert writer.counts["written"] == 50


def test_close_writes_the_queue_of_a_writer_without_a_thread(make: Any, tmp_path: Path) -> None:
    writer = make()
    writer.submit("x")
    writer.close()
    assert lines(tmp_path / "traces" / "2026-10-12.jsonl") == ["x"]


def test_a_line_submitted_after_close_is_counted_and_not_queued(make: Any, tmp_path: Path) -> None:
    writer = make(start=True)
    writer.submit("before")
    writer.close()
    writer.submit("after")
    assert writer.counts["dropped_error"] == 1
    assert writer.pending().qsize() == 0
    assert lines(tmp_path / "traces" / "2026-10-12.jsonl") == ["before"]


def test_many_threads_submitting_lose_and_split_nothing(make: Any, tmp_path: Path) -> None:
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
    make: Any, new_clock: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    clock = new_clock(T0)
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
    make: Any, new_clock: Any, tmp_path: Path
) -> None:
    clock = new_clock(T0)
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
def test_an_unusable_status_file_is_ignored(make: Any, tmp_path: Path, content: str) -> None:
    directory = tmp_path / "traces"
    directory.mkdir()
    (directory / ".status.json").write_text(content, encoding="utf-8")
    writer = make(window_days=1)
    writer.submit("a")
    writer.drain()
    assert writer.counts["written"] == 1
    assert not writer.expired


def test_an_unwritable_directory_counts_errors_and_never_raises(make: Any, tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory", encoding="utf-8")
    writer = make(blocker / "traces")
    writer.submit("a")
    writer.submit("b")
    writer.drain()
    writer.close()
    assert writer.counts["dropped_error"] == 3  # the writer's own check, then each line
    assert writer.counts["written"] == 0


def test_a_full_disk_counts_the_error_and_the_thread_carries_on(
    make: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
    make: Any, tmp_path: Path
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


def test_a_short_write_is_an_error(make: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    writer = make()
    writer.submit("warm")
    writer.drain()
    fail_writes(monkeypatch, writer, 1)
    writer.submit("half")
    writer.drain()
    assert writer.counts["dropped_error"] == 1
    assert writer.counts["written"] == 1


def test_a_line_that_cannot_be_encoded_is_an_error_not_a_crash(make: Any, tmp_path: Path) -> None:
    writer = make()
    writer.submit('{"a":"\ud800"}')
    writer.submit("ok")
    writer.drain()
    assert writer.counts["dropped_error"] == 1
    assert lines(tmp_path / "traces" / "2026-10-12.jsonl") == ["ok"]


def test_a_partial_call_is_counted_and_saved_with_the_other_counts(
    make: Any, tmp_path: Path
) -> None:
    writer = make()
    writer.count("dropped_partial")
    writer.count("dropped_partial")
    writer.close()
    saved = json.loads((tmp_path / "traces" / ".status.json").read_text(encoding="utf-8"))
    assert saved["counts"]["dropped_partial"] == 2
