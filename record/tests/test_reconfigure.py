"""Calls that land while ``install`` swaps the recorder are all accounted for."""

from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any, Dict, List

from _api_support import record, saved_counts
import pytest

import dagnam_record
from dagnam_record import _state
from dagnam_record._writer import COUNT_KEYS, Writer

pytestmark = pytest.mark.usefixtures("fresh_state")
CALL: Dict[str, Any] = {"prompt": "hello"}
DROPS = [key for key in COUNT_KEYS if key.startswith("dropped")]


def stored(directory: Path) -> List[str]:
    found: List[str] = []
    for path in sorted(directory.glob("*.jsonl")):
        found += path.read_text(encoding="utf-8").splitlines()
    return found


def send_through(recorder: Any, times: int) -> None:
    for _ in range(times):
        recorder.record("other", "custom", CALL, CALL, status=200, latency_ms=1.0)


def test_calls_that_land_during_a_reconfiguration_are_all_kept_and_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "t"
    dagnam_record.install(dir=str(directory), window_days=5)
    for _ in range(3):
        record()
    installed = _state.STATE.installed
    assert installed is not None
    old_recorder = installed.recorder
    seen: List[str] = []
    real = Writer.hand_over

    def spy(self: Writer, successor: Writer) -> None:
        # The new recorder is already in use here; the old one is still held by a call that began
        # before the swap.
        current = _state.STATE.installed
        seen.append("in use" if current is not None and current.writer is successor else "not yet")
        send_through(old_recorder, 5)  # reaches the old writer before it stops
        record()
        record()  # new calls go to the new recorder
        real(self, successor)
        send_through(old_recorder, 4)  # reaches the old writer after it stopped
        old_recorder.writer.count("dropped_partial")  # a count that arrives late, too

    monkeypatch.setattr(Writer, "hand_over", spy)
    dagnam_record.install(dir=str(directory), window_days=6)
    monkeypatch.undo()
    assert seen == ["in use"]
    dagnam_record.uninstall()
    sent = 3 + 5 + 2 + 4
    assert len(stored(directory)) == sent
    counts = saved_counts(directory)
    assert counts["written"] == sent
    assert counts["dropped_partial"] == 1
    assert [counts[key] for key in DROPS if key != "dropped_partial"] == [0, 0, 0, 0]
    assert all(json.loads(line)["provider"] == "other" for line in stored(directory))


def test_a_stress_of_calls_from_four_threads_during_many_reconfigurations_adds_up(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "t"
    dagnam_record.install(dir=str(directory), window_days=3)
    per_thread = 250
    barrier = threading.Barrier(5)

    def work() -> None:
        barrier.wait(timeout=30)
        for _ in range(per_thread):
            record()

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=30)
    for number in range(20):
        dagnam_record.install(dir=str(directory), window_days=4 + number % 2)
    for thread in threads:
        thread.join(60)
        assert not thread.is_alive()
    dagnam_record.uninstall()
    counts = saved_counts(directory)
    sent = 4 * per_thread
    assert len(stored(directory)) == counts["written"]
    assert counts["written"] + sum(counts[key] for key in DROPS) == sent
    assert counts["dropped_error"] == 0


def test_a_stress_across_two_alternating_directories_adds_up(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    dagnam_record.install(dir=str(first))
    per_thread = 300
    barrier = threading.Barrier(7)

    def work() -> None:
        barrier.wait(timeout=30)
        for _ in range(per_thread):
            record()

    threads = [threading.Thread(target=work) for _ in range(6)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=30)
    for number in range(15):
        dagnam_record.install(dir=str(second if number % 2 == 0 else first))
    for thread in threads:
        thread.join(60)
        assert not thread.is_alive()
    dagnam_record.uninstall()
    # Each directory's file holds the counts of the writers that worked there, and a count that
    # reached a writer after it was handed over is on its successor's directory: together, all.
    counted = {key: saved_counts(first)[key] + saved_counts(second)[key] for key in COUNT_KEYS}
    assert sum(counted.values()) == 6 * per_thread
    assert len(stored(first)) + len(stored(second)) == counted["written"]
    assert counted["dropped_error"] == 0
