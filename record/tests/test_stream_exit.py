"""Streams that are finished but never closed, and still alive when the process ends, are recorded.

Each case runs a child interpreter (see ``_exit_child``): the garbage collector is off, the stream
sits in a reference cycle or in a module global, and nothing closes the writer by hand. The exit
hooks must record a finished stream and count an unfinished one, exactly once, before the writer
closes.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import pytest

TESTS = Path(__file__).parent
RECORD = TESTS.parent


def run_child(scenario: str, directory: Path) -> None:
    result = subprocess.run(  # noqa: S603 - fixed interpreter and script path, no untrusted input
        [sys.executable, str(TESTS / "_exit_child.py"), scenario, str(directory)],
        capture_output=True,
        text=True,
        timeout=120,
        env={
            "PYTHONPATH": f"{RECORD}{':'}{TESTS}",
            "PATH": "/usr/bin:/bin",
            "PYTHONWARNINGS": "ignore",
        },
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "done" in result.stdout
    assert "Exception ignored" not in result.stderr, result.stderr
    assert "Traceback" not in result.stderr, result.stderr


def outcome(directory: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    lines: List[Dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        lines += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    counts = json.loads((directory / ".status.json").read_text(encoding="utf-8"))["counts"]
    return lines, counts


@pytest.mark.parametrize(
    ("scenario", "provider"),
    [
        ("anthropic_local", "anthropic"),
        ("anthropic_global", "anthropic"),
        ("anthropic_helper", "anthropic"),
        ("openai_local", "openai"),
        ("openai_global", "openai"),
    ],
)
def test_a_finished_stream_that_was_never_closed_is_recorded_once_at_exit(
    tmp_path: Path, scenario: str, provider: str
) -> None:
    run_child(scenario, tmp_path / "traces")
    lines, counts = outcome(tmp_path / "traces")
    assert [line["provider"] for line in lines] == [provider]
    assert lines[0]["norm"]["response"] in ("Hello", "Hello there")
    assert counts["written"] == 1
    assert counts["dropped_partial"] == 0
    assert counts["dropped_error"] == 0


@pytest.mark.parametrize(
    "scenario", ["anthropic_unfinished", "anthropic_helper_unfinished", "openai_unfinished"]
)
def test_an_unfinished_stream_still_alive_at_exit_is_counted_once_and_not_written(
    tmp_path: Path, scenario: str
) -> None:
    run_child(scenario, tmp_path / "traces")
    lines, counts = outcome(tmp_path / "traces")
    assert lines == []
    expected = 2 if scenario == "anthropic_unfinished" else 1
    assert counts["dropped_partial"] == expected
    assert counts["dropped_error"] == 0


def test_a_writer_made_after_the_exit_hook_still_gets_its_row(tmp_path: Path) -> None:
    run_child("late_writer", tmp_path / "traces")
    first, _ = outcome(tmp_path / "traces")
    later, counts = outcome(tmp_path / "traces-later")
    assert len(first) == 1
    assert len(later) == 1
    assert counts["dropped_error"] == 0


def test_a_collection_inside_submit_cannot_deadlock_on_the_writers_lock(tmp_path: Path) -> None:
    """The collector finalizes a finished stream inside ``submit`` (holding its gate): no hang."""
    run_child("gc_in_submit", tmp_path / "traces")
    lines, counts = outcome(tmp_path / "traces")
    assert [line["provider"] for line in lines] == ["anthropic"]  # delivered after the lock
    assert counts["dropped_error"] == 0
    hooked = (tmp_path / "traces-hooked").glob("*.jsonl")
    assert [path.read_text(encoding="utf-8") for path in hooked] == ["{}\n"]
