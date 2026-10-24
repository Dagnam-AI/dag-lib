"""Helpers for the tests of the public API, the redaction hook and the command line."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
from typing import Any, Callable, Dict, List

import dagnam_record
from dagnam_record._envelope import build_line

REQUEST: Dict[str, Any] = {"prompt": "hello", "model": "m"}
RESPONSE: Dict[str, Any] = {"text": "world"}


def record(**overrides: Any) -> None:
    """One ``record_call`` with a small valid call, ``overrides`` replacing any argument."""
    arguments: Dict[str, Any] = {
        "provider": "other",
        "request": REQUEST,
        "response": RESPONSE,
        "latency_ms": 12.5,
    }
    arguments.update(overrides)
    dagnam_record.record_call(**arguments)


def finish_lines(directory: Path) -> List[Dict[str, Any]]:
    """Stop recording (which writes what is queued) and return every recorded line, parsed."""
    dagnam_record.uninstall()
    found: List[Dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        found += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return found


def saved_counts(directory: Path) -> Dict[str, int]:
    """The counts in ``.status.json``."""
    data = json.loads((directory / ".status.json").read_text(encoding="utf-8"))
    counts: Dict[str, int] = data["counts"]
    return counts


def run_forked(child: Callable[[], int]) -> int:
    """Run ``child`` in a forked process and return its exit code (an alarm bounds the wait)."""
    pid = os.fork()
    if pid == 0:
        code = 99
        try:
            signal.alarm(10)
            code = child()
        finally:
            os._exit(code)
    _, wait_status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(wait_status)


def envelope(prompt: str, now: float) -> str:
    """One real envelope line for a call made at ``now`` (seconds since the epoch)."""
    return build_line(
        provider="other",
        endpoint="custom",
        request={"prompt": prompt},
        response={"text": "reply"},
        latency_ms=1.0,
        now=now,
    )


def write_data(directory: Path, name: str, lines: List[str]) -> Path:
    """A data file with ``lines``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path
