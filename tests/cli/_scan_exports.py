"""Exports and scans for the CLI scan tests: label calls worth auditing, and the command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner

CALLS = 1_200
TEXT = "Label the ticket."


def row(i: int, response: object, system: str = TEXT) -> dict[str, object]:
    return {
        "trace_id": f"t{i}",
        "ts": f"2026-08-{1 + i % 28:02d}T{i % 24:02d}:{i % 60:02d}:00Z",
        "model": "gpt-4o-mini",
        "system": system,
        "messages": [{"role": "user", "content": f"ticket {i}"}],
        "response": response,
        "prompt_tokens": 100,
        "completion_tokens": 2,
        "latency_ms": 500,
        "cost_usd": 0.5,  # $600 for 1,200 calls
    }


def export(
    tmp_path: Path,
    answer: object = None,
    *,
    every: int = 0,
    calls: int = CALLS,
    system: str = TEXT,
    name: str = "t.jsonl",
) -> Path:
    """Label calls; every ``every``-th one (when given) answers with ``answer`` instead."""
    rows = [
        row(i, answer if every and i % every == 0 else "ab"[i % 2], system) for i in range(calls)
    ]
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def scan(run_cli: CliRunner, source: Path, out: Path, *extra: str) -> int:
    return run_cli(["audit", "scan", str(source), "--source", "jsonl", "--out", str(out), *extra])


def refused(run_cli: CliRunner, source: Path, out: Path) -> None:
    with pytest.raises(SystemExit) as exit_:
        scan(run_cli, source, out)
    assert exit_.value.code == 1


def folder_of(out: Path) -> Path:
    """The one workload folder a scan of :func:`export` wrote."""
    report = json.loads((out / "scan-report.json").read_text(encoding="utf-8"))
    (workload,) = (w for w in report["workloads"] if w["dataset"] is not None)
    return out / "workloads" / workload["id"]
