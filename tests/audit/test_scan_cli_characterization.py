"""``dagnam audit scan`` pinned end to end: reports, derived rows, stdout and exit codes.

The goldens in ``fixtures/scan_report_golden.json`` were taken from the command BEFORE
``scan_export`` was extracted from it, so this file is the proof that the extraction changed
nothing a user can see. Only the generation timestamp is normalised. Regenerate the goldens with
``UPDATE_SCAN_GOLDENS=1`` after a deliberate change to the report.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture

from dagnam.cli import main as cli_main

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = FIXTURES / "scan_report_golden.json"
GENERATED_AT = "<generated_at>"
CALLS = 1_200


@pytest.fixture
def run_cli(monkeypatch: PytestMonkeyPatch) -> CliRunner:
    def run(argv: list[str]) -> int:
        monkeypatch.setattr("sys.argv", ["dagnam", *argv])
        return cli_main()

    return run


def _prices(tmp_path: Path) -> Path:
    """A fixed table dated today, so no price-staleness warning depends on the day the suite runs."""
    row = {"input_per_m": 0.15, "output_per_m": 0.6, "cached_input_per_m": None}
    table = {
        "version": "golden-1",
        "as_of": datetime.now(UTC).date().isoformat(),
        "rows": {"gpt-4o-mini": {**row, "cheaper_variant": None}},
    }
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    return path


def _generated_export(tmp_path: Path) -> Path:
    """A generic export with one label workload worth auditing and one tiny free-text one."""
    rows: list[dict[str, Any]] = []
    for i in range(CALLS + 3):
        label = i < CALLS
        rows.append(
            {
                "trace_id": f"t{i}",
                "ts": f"2026-08-{1 + i % 28:02d}T{i % 24:02d}:{i % 60:02d}:00Z",
                "model": "gpt-4o-mini",
                "system": "Label the ticket." if label else "Write a reply.",
                "messages": [{"role": "user", "content": f"ticket {i} from customer@example.com"}],
                "out": "ab"[i % 2] if label else f"Dear customer {i}, " * 20,
                "prompt_tokens": 100,
                "completion_tokens": 2,
                "latency_ms": 500,
                "cost_usd": 0.6,  # $720/month against a $50 floor: a candidate
            }
        )
    path = tmp_path / "generated.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _argv(case: str, tmp_path: Path, out: Path) -> list[str]:
    prices = ["--price-table", str(_prices(tmp_path))]
    if case == "generated":
        export = _generated_export(tmp_path)
        extra = ["--map", "response=out", "--window", "60d", "--sample-rate", "0.5", *prices]
        return ["audit", "scan", str(export), "--source", "jsonl", *extra, "--out", str(out)]
    export = (
        FIXTURES / "recorder_openai_chat.jsonl"
        if case == "dagnam"
        else FIXTURES / f"{case}_sample.jsonl"
    )
    return ["audit", "scan", str(export), "--source", case, *prices, "--out", str(out)]


def _digests(root: Path) -> dict[str, str]:
    """The sha256 of every file under ``root`` but the two scan reports (compared on their own)."""
    skip = {"scan-report.json", "scan-report.md"}
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name not in skip
    }


def _capture(case: str, run_cli: CliRunner, tmp_path: Path, capsys: StrCapture) -> dict[str, Any]:
    out = tmp_path / "audit"
    assert run_cli(_argv(case, tmp_path, out)) == 0
    captured = capsys.readouterr()
    report = json.loads((out / "scan-report.json").read_text(encoding="utf-8"))
    report["generated_at"] = GENERATED_AT
    markdown = (out / "scan-report.md").read_text(encoding="utf-8")
    markdown = "\n".join(
        f"- generated: {GENERATED_AT}" if line.startswith("- generated: ") else line
        for line in markdown.splitlines()
    )
    return {
        "report": report,
        "markdown": markdown,
        "files": _digests(out),
        "stdout": captured.out,
        "stderr": captured.err.replace(str(out.resolve()), "<out>"),
    }


CASES = ("openai", "langfuse", "langsmith", "dagnam", "generated")


@pytest.mark.parametrize("case", CASES)
def test_the_scan_command_output_matches_its_golden(
    case: str, run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    actual = _capture(case, run_cli, tmp_path, capsys)
    if os.environ.get("UPDATE_SCAN_GOLDENS"):
        goldens = json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.exists() else {}
        goldens[case] = actual
        GOLDEN.write_text(json.dumps(goldens, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))[case]
    assert actual == expected


def test_the_generated_golden_exercises_the_derived_rows() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))["generated"]
    datasets = [w["dataset"] for w in golden["report"]["workloads"] if w["dataset"] is not None]
    assert len(datasets) == 1
    assert datasets[0]["rows"] > 0
    assert any(name.startswith("workloads/") for name in golden["files"])


def test_no_traces_fails_with_exit_one_and_the_reason(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    out = tmp_path / "audit"
    argv = ["audit", "scan", str(empty), "--source", "jsonl", "--out", str(out)]

    with pytest.raises(SystemExit) as human:
        run_cli(argv)
    captured = capsys.readouterr()
    assert human.value.code == 1
    assert captured.out == ""
    assert captured.err == f"Error: {empty}: no traces to audit\n\nDocs: https://dagnam.ai/docs\n"

    with pytest.raises(SystemExit) as machine:
        run_cli([*argv, "--json"])
    assert machine.value.code == 1
    assert json.loads(capsys.readouterr().out) == {
        "error": f"{empty}: no traces to audit",
        "hint": None,
    }
    assert not out.exists()
