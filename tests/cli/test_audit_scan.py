"""CLI ``audit scan``: what it reads, what it writes, and the flags it parses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
from typing import TYPE_CHECKING

import pytest

from dagnam.audit.candidates import CandidateKind
from dagnam.cli.audit import parse_credits, parse_map, parse_window

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture

HEAD, SFT, HOSTED = CandidateKind.HEAD_TUNE, CandidateKind.SFT_SMALL, CandidateKind.HOSTED_FLOOR
CALLS = 1_200
"""Enough traces for one workload to clear ``MIN_TRACES_PER_WORKLOAD`` and ``MIN_HOLDOUT``."""


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)


def _row(i: int, *, response_key: str = "response") -> dict[str, object]:
    return {
        "trace_id": f"t{i}",
        "ts": f"2026-08-{1 + i % 28:02d}T{i % 24:02d}:{i % 60:02d}:00Z",
        "model": "gpt-4o-mini",
        "system": "Label the ticket.",
        "messages": [{"role": "user", "content": f"ticket {i} from customer@example.com"}],
        response_key: "ab"[i % 2],
        "prompt_tokens": 100,
        "completion_tokens": 2,
        "latency_ms": 500,
        "cost_usd": 0.6,  # $720/month against a $50 maintenance floor: a candidate
    }


@pytest.fixture
def export(tmp_path: Path) -> Path:
    """A generic JSONL export: one label workload worth auditing plus one tiny free-text one."""
    rows = [_row(i) for i in range(CALLS)]
    rows += [
        {**_row(CALLS + i), "system": "Write a reply.", "response": f"Dear customer {i}, " * 20}
        for i in range(3)
    ]
    path = tmp_path / "traces.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_scan_opens_no_network(
    run_cli: CliRunner,
    export: Path,
    tmp_path: Path,
    monkeypatch: PytestMonkeyPatch,
    capsys: StrCapture,
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network")

    monkeypatch.setattr(socket, "socket", refuse)
    out = tmp_path / "audit"
    assert run_cli(["audit", "scan", str(export), "--source", "jsonl", "--out", str(out)]) == 0

    assert (out / "scan-report.md").exists()
    assert (out / "workloads").is_dir()
    report = json.loads((out / "scan-report.json").read_text(encoding="utf-8"))
    label, free_text = report["workloads"]
    assert label["verdict"]["status"] == "candidate"
    assert label["dataset"] == {
        "rows": CALLS,
        "dedup_removed": 0,
        "redactions": CALLS,
        "truncated": 0,
        "split": {"train": CALLS - CALLS // 5, "eval_holdout": CALLS // 5},
    }
    assert report["pii"]["counts"]["PII_EMAIL"] == CALLS
    assert report["pii"]["pass_list"] == list(report["pii"]["counts"])
    assert report["window"]["days"] == 30.0
    workload_dir = out / "workloads" / label["id"]
    assert {p.name for p in workload_dir.iterdir()} == {"dataset.jsonl", "split.json", "meta.json"}
    assert "customer@example.com" not in workload_dir.joinpath("dataset.jsonl").read_text()
    assert free_text["verdict"]["status"] == "not_audited"
    assert free_text["dataset"] is None
    captured = capsys.readouterr()
    assert "1 candidate, 1 not_audited" in captured.out
    assert f"Next: dagnam audit run {out}" in captured.err


def test_scan_with_nothing_worth_auditing_writes_no_workloads(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    sample = Path(__file__).parents[1] / "audit" / "fixtures" / "langfuse_sample.jsonl"
    out = tmp_path / "audit"
    assert run_cli(["audit", "scan", str(sample), "--source", "langfuse", "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert "Next:" not in captured.err
    assert not (out / "workloads").exists()
    report = json.loads((out / "scan-report.json").read_text(encoding="utf-8"))
    assert {w["verdict"]["status"] for w in report["workloads"]} <= {
        "too_few_samples",
        "not_audited",
    }
    assert all(w["dataset"] is None for w in report["workloads"])


def test_scan_json_map_window_and_price_table(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text(
        "".join(json.dumps(_row(i, response_key="out")) + "\n" for i in range(CALLS)),
        encoding="utf-8",
    )
    table = tmp_path / "prices.json"
    table.write_text(
        json.dumps({"version": "custom-1", "as_of": "2026-09-01", "rows": {}}), encoding="utf-8"
    )
    out = tmp_path / "audit"
    argv = [
        "audit",
        "scan",
        str(path),
        "--source",
        "jsonl",
        "--map",
        "response=out",
        "--window",
        "60d",
        "--price-table",
        str(table),
        "--out",
        str(out),
        "--json",
    ]
    assert run_cli(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads((out / "scan-report.json").read_text(encoding="utf-8"))
    assert printed["window"]["days"] == 60.0
    assert printed["price_table_version"] == "custom-1"
    assert printed["workloads"][0]["calls"] == CALLS


def test_scan_empty_export_fails_with_a_reason(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "scan", str(empty), "--source", "jsonl", "--out", str(tmp_path)])
    assert exc.value.code == 1
    assert "no traces to audit" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "scan", str(empty), "--source", "jsonl", "--json"])
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["error"].endswith("no traces to audit")


def test_scan_flag_parsers() -> None:
    assert parse_window("30d") == 30
    assert parse_window("7") == 7
    # "\u00b2" is a digit to `str.isdigit` and not a number to `int`.
    for bad in ("0d", "d", "-3", "month", "\u00b2", "\u00b2d"):
        with pytest.raises(argparse.ArgumentTypeError, match="--window expects"):
            parse_window(bad)
    assert parse_credits("0") == 0
    assert parse_credits("500") == 500
    for bad in ("-5", "5.5", "\u00b2", "many"):
        with pytest.raises(argparse.ArgumentTypeError, match="--max-credits expects"):
            parse_credits(bad)
    assert parse_map(None) is None
    assert parse_map([]) is None
    assert parse_map(["response=out", "system=sys"]) == {"response": "out", "system": "sys"}
    for bad in ("response", "=out", "response="):
        with pytest.raises(argparse.ArgumentTypeError, match="--map expects"):
            parse_map([bad])


def test_scan_argparse_rejects_bad_window(run_cli: CliRunner, capsys: StrCapture) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "scan", "x.jsonl", "--source", "jsonl", "--window", "soon"])
    assert exc.value.code == 2
    assert "--window expects" in capsys.readouterr().err
