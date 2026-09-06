"""CLI ``audit run``: the upload listing, the confirmation, the frontier, the report."""

from __future__ import annotations

from functools import partial
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest import mock

import pytest
from tests.audit._platform import Clock, FakePlatform, json_row, label_row, serve_chat, teacher

from dagnam.audit.orchestrate import run_audit
from dagnam.audit.workspace import write_workload

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, RequestsMocker, StrCapture

PASS_LIST = ["PII_EMAIL", "PII_PHONE", "PII_PAYMENT_CARD", "PII_NATIONAL_ID"]
SPLIT = {"train": list(range(16)), "eval_holdout": [16, 17, 18, 19]}


def _workload(workload_id: str, cls: str, status: str) -> dict[str, Any]:
    return {
        "id": workload_id,
        "template_hash": workload_id,
        "template_excerpt": "Classify",
        "structure_class": cls,
        "confidence": "high",
        "calls": 3_000,
        "calls_per_day": 100.0,
        "tokens": {"prompt": 300_000, "completion": 6_000},
        "cost_usd_month": 900.0,
        "cost_source": "export",
        "latency_ms": {"p50": 400.0, "p95": 900.0},
        "distinct_outputs": 2,
        "entropy": 1.0,
        "models": ["gpt-4o-mini"],
        "verdict": {"status": status, "ratio": 12.0, "savings_usd_month": 800.0, "reason": "x"},
        "dataset": None,
    }


SCAN: dict[str, Any] = {
    "schema": "dagnam.audit.scan/1",
    "generated_at": "2026-09-06T10:00:00+00:00",
    "source": "langfuse",
    "window": {
        "start": "2026-08-01T00:00:00+00:00",
        "end": "2026-08-31T00:00:00+00:00",
        "days": 30,
    },
    "price_table_version": "2026-09",
    "totals": {
        "calls": 6_000,
        "cost_usd_month": 1_800.0,
        "prompt_tokens": 1,
        "completion_tokens": 1,
    },
    "pii": {"pass_list": PASS_LIST, "counts": {"PII_EMAIL": 2}},
    "workloads": [
        _workload("w1", "enum_label", "candidate"),
        _workload("w2", "json_object", "marginal"),
        _workload("w3", "free_text", "not_audited"),
    ],
    "warnings": [],
}


def _stats(format_key: str, counts: dict[str, int]) -> dict[str, Any]:
    return {
        "format_key": format_key,
        "redact": {"counts": counts, "pass_list": PASS_LIST, "rows_changed": sum(counts.values())},
    }


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)


@pytest.fixture
def prepared_dir(tmp_path: Path) -> Path:
    root = tmp_path / "audit"
    root.mkdir()
    write_workload(
        root,
        "w1",
        [label_row(i) for i in range(20)],
        SPLIT,
        _stats("labeled-example", {"PII_EMAIL": 2}),
    )
    write_workload(root, "w2", [json_row(i) for i in range(20)], SPLIT, _stats("chat-messages", {}))
    (root / "scan-report.json").write_text(json.dumps(SCAN), encoding="utf-8")
    return root


@pytest.fixture
def platform(monkeypatch: PytestMonkeyPatch, requests_mock: RequestsMocker) -> FakePlatform:
    """The fake platform behind ``client_from_env``, with the waits on a fake clock."""
    fake = FakePlatform()
    clock = Clock()
    monkeypatch.setattr("dagnam.cli.audit_run.client_from_env", lambda: fake)
    monkeypatch.setattr(
        "dagnam.audit.orchestrate.run_audit", partial(run_audit, sleep=clock.sleep, now=clock.now)
    )
    serve_chat(requests_mock, teacher)
    return fake


@pytest.fixture
def tty(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))


def test_run_lists_exactly_what_will_be_uploaded_and_needs_yes(
    run_cli: CliRunner,
    prepared_dir: Path,
    platform: FakePlatform,
    capsys: StrCapture,
    monkeypatch: PytestMonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # not a terminal
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(prepared_dir)])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    listing, _, _ = captured.out.partition("credit ceiling")
    assert "About to upload (redacted, derived rows only; raw traces stay here):" in listing
    assert (
        "  w1 (enum_label): 20 rows (split {'train': 16, 'eval_holdout': 4});"
        " redactions: PII_EMAIL: 2 in 2 rows; scanned for PII_EMAIL, PII_PHONE,"
        " PII_PAYMENT_CARD, PII_NATIONAL_ID"
    ) in listing
    assert "  w2 (json_object): 20 rows" in listing
    assert "redactions: none found in 0 rows" in listing
    assert "w3" not in listing
    assert "  to: a new private project 'workload-audit-audit'" in listing
    assert "credit ceiling: 500" in captured.out
    assert "refusing to upload without confirmation on a non-interactive terminal" in captured.err
    assert f"dagnam audit run {prepared_dir} --yes" in captured.err
    assert platform.call_log == []  # nothing left the machine


def test_run_declined_at_the_prompt_uploads_nothing(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform, tty: None, capsys: StrCapture
) -> None:
    with mock.patch("builtins.input", return_value="no"), pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(prepared_dir), "--workloads", "w1", "--max-credits", "50"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "w1 (enum_label)" in captured.out
    assert "w2" not in captured.out
    assert "credit ceiling: 50" in captured.out
    assert "confirmation not received" in captured.err
    assert platform.call_log == []


def test_run_confirmed_runs_the_frontier_and_writes_the_report(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform, tty: None, capsys: StrCapture
) -> None:
    # Four holdout rows give a wide interval, so the floor is lowered to see a winner.
    with mock.patch("builtins.input", return_value="yes"):
        assert run_cli(["audit", "run", str(prepared_dir), "--floor", "0.5"]) == 0
    out = capsys.readouterr().out
    assert platform.call_log[0] == "create_project"
    assert platform.submits == 2
    report = json.loads((prepared_dir / "audit-report.json").read_text(encoding="utf-8"))
    assert report["schema"] == "dagnam.audit.report/1"
    assert report["project_id"] == "proj-1"
    w1, w2, w3 = report["workloads"]
    assert w1["winner"]["kind"] == "head_tune"
    assert w1["switch"] == {"base_url": "https://x/v1", "model": "dep-1", "key_ref": "w1/head_tune"}
    assert w2["winner"]["kind"] == "sft_small"
    assert w3["candidates"] == []
    assert "dk-secret" not in json.dumps(report)
    md = (prepared_dir / "audit-report.md").read_text(encoding="utf-8")
    assert "### w1 - REPLACE" in md
    assert "dk-secret" not in md
    assert "w1: REPLACE - head_tune at $" in out
    assert "switch model dep-1" in out
    assert f"Report: {prepared_dir / 'audit-report.md'}" in out
    assert "  to: existing project" not in out  # the project was created by this run

    # A second run resumes: the listing names the existing project and no platform call is made.
    calls = list(platform.call_log)
    assert run_cli(["audit", "run", str(prepared_dir), "--yes", "--json"]) == 0
    printed = json.loads(capsys.readouterr().out.partition("\n{")[2].join(["{", ""]))
    assert printed["workloads"][0]["winner"]["kind"] == "head_tune"
    assert platform.call_log == calls


def test_run_json_listing_then_report(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform, capsys: StrCapture
) -> None:
    platform.revision_final = "deploying"  # no candidate reaches an endpoint: NOT YET
    assert run_cli(["audit", "run", str(prepared_dir), "--yes", "--floor", "0.5", "--json"]) == 0
    out = capsys.readouterr().out
    assert "  to: a new private project" in out
    report = json.loads(out[out.index("\n{") + 1 :])
    assert report["workloads"][0]["winner"] is None
    assert report["workloads"][0]["candidates"][1]["status"] == "deploy_timeout"


def test_run_halted_exits_nonzero_with_the_reason(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform, capsys: StrCapture
) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(prepared_dir), "--yes", "--max-credits", "50"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "audit halted: budget" in captured.err
    assert f"dagnam audit status {prepared_dir}" in captured.err
    assert (prepared_dir / "audit-report.md").exists()
    assert platform.submits == 1  # w1 was submitted; its 100-credit estimate halts w2

    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(prepared_dir), "--yes", "--max-credits", "50", "--json"])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    error = json.loads(out[out.index("\n{") + 1 :])
    assert error["error"].startswith("audit halted: budget")
    assert error["hint"] == f"dagnam audit status {prepared_dir}"


def test_run_no_wait_returns_early_and_says_how_to_resume(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform, capsys: StrCapture
) -> None:
    assert run_cli(["audit", "run", str(prepared_dir), "--yes", "--no-wait"]) == 0
    captured = capsys.readouterr()
    assert platform.submits == 1
    assert "get_foundation_run" not in platform.call_log
    assert f"Next: dagnam audit run {prepared_dir} --yes" in captured.err
    assert "w1: NOT YET" in captured.out


def test_run_refuses_unknown_workloads_missing_scan_and_nothing_to_run(
    run_cli: CliRunner,
    prepared_dir: Path,
    platform: FakePlatform,
    tmp_path: Path,
    capsys: StrCapture,
) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(prepared_dir), "--yes", "--workloads", "w9"])
    assert exc.value.code == 1
    assert "workloads not in scan-report.json: ['w9']" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(tmp_path / "nowhere"), "--yes", "--json"])
    assert exc.value.code == 1
    assert "run `dagnam audit scan` first" in json.loads(capsys.readouterr().out)["error"]

    scan = {**SCAN, "workloads": [_workload("w3", "free_text", "not_audited")]}
    (prepared_dir / "scan-report.json").write_text(json.dumps(scan), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(prepared_dir), "--yes"])
    assert exc.value.code == 1
    assert "nothing to run" in capsys.readouterr().err
    assert platform.call_log == []
