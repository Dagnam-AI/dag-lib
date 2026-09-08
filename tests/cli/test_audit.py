"""CLI ``audit scan`` / ``status`` / ``cancel`` / ``delete`` (``run`` is tests/cli/test_audit_run.py)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import socket
import sys
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from tests.audit._platform import FakeCleanup

from dagnam._core.exceptions import DeploymentNotFoundError
from dagnam.audit.candidates import CandidateKind
from dagnam.audit.state import AuditState, StepState, load_state, save_state
from dagnam.cli.audit import parse_credits, parse_map, parse_window, status_rows
from dagnam.cli.audit_run import client_from_env

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


def _state() -> AuditState:
    state = AuditState(project_id="proj-1")
    state.workloads["w1"] = {
        HOSTED: StepState(),
        HEAD: StepState(
            dataset_id="ds-1",
            run_id="run-1",
            training_job_id="job-1",
            run_status="running",
            deployment_id="dep-1",
            deploy_status="deploying",
        ),
    }
    state.workloads["w2"] = {
        SFT: StepState(
            dataset_id="ds-2",
            run_id="run-2",
            training_job_id="job-2",
            run_status="completed",
            model_version_id="mv-2",
            deployment_id="dep-2",
            deploy_status="paused",
            scored=True,
            agreement={"metric": "exact", "value": 0.9876, "ci95": [0.97, 1.0], "n": 200},
            training_cost_credits=42.0,
        ),
    }
    return state


@pytest.fixture
def audit_dir(tmp_path: Path) -> Path:
    root = tmp_path / "audit"
    save_state(root, _state())
    return root


def test_status_table_and_json(
    run_cli: CliRunner, audit_dir: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")

    def metrics_for(deployment_id: str, *, time_range: str) -> dict[str, object]:
        return {"requests_count": 7 if deployment_id == "dep-1" else "n/a"}

    metrics = mock.Mock(side_effect=metrics_for)
    with mock.patch("dagnam._core.client.DagnamClient.get_deployment_metrics", metrics):
        assert run_cli(["audit", "status", str(audit_dir)]) == 0
    out = re.sub(r" +", " ", capsys.readouterr().out)
    assert metrics.call_args_list == [
        mock.call("dep-1", time_range="7d"),
        mock.call("dep-2", time_range="7d"),
    ]
    assert "Workload Candidate Status Job Deployment Agreement Credits Req/7d" in out
    assert "w1 hosted_floor untested - - - - -" in out
    assert "w1 head_tune deploying job-1 dep-1 - - 7" in out
    assert "w2 sft_small scored job-2 dep-2 0.988 42.0 -" in out
    assert "halted" not in out

    state = load_state(audit_dir)
    state.halted = {"reason": "budget"}
    save_state(audit_dir, state)
    with mock.patch("dagnam._core.client.DagnamClient.get_deployment_metrics", metrics):
        assert run_cli(["audit", "status", str(audit_dir), "--json"]) == 0
    js = json.loads(capsys.readouterr().out)
    assert js["project_id"] == "proj-1"
    assert js["halted"] == {"reason": "budget"}
    assert [(r["workload"], r["candidate"], r["status"]) for r in js["rows"]] == [
        ("w1", "hosted_floor", "untested"),
        ("w1", "head_tune", "deploying"),
        ("w2", "sft_small", "scored"),
    ]
    assert [r["requests_7d"] for r in js["rows"]] == [None, 7, None]
    assert js["rows"][2]["agreement"] == 0.9876


def test_status_reports_no_requests_when_the_deployment_is_gone(
    monkeypatch: PytestMonkeyPatch,
) -> None:
    """A deployment deleted out from under the state file must not crash ``audit status``."""
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    metrics = mock.Mock(side_effect=DeploymentNotFoundError("dep-1"))
    with mock.patch("dagnam._core.client.DagnamClient.get_deployment_metrics", metrics):
        rows = status_rows(_state(), client_from_env())
    assert metrics.call_args_list == [
        mock.call("dep-1", time_range="7d"),
        mock.call("dep-2", time_range="7d"),
    ]
    assert [r["requests_7d"] for r in rows] == [None, None, None]
    assert rows[1]["deployment_id"] == "dep-1"
    assert rows[1]["status"] == "deploying"
    assert rows[1]["training_job_id"] == "job-1"
    assert rows[2]["credits"] == 42.0


def test_status_without_deployments_needs_no_credentials(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.delenv("DAGNAM_API_KEY", raising=False)
    root = tmp_path / "audit"
    assert run_cli(["audit", "status", str(root)]) == 0
    assert "No candidates yet" in capsys.readouterr().out

    state = AuditState()
    state.workloads["w1"] = {HEAD: StepState(dataset_id="ds-1")}
    state.halted = {"reason": "error"}
    save_state(root, state)
    with mock.patch("dagnam._core.client.DagnamClient") as client:
        assert run_cli(["audit", "status", str(root)]) == 0
    assert client.call_count == 0
    out = capsys.readouterr().out
    assert "uploaded" in out
    assert "halted: {'reason': 'error'}" in out


def test_cancel_cancels_running_jobs_pauses_deployments_and_nothing_else(
    run_cli: CliRunner, audit_dir: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch("dagnam._core.client.DagnamClient") as client:
        assert run_cli(["audit", "cancel", str(audit_dir)]) == 0
    assert client.return_value.method_calls == [
        mock.call.cancel_training_job("job-1"),
        mock.call.pause_deployment("dep-1"),
    ]
    out = capsys.readouterr().out
    assert "cancelled_job job-1 (w1/head_tune)" in out
    assert "paused_deployment dep-1 (w1/head_tune)" in out
    assert out.rstrip().endswith("Cancelled.")
    state = load_state(audit_dir)
    assert state.halted == {"reason": "cancelled"}
    head = state.workloads["w1"][HEAD]
    assert (head.run_status, head.deploy_status) == ("cancelled", "paused")

    with mock.patch("dagnam._core.client.DagnamClient") as client:
        assert run_cli(["audit", "cancel", str(audit_dir), "--json"]) == 0
    assert client.return_value.method_calls == []  # idempotent: nothing left in flight
    assert json.loads(capsys.readouterr().out) == {
        "halted": {"reason": "cancelled"},
        "actions": [],
    }


def test_cancel_records_a_deployment_the_platform_refuses_to_pause(
    run_cli: CliRunner, audit_dir: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    """A revision that never activated leaves the deployment unpausable; cancel still finishes."""
    from dagnam._core.exceptions import DeploymentStateError

    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    refusal = DeploymentStateError("Invalid status transition from not_provisioned to paused")
    with mock.patch("dagnam._core.client.DagnamClient") as client:
        client.return_value.pause_deployment.side_effect = refusal
        assert run_cli(["audit", "cancel", str(audit_dir), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "halted": {"reason": "cancelled"},
        "actions": [
            {"candidate": "w1/head_tune", "action": "cancelled_job", "id": "job-1"},
            {
                "candidate": "w1/head_tune",
                "action": "pause_refused",
                "id": "dep-1",
                "reason": "Invalid status transition from not_provisioned to paused",
            },
        ],
    }
    state = load_state(audit_dir)
    assert state.halted == {"reason": "cancelled"}
    head = state.workloads["w1"][HEAD]
    assert (head.run_status, head.deploy_status) == ("cancelled", "deploying")


@pytest.fixture
def cleanup(monkeypatch: PytestMonkeyPatch) -> FakeCleanup:
    fake = FakeCleanup(
        deployment=["dep-1", "dep-2"],
        model=["entry-2"],
        job=["job-1", "job-2"],
        dataset=["ds-1", "ds-2"],
        project=["proj-1"],
    )
    fake.entry_of = {"mv-2": "entry-2"}
    monkeypatch.setattr("dagnam.cli.audit.client_from_env", lambda: fake)
    return fake


def test_delete_asks_first_and_writes_the_receipt(
    run_cli: CliRunner, audit_dir: Path, cleanup: FakeCleanup, capsys: StrCapture
) -> None:
    with mock.patch("builtins.input", return_value="no"), pytest.raises(SystemExit) as exc:
        run_cli(["audit", "delete", str(audit_dir)])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "deployment: dep-1, dep-2" in captured.out
    assert "training_job: job-1, job-2" in captured.out
    assert "project: proj-1" in captured.out
    assert "confirmation not received" in captured.err
    assert cleanup.call_log == []

    assert run_cli(["audit", "delete", str(audit_dir), "--yes"]) == 0
    out = capsys.readouterr().out
    assert "deployment dep-1: deleted" in out
    assert "model_version mv-2: deleted" in out
    assert "project proj-1: deleted" in out
    assert f"Receipt: {audit_dir / 'deleted.json'}" in out
    receipt = json.loads((audit_dir / "deleted.json").read_text(encoding="utf-8"))
    assert receipt["schema"] == "dagnam.audit.deleted/1"
    assert all(not ids for ids in cleanup.present.values())

    assert run_cli(["audit", "delete", str(audit_dir), "--yes", "--json"]) == 0
    again = json.loads(capsys.readouterr().out)
    assert {i["status"] for i in again["items"]} == {"already_absent"}


def test_delete_names_the_reason_an_artifact_is_blocked(
    run_cli: CliRunner, audit_dir: Path, cleanup: FakeCleanup, capsys: StrCapture
) -> None:
    cleanup.running = {"job-1"}  # w1 is still training, so its dataset is still referenced
    cleanup.held_by_job = {"ds-1": "job-1"}

    assert run_cli(["audit", "delete", str(audit_dir), "--yes"]) == 0

    out = capsys.readouterr().out
    assert "Cannot delete job with status running" in out
    assert "dataset ds-1: blocked (Dataset is referenced by a training run" in out
    assert "dataset ds-2: deleted" in out


def test_audit_is_grouped_and_described() -> None:
    from dagnam.cli._parser import ALL_GROUPED_COMMANDS, COMMAND_DESCRIPTIONS, EXAMPLES

    assert "audit" in ALL_GROUPED_COMMANDS
    assert COMMAND_DESCRIPTIONS["audit"].startswith("Audit exported LLM traces")
    assert any("dagnam audit scan" in line for line in EXAMPLES)


# ------------------------------------- a run that published: the server cleans up


CANCEL_RECEIPT: dict[str, object] = {
    "schema": "dagnam.audit.deleted/1",
    "deleted_at": "2026-09-07T10:00:00+00:00",
    "entries": [
        {"kind": "training_job", "id": "job-1", "status": "stopped", "reason": None},
        {"kind": "deployment", "id": "dep-1", "status": "blocked", "reason": "not provisioned"},
    ],
}
DELETE_RECEIPT: dict[str, object] = {
    "schema": "dagnam.audit.deleted/1",
    "deleted_at": "2026-09-07T10:00:00+00:00",
    "entries": [
        {"kind": "deployment", "id": "dep-1", "status": "deleted", "reason": None},
        {"kind": "project", "id": "proj-1", "status": "deleted", "reason": None},
    ],
}


@pytest.fixture
def published_dir(tmp_path: Path) -> Path:
    """An audit dir whose run published, so ``cancel``/``delete`` go through the server."""
    root = tmp_path / "audit"
    state = _state()
    state.audit_id = "audit-1"
    state.workloads["w1"][HEAD].key_ref = "w1/head_tune"
    save_state(root, state)
    (root / "workloads" / "w1").mkdir(parents=True)
    (root / "workloads" / "w1" / "dataset.jsonl").write_text("{}\n", encoding="utf-8")
    return root


def test_cancel_of_a_published_run_goes_to_the_server_and_writes_its_receipt(
    run_cli: CliRunner, published_dir: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    client = mock.Mock()
    client.cancel_audit.return_value = CANCEL_RECEIPT
    monkeypatch.setattr("dagnam.cli.audit.client_from_env", lambda: client)

    assert run_cli(["audit", "cancel", str(published_dir)]) == 0

    assert client.method_calls == [mock.call.cancel_audit("audit-1")]
    out = capsys.readouterr().out
    assert "training_job job-1: stopped" in out
    assert "deployment dep-1: blocked (not provisioned)" in out
    assert f"Receipt: {published_dir / 'cancelled.json'}" in out
    assert (
        json.loads((published_dir / "cancelled.json").read_text(encoding="utf-8")) == CANCEL_RECEIPT
    )
    assert not (published_dir / "deleted.json").exists()  # nothing was deleted
    state = load_state(published_dir)
    assert state.halted == {"reason": "cancelled"}
    # The same local marks the local cancel leaves, so `status` reads the same.
    head, done = state.workloads["w1"][HEAD], state.workloads["w2"][SFT]
    # dep-1 is `blocked` in the receipt: the server could not pause it, so neither
    # does the local state -- exactly what the local cancel path records.
    assert (head.run_status, head.deploy_status) == ("cancelled", "deploying")
    assert head.error == "cancelled: stopped by `dagnam audit cancel`"
    assert (done.run_status, done.deploy_status) == ("completed", "paused")
    assert done.error is None  # it scored before the cancel; nothing to stop

    # ``status`` reads the same client_from_env patched above, so no key is needed.
    assert run_cli(["audit", "status", str(published_dir), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)["rows"]
    assert [(r["candidate"], r["status"], r["run_status"]) for r in rows] == [
        ("hosted_floor", "untested", None),
        ("head_tune", "cancelled", "cancelled"),
        ("sft_small", "scored", "completed"),
    ]


def test_delete_of_a_published_run_names_the_audit_and_drops_the_local_rows(
    run_cli: CliRunner, published_dir: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    client = mock.Mock()
    client.delete_audit.return_value = DELETE_RECEIPT
    monkeypatch.setattr("dagnam.cli.audit.client_from_env", lambda: client)

    with mock.patch("builtins.input", return_value="no"), pytest.raises(SystemExit):
        run_cli(["audit", "delete", str(published_dir)])
    assert "audit: audit-1 (and its published report)" in capsys.readouterr().out
    assert client.method_calls == []

    assert run_cli(["audit", "delete", str(published_dir), "--yes"]) == 0
    assert client.method_calls == [mock.call.delete_audit("audit-1")]
    out = capsys.readouterr().out
    assert "project proj-1: deleted" in out
    assert json.loads((published_dir / "deleted.json").read_text(encoding="utf-8")) == (
        DELETE_RECEIPT
    )
    assert not (published_dir / "workloads").exists()


def test_delete_of_a_published_run_keeps_the_local_rows_when_something_is_blocked(
    run_cli: CliRunner, published_dir: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    client = mock.Mock()
    client.delete_audit.return_value = CANCEL_RECEIPT  # carries a `blocked` entry
    monkeypatch.setattr("dagnam.cli.audit.client_from_env", lambda: client)

    assert run_cli(["audit", "delete", str(published_dir), "--yes"]) == 0

    assert (published_dir / "workloads" / "w1" / "dataset.jsonl").exists()
    assert "deployment dep-1: blocked (not provisioned)" in capsys.readouterr().out


def test_a_cancelled_candidate_is_not_resumed_into_wait_run(published_dir: Path) -> None:
    """`audit run` after a cancel must not poll a job the platform already stopped."""
    from dagnam.audit.cleanup import mark_cancelled
    from dagnam.audit.steps_train import wait_run

    state = load_state(published_dir)
    mark_cancelled(state)
    step = state.workloads["w1"][HEAD]
    assert step.run_status == "cancelled"

    polled: list[str] = []
    ctx = mock.Mock()
    ctx.step.return_value = step
    ctx.client.get_foundation_run.side_effect = lambda run_id: polled.append(run_id)

    wait_run(state, ctx)

    assert polled == []
    assert step.error is not None
    assert step.error.startswith("run_cancelled:")


def test_a_receipt_row_missing_a_field_still_renders(
    run_cli: CliRunner, published_dir: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    """A row the server spells differently must not cost the whole receipt."""
    client = mock.Mock()
    client.cancel_audit.return_value = {
        "schema": "dagnam.audit.deleted/1",
        "entries": [{"kind": "deployment", "id": "dep-1"}, {"id": "job-1", "status": "stopped"}],
    }
    monkeypatch.setattr("dagnam.cli.audit.client_from_env", lambda: client)

    assert run_cli(["audit", "cancel", str(published_dir)]) == 0

    out = capsys.readouterr().out
    assert "deployment dep-1: ?" in out
    assert "? job-1: stopped" in out


def test_status_names_the_published_audit_and_where_to_watch_it(
    run_cli: CliRunner, published_dir: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    monkeypatch.setenv("DAGNAM_API_URL", "https://api.dagnam.ai")
    url = "https://dagnam.ai/audits/audit-1"
    with mock.patch("dagnam._core.client.DagnamClient.get_deployment_metrics", return_value={}):
        assert run_cli(["audit", "status", str(published_dir)]) == 0
    assert f"audit audit-1: {url}" in capsys.readouterr().out

    with mock.patch("dagnam._core.client.DagnamClient.get_deployment_metrics", return_value={}):
        assert run_cli(["audit", "status", str(published_dir), "--json"]) == 0
    js = json.loads(capsys.readouterr().out)
    assert (js["audit_id"], js["url"]) == ("audit-1", url)


def test_status_of_a_local_only_run_names_no_audit(
    run_cli: CliRunner, audit_dir: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch("dagnam._core.client.DagnamClient.get_deployment_metrics", return_value={}):
        assert run_cli(["audit", "status", str(audit_dir), "--json"]) == 0
    js = json.loads(capsys.readouterr().out)
    assert (js["audit_id"], js["url"]) == (None, None)
