"""CLI ``audit status`` (``scan``, ``run``, ``cancel`` and ``delete`` have their own files)."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from tests.cli._audit_dirs import HEAD, cli_state, published_dir

from dagnam._core.exceptions import APIError, DeploymentNotFoundError
from dagnam.audit.state import AuditState, StepState, load_state, save_state
from dagnam.cli.audit import status_rows
from dagnam.cli.audit_run import client_from_env

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)


@pytest.fixture
def audit_dir(tmp_path: Path) -> Path:
    root = tmp_path / "audit"
    save_state(root, cli_state())
    return root


@pytest.fixture
def published(tmp_path: Path) -> Path:
    return published_dir(tmp_path)


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


def test_a_metrics_read_that_fails_does_not_fail_the_status(
    run_cli: CliRunner, audit_dir: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    """The request count is a nicety; the status of the audit is not hostage to it."""
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    broken = mock.Mock(side_effect=APIError(503, "metrics are down"))

    with mock.patch("dagnam._core.client.DagnamClient.get_deployment_metrics", broken):
        assert run_cli(["audit", "status", str(audit_dir), "--json"]) == 0

    rows = json.loads(capsys.readouterr().out)["rows"]
    assert [r["requests_7d"] for r in rows] == [None, None, None]
    assert broken.called


def test_status_lists_the_candidates_a_forced_rescan_retired_and_how_to_stop_them(
    run_cli: CliRunner, audit_dir: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    """Retired candidates used to vanish from the table while their run kept billing."""
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    state = load_state(audit_dir)
    state.retired = [*state.workloads.pop("w1").values(), *state.workloads.pop("w2").values()]
    save_state(audit_dir, state)
    metrics = mock.Mock(return_value={"requests_count": 3})

    with mock.patch("dagnam._core.client.DagnamClient.get_deployment_metrics", metrics):
        assert run_cli(["audit", "status", str(audit_dir)]) == 0
    out = re.sub(r" +", " ", capsys.readouterr().out)
    # A retired endpoint that still takes traffic is exactly what the table is for.
    assert metrics.call_count == 2
    assert "w1 head_tune retired job-1 dep-1 - - 3" in out
    assert "w2 sft_small retired job-2 dep-2 0.988 42.0 3" in out
    assert out.rstrip().splitlines()[-4:] == [
        "warning: 2 retired run(s) or endpoint(s) are still live and still billed:",
        " training_job job-1",
        " deployment dep-1",  # job-2 completed and dep-2 is paused: neither is live
        f"Stop them: dagnam audit cancel {audit_dir}",
    ]

    with mock.patch("dagnam._core.client.DagnamClient.get_deployment_metrics", metrics):
        assert run_cli(["audit", "status", str(audit_dir), "--json"]) == 0
    js = json.loads(capsys.readouterr().out)
    assert [(r["workload"], r["candidate"], r["status"], r["run_status"]) for r in js["rows"]] == [
        # The hosted floor was retired too, but it never left anything on the platform.
        ("w1", "head_tune", "retired", "running"),
        ("w2", "sft_small", "retired", "completed"),
    ]
    assert js["retired_live"] == [
        {"kind": "training_job", "id": "job-1"},
        {"kind": "deployment", "id": "dep-1"},
    ]


def test_status_shows_a_candidate_retired_before_the_state_kept_its_workload(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    """An older state's retired steps do not say which candidate they were: shown, unnamed."""
    state = AuditState(retired=[StepState(training_job_id="job-old", run_status="running")])
    save_state(tmp_path, state)
    assert run_cli(["audit", "status", str(tmp_path), "--json"]) == 0
    [row] = json.loads(capsys.readouterr().out)["rows"]
    assert (row["workload"], row["candidate"], row["status"]) == (None, None, "retired")
    assert row["training_job_id"] == "job-old"


def test_status_reports_no_requests_when_the_deployment_is_gone(
    monkeypatch: PytestMonkeyPatch,
) -> None:
    """A deployment deleted out from under the state file must not crash ``audit status``."""
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    metrics = mock.Mock(side_effect=DeploymentNotFoundError("dep-1"))
    with mock.patch("dagnam._core.client.DagnamClient.get_deployment_metrics", metrics):
        rows = status_rows(cli_state(), client_from_env())
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


def test_audit_is_grouped_and_described() -> None:
    from dagnam.cli._parser import ALL_GROUPED_COMMANDS, COMMAND_DESCRIPTIONS, EXAMPLES

    assert "audit" in ALL_GROUPED_COMMANDS
    assert COMMAND_DESCRIPTIONS["audit"].startswith("Audit exported LLM traces")
    assert any("dagnam audit scan" in line for line in EXAMPLES)


def test_status_names_the_published_audit_and_where_to_watch_it(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    monkeypatch.setenv("DAGNAM_API_URL", "https://api.dagnam.ai")
    url = "https://dagnam.ai/audits/audit-1"
    with mock.patch("dagnam._core.client.DagnamClient.get_deployment_metrics", return_value={}):
        assert run_cli(["audit", "status", str(published)]) == 0
    assert f"audit audit-1: {url}" in capsys.readouterr().out

    with mock.patch("dagnam._core.client.DagnamClient.get_deployment_metrics", return_value={}):
        assert run_cli(["audit", "status", str(published), "--json"]) == 0
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
