"""CLI ``audit cancel``: every live run and endpoint is stopped, and the exit status says what is left."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from tests.audit import _receipts as r
from tests.audit._cleanup import FakeCleanup
from tests.cli._audit_dirs import (
    HEAD,
    SFT,
    cli_state,
    platform_with_everything,
    published_dir,
)

from dagnam._core.exceptions import APIError, DeploymentStateError
from dagnam._types import JsonObject
from dagnam.audit.cleanup import CANCELLED_FILE
from dagnam.audit.state import AuditState, StepState, load_state, lock_audit, save_state

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture

CANCELLED = r.SCHEMA_CANCELLED


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


def _platform(monkeypatch: PytestMonkeyPatch, receipt: JsonObject | None = None) -> FakeCleanup:
    """Every artifact still on the platform; the account answering ``receipt``."""
    fake = platform_with_everything(receipt=receipt)
    monkeypatch.setattr("dagnam.cli.audit_cleanup.client_from_env", lambda: fake)
    return fake


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
    assert "training_job job-1: stopped" in out
    assert "deployment dep-1: stopped" in out
    assert f"Receipt: {audit_dir / CANCELLED_FILE}" in out
    state = load_state(audit_dir)
    assert state.halted == {"reason": "cancelled"}
    head = state.workloads["w1"][HEAD]
    assert (head.run_status, head.deploy_status) == ("cancelled", "paused")

    with mock.patch("dagnam._core.client.DagnamClient") as client:
        assert run_cli(["audit", "cancel", str(audit_dir), "--json"]) == 0
    assert client.return_value.method_calls == []  # idempotent: nothing left in flight
    receipt = json.loads(capsys.readouterr().out)
    assert (receipt["schema"], receipt["entries"]) == (CANCELLED, [])


def test_a_deployment_that_cannot_be_paused_fails_the_cancel_and_is_not_marked(
    run_cli: CliRunner, audit_dir: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    """An unpublished audit has no platform to decide: an endpoint that will not pause may be serving."""
    fake = _platform(monkeypatch)
    fake.unpausable = {"dep-1"}

    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "cancel", str(audit_dir), "--json"])

    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["entries"] == [
        {"kind": "training_job", "id": "job-1", "status": "stopped"},
        {
            "kind": "deployment",
            "id": "dep-1",
            "status": "blocked",
            "code": "not_removed",
            "reason": "Invalid status transition from not_provisioned to paused",
        },
    ]
    state = load_state(audit_dir)
    assert state.halted == {"reason": "cancelled"}
    head = state.workloads["w1"][HEAD]
    assert (head.run_status, head.deploy_status) == ("cancelled", "deploying")


def test_an_endpoint_that_may_still_be_serving_fails_the_cancel_and_says_what_to_run(
    run_cli: CliRunner, audit_dir: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    """The receipt and the marks are written; the exit status says an endpoint was not stopped."""
    fake = _platform(monkeypatch)
    fake.unpausable = {"dep-1"}

    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "cancel", str(audit_dir), "--json"])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert (
        json.loads(captured.out)["entries"][1]["status"] == "blocked"
    )  # stdout is the receipt alone
    err = " ".join(captured.err.split())
    assert "something may still be running or serving, or the platform did not answer" in err
    assert f"dagnam audit cancel {audit_dir}" in err
    assert (audit_dir / CANCELLED_FILE).exists()
    assert load_state(audit_dir).halted == {"reason": "cancelled"}


def test_a_run_that_finished_while_nobody_watched_is_already_stopped_and_stays_resumable(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    """`--no-wait` left job-1 `queued`; the job completed, so the platform answers 400.

    That 400 escaped the command: no receipt, no saved marks, and the live
    endpoint of the next candidate never paused -- on every retry. It is not
    running, so it is not a reason to fail the command either.
    """
    root = tmp_path / "audit"
    state = AuditState(project_id="proj-1")
    state.workloads["w1"] = {HEAD: StepState(training_job_id="job-1", run_status="queued")}
    state.workloads["w2"] = {
        SFT: StepState(
            training_job_id="job-2",
            run_status="completed",
            deployment_id="dep-2",
            deploy_status="running",
            scored=True,
        )
    }
    save_state(root, state)
    fake = _platform(monkeypatch)
    fake.finished = {"job-1"}

    for _ in range(2):  # and running it again changes nothing
        assert run_cli(["audit", "cancel", str(root)]) == 0
    out = capsys.readouterr().out
    assert "training_job job-1: stopped [already_stopped]" in out
    assert "deployment dep-2: stopped" in out
    assert ("pause_deployment", "dep-2") in fake.call_log
    saved = load_state(root)
    assert saved.halted == {"reason": "cancelled"}
    assert saved.workloads["w2"][SFT].deploy_status == "paused"
    head = saved.workloads["w1"][HEAD]
    assert (head.run_status, head.error) == ("queued", None)


def test_every_run_is_stopped_even_when_the_first_stop_fails_and_the_exit_says_so(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    """Three runs are training; the first cancel answers 500. The other two kept billing."""
    root = tmp_path / "audit"
    state = AuditState(project_id="proj-1")
    state.workloads["w1"] = {
        HEAD: StepState(training_job_id="job-w1", run_status="running"),
        SFT: StepState(training_job_id="job-w2", run_status="running"),
    }
    state.workloads["w2"] = {HEAD: StepState(training_job_id="job-w3", run_status="running")}
    save_state(root, state)
    fake = _platform(monkeypatch)
    fake.present["job"] |= {"job-w1", "job-w2", "job-w3"}
    real = fake.cancel_training_job

    def flaky(job_id: str) -> JsonObject:
        if job_id == "job-w1":
            raise APIError(500, "boom")
        return real(job_id)

    monkeypatch.setattr(fake, "cancel_training_job", flaky)

    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "cancel", str(root)])

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "training_job job-w1: blocked [not_removed] (API error 500: boom)" in out
    assert "training_job job-w2: stopped" in out
    assert "training_job job-w3: stopped" in out
    assert (root / CANCELLED_FILE).exists()


def test_cancel_under_a_live_local_only_run_says_to_stop_it_first(
    run_cli: CliRunner, audit_dir: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setattr("dagnam.cli.audit_cleanup.client_from_env", mock.Mock)
    with lock_audit(audit_dir), pytest.raises(SystemExit) as exc:
        run_cli(["audit", "cancel", str(audit_dir)])
    assert exc.value.code == 1
    assert "stop that `dagnam audit run` (Ctrl+C) first" in capsys.readouterr().err


def test_a_mistyped_directory_is_refused_and_never_created(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    """The lock used to `mkdir` whatever path it was given."""
    fake = _platform(monkeypatch)
    typo = tmp_path / "adit"
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "cancel", str(typo)])
    assert exc.value.code == 1
    assert "is not an audit directory" in capsys.readouterr().err
    assert not typo.exists()
    assert fake.call_log == []


def test_cancel_pauses_a_scored_candidates_live_endpoint_exactly_once(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    """A scored candidate keeps its result; its deployment is paused, and only once."""
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    root = tmp_path / "audit"
    state = AuditState(project_id="proj-1")
    state.workloads["w1"] = {
        SFT: StepState(
            training_job_id="job-1",
            run_status="completed",
            deployment_id="dep-1",
            deploy_status="running",
            scored=True,
        )
    }
    save_state(root, state)

    with mock.patch("dagnam._core.client.DagnamClient") as client:
        assert run_cli(["audit", "cancel", str(root)]) == 0
    assert client.return_value.method_calls == [mock.call.pause_deployment("dep-1")]
    step = load_state(root).workloads["w1"][SFT]
    assert (step.run_status, step.deploy_status, step.error) == ("completed", "paused", None)

    with mock.patch("dagnam._core.client.DagnamClient") as client:
        assert run_cli(["audit", "cancel", str(root)]) == 0
    assert client.return_value.method_calls == []  # the state says it is already paused


def test_a_deployment_error_that_is_not_an_api_error_is_still_that_ids_row(
    run_cli: CliRunner, audit_dir: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    """The deployment client types a refused pause as its own error, not an ``APIError``."""
    fake = _platform(monkeypatch)
    fake.statuses["dep-1"] = "running"

    def refuse(_: str) -> JsonObject:
        raise DeploymentStateError("Invalid status transition from paused to paused")

    monkeypatch.setattr(fake, "pause_deployment", refuse)
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "cancel", str(audit_dir)])
    assert exc.value.code == 1  # nothing recorded says it is paused: it may be serving
