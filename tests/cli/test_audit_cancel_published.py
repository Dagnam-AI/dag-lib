"""CLI ``audit cancel`` of a run that published: the platform is asked once, its receipt shown and recorded."""

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
    platform_with_everything,
    published_dir,
)

from dagnam._core.exceptions import APIError
from dagnam._types import JsonObject
from dagnam.audit.cleanup import CANCELLED_ERROR, CANCELLED_FILE, mark_cancelled
from dagnam.audit.state import load_state, lock_audit
from dagnam.audit.steps_train import wait_run

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture

CANCELLED = r.SCHEMA_CANCELLED


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)


@pytest.fixture
def published(tmp_path: Path) -> Path:
    return published_dir(tmp_path)


def _platform(monkeypatch: PytestMonkeyPatch, receipt: JsonObject | None = None) -> FakeCleanup:
    """Every artifact still on the platform; the account answering ``receipt``."""
    fake = platform_with_everything(receipt=receipt)
    monkeypatch.setattr("dagnam.cli.audit_cleanup.client_from_env", lambda: fake)
    return fake


def test_a_cancel_under_a_live_published_run_cancels_in_the_account_only(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    """The live run owns `state.json`; the platform's cancel stops it at its next publish."""
    fake = _platform(
        monkeypatch, r.receipt(r.row("training_job", "job-1", "stopped"), schema=CANCELLED)
    )
    before = (published / "state.json").read_text(encoding="utf-8")
    with lock_audit(published):
        assert run_cli(["audit", "cancel", str(published)]) == 0
    assert fake.call_log == [("cancel_audit", "audit-1")]
    assert "training_job job-1: stopped" in capsys.readouterr().out
    assert (published / "state.json").read_text(encoding="utf-8") == before


def test_cancel_of_a_published_run_goes_to_the_platform_and_writes_its_receipt_as_sent(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    receipt = r.designed(
        r.row("training_job", "job-1", "stopped"),
        r.row("deployment", "dep-1", "stopped", "already_stopped"),
        r.row("deployment", "dep-2", "stopped"),
        status="halted",
        schema=CANCELLED,
    )
    fake = _platform(monkeypatch, receipt)

    assert run_cli(["audit", "cancel", str(published)]) == 0

    assert fake.call_log == [("cancel_audit", "audit-1")]  # no stop of the client's own
    out = capsys.readouterr().out
    assert "training_job job-1: stopped" in out
    assert "deployment dep-1: stopped [already_stopped]" in out
    assert f"Receipt: {published / CANCELLED_FILE}" in out
    assert json.loads((published / CANCELLED_FILE).read_text(encoding="utf-8")) == receipt
    assert not (published / "deleted.json").exists()  # nothing was deleted
    state = load_state(published)
    assert state.halted == {"reason": "cancelled"}
    head, done = state.workloads["w1"][HEAD], state.workloads["w2"][SFT]
    assert (head.run_status, head.deploy_status) == (
        "cancelled",
        "deploying",
    )  # run stopped, endpoint untouched
    assert head.error == CANCELLED_ERROR
    # It scored before the cancel, so it keeps its result -- but not its endpoint.
    assert (done.run_status, done.deploy_status) == ("completed", "paused")
    assert done.error is None

    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch("dagnam._core.client.DagnamClient.get_deployment_metrics", return_value={}):
        assert run_cli(["audit", "status", str(published), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)["rows"]
    assert [(row["candidate"], row["status"], row["run_status"]) for row in rows] == [
        ("hosted_floor", "untested", None),
        ("head_tune", "cancelled", "cancelled"),
        ("sft_small", "scored", "completed"),
    ]


def test_a_run_the_platform_found_already_finished_stays_resumable(
    run_cli: CliRunner, published: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    fake = _platform(
        monkeypatch,
        r.designed(r.ALREADY_STOPPED_ROW, status="halted", schema=CANCELLED),
    )

    assert run_cli(["audit", "cancel", str(published)]) == 0

    head = load_state(published).workloads["w1"][HEAD]
    assert (head.run_status, head.error) == ("running", None)
    assert fake.call_log == [("cancel_audit", "audit-1")]


def test_a_blocked_row_fails_the_cancel_and_nothing_is_stopped_from_here(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    fake = _platform(monkeypatch, r.designed(r.HAS_SERVED_ROW, status="halted", schema=CANCELLED))

    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "cancel", str(published)])

    assert exc.value.code == 1
    assert "deployment dep-1: blocked [has_served]" in capsys.readouterr().out
    assert fake.call_log == [("cancel_audit", "audit-1")]


def test_a_platform_that_does_not_answer_is_one_blocked_row_exit_one_and_nothing_is_touched(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    fake = _platform(monkeypatch)
    fake.account_error = APIError(500, "server error")
    before = (published / "state.json").read_text(encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "cancel", str(published)])

    assert exc.value.code == 1
    assert "audit audit-1: blocked [not_answered]" in capsys.readouterr().out
    assert fake.call_log == [("cancel_audit", "audit-1")]
    assert (published / "state.json").read_text(encoding="utf-8") == before


def test_a_failed_cancel_under_a_live_run_is_an_error_that_says_what_to_do(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    fake = _platform(monkeypatch)
    fake.account_error = APIError(500, "server error")
    with lock_audit(published), pytest.raises(SystemExit) as exc:
        run_cli(["audit", "cancel", str(published)])
    assert exc.value.code == 1
    err = " ".join(capsys.readouterr().err.split())
    assert "the platform did not answer the cancel (API error 500: server error)" in err
    assert "stop that `dagnam audit run` (Ctrl+C), then cancel again" in err


def test_a_404_is_not_a_decision_it_prints_the_account_keeps_everything_and_exits_one(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "dk-secret-0000")
    monkeypatch.setenv("DAGNAM_API_URL", "https://api.example.test")
    fake = _platform(monkeypatch)
    fake.account_error = APIError(404, "not found")
    before = (published / "state.json").read_text(encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "cancel", str(published)])

    assert exc.value.code == 1
    err = " ".join(capsys.readouterr().err.split())
    assert "https://api.example.test" in err
    assert "dk-secret-0000" not in err  # the key is masked
    assert "may belong to another account" in err
    assert (published / "state.json").read_text(encoding="utf-8") == before
    assert fake.call_log == [("cancel_audit", "audit-1")]

    fake.account_error = None  # the right key, afterwards, still cancels
    assert run_cli(["audit", "cancel", str(published)]) == 0


def test_a_404_under_a_live_run_says_what_to_do(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    fake = _platform(monkeypatch)
    fake.account_error = APIError(404, "not found")
    with lock_audit(published), pytest.raises(SystemExit) as exc:
        run_cli(["audit", "cancel", str(published)])
    assert exc.value.code == 1
    err = " ".join(capsys.readouterr().err.split())
    assert "no audit with this id for this key" in err
    assert "stop that `dagnam audit run` (Ctrl+C), then cancel again" in err


def test_a_deleted_audit_refuses_a_cancel(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    fake = _platform(monkeypatch, r.designed(r.DELETED_ROW))
    assert run_cli(["audit", "delete", str(published), "--yes"]) == 0
    fake.call_log.clear()

    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "cancel", str(published)])

    assert exc.value.code == 1
    assert "is a deleted audit" in capsys.readouterr().err
    assert fake.call_log == []


def test_what_the_platform_kept_is_shown_and_does_not_fail_the_cancel(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    kept = r.row("training_job", "job-1", "kept", "not_in_project", r.NOT_IN_PROJECT)
    fake = _platform(
        monkeypatch,
        r.designed(
            kept, r.row("deployment", "dep-2", "stopped"), status="halted", schema=CANCELLED
        ),
    )

    assert run_cli(["audit", "cancel", str(published)]) == 0

    assert "training_job job-1: kept [not_in_project] (not in this audit's project" in (
        capsys.readouterr().out
    )
    assert ("cancel_training_job", "job-1") not in fake.call_log


def test_a_row_from_a_newer_platform_is_shown_never_acted_on_and_fails_the_cancel(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    odd = r.row("deployment", "dep-1", "quarantined", "needs_a_human", "ask somebody")
    fake = _platform(monkeypatch, r.receipt(odd, schema=CANCELLED))

    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "cancel", str(published)])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "deployment dep-1: quarantined [needs_a_human] (ask somebody)" in captured.out
    assert "1 receipt row has a status this version of dagnam does not know" in captured.err
    assert fake.call_log == [("cancel_audit", "audit-1")]


def test_a_receipt_row_missing_a_field_still_renders(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    """A row the server spells differently must not cost the whole receipt."""
    client = mock.Mock()
    client.cancel_audit.return_value = {
        "schema": "dagnam.audit.deleted/1",
        "entries": [{"kind": "deployment", "id": "dep-1"}, {"id": "job-1", "status": "stopped"}],
    }
    monkeypatch.setattr("dagnam.cli.audit_cleanup.client_from_env", lambda: client)

    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "cancel", str(published)])

    assert exc.value.code == 1  # a row it cannot read is not a promise it can make
    captured = capsys.readouterr()
    assert "deployment dep-1: ?" in captured.out
    assert "? job-1: stopped" in captured.out
    assert "1 receipt row has a status this version" in captured.err


def test_a_cancelled_candidate_is_not_resumed_into_wait_run(published: Path) -> None:
    """`audit run` after a cancel must not poll a job the platform already stopped."""
    state = load_state(published)
    mark_cancelled(state, {"job-1"})
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
