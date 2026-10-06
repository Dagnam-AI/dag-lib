"""CLI ``audit delete``: everything the run created is removed, and the exit status says what is left."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from tests.audit._cleanup import FakeCleanup
from tests.cli._audit_dirs import cli_state, platform_with_everything

from dagnam._core.exceptions import APIError
from dagnam.audit.cleanup import DELETED_FILE
from dagnam.audit.state import AuditBusyError, load_state, lock_audit, save_state

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
def cleanup(monkeypatch: PytestMonkeyPatch) -> FakeCleanup:
    fake = platform_with_everything()
    monkeypatch.setattr("dagnam.cli.audit_cleanup.client_from_env", lambda: fake)
    return fake


def _delete_leaving_something(run_cli: CliRunner, *argv: str) -> None:
    """Run ``audit delete`` expecting it to finish and exit 1: something is still there."""
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "delete", *argv])
    assert exc.value.code == 1


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
    assert f"Receipt: {audit_dir / DELETED_FILE}" in out
    receipt = json.loads((audit_dir / DELETED_FILE).read_text(encoding="utf-8"))
    assert receipt["schema"] == "dagnam.audit.deleted/1"
    assert all(not ids for ids in cleanup.present.values())

    assert run_cli(["audit", "delete", str(audit_dir), "--yes", "--json"]) == 0
    again = json.loads(capsys.readouterr().out)
    assert again["entries"] == []  # deleted already: the platform is not asked again


def test_delete_holds_the_directory_from_before_it_lists_what_it_deletes(
    run_cli: CliRunner, audit_dir: Path, cleanup: FakeCleanup
) -> None:
    """The listing the user confirms is the state no run can change until the delete ends."""
    seen: list[bool] = []

    def answer(_prompt: str) -> str:
        try:
            with lock_audit(audit_dir):
                seen.append(False)
        except AuditBusyError:
            seen.append(True)
        return "no"

    with mock.patch("builtins.input", side_effect=answer), pytest.raises(SystemExit):
        run_cli(["audit", "delete", str(audit_dir)])
    assert seen == [True]


def test_a_mistyped_directory_is_refused_and_never_created(
    run_cli: CliRunner, tmp_path: Path, cleanup: FakeCleanup, capsys: StrCapture
) -> None:
    """The lock used to `mkdir` whatever path it was given."""
    typo = tmp_path / "adit"
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "delete", str(typo), "--yes"])
    assert exc.value.code == 1
    assert "is not an audit directory" in capsys.readouterr().err
    assert not typo.exists()
    assert cleanup.call_log == []


def test_delete_waits_for_a_live_run_to_finish(
    run_cli: CliRunner, audit_dir: Path, cleanup: FakeCleanup, capsys: StrCapture
) -> None:
    with lock_audit(audit_dir), pytest.raises(SystemExit) as exc:
        run_cli(["audit", "delete", str(audit_dir), "--yes"])
    assert exc.value.code == 1
    assert "is in use by another `dagnam audit` command" in capsys.readouterr().err
    assert cleanup.call_log == []


def test_delete_names_the_reason_an_artifact_is_blocked_and_exits_nonzero(
    run_cli: CliRunner, audit_dir: Path, cleanup: FakeCleanup, capsys: StrCapture
) -> None:
    """A script must be able to tell "everything is gone" from "something was left behind".

    The delete still finishes and still prints its receipt; the exit status is
    what says an artifact is on the platform yet, whatever kept it there.
    """
    cleanup.present["job"].add("job-9")  # a run this audit never recorded still reads ds-1
    cleanup.held_by_job = {"ds-1": "job-9"}

    _delete_leaving_something(run_cli, str(audit_dir), "--yes")

    captured = capsys.readouterr()
    out = captured.out
    assert "training_job job-1: deleted" in out  # still running: cancelled, then deleted
    assert "dataset ds-1: blocked [not_removed] (Dataset is referenced by a training run" in out
    assert "dataset ds-2: deleted" in out
    assert f"Receipt: {audit_dir / DELETED_FILE}" in out  # the receipt comes first
    err = " ".join(captured.err.split())
    assert "1 artifact is still there (blocked above)" in err
    assert f"dagnam audit delete {audit_dir} --yes" in err
    assert (audit_dir / DELETED_FILE).exists()
    assert not (audit_dir / "workloads").exists()

    cleanup.present["job"].discard("job-9")  # the block is lifted: the retry it names finishes
    assert run_cli(["audit", "delete", str(audit_dir), "--yes"]) == 0


def test_a_delete_blocked_by_a_failure_exits_nonzero_with_the_receipt_alone_on_stdout(
    run_cli: CliRunner, audit_dir: Path, cleanup: FakeCleanup, capsys: StrCapture
) -> None:
    """Under ``--json`` stdout is the receipt and nothing else; the status says it is incomplete."""
    cleanup.dataset_error = APIError(500, "boom")

    _delete_leaving_something(run_cli, str(audit_dir), "--yes", "--json")

    captured = capsys.readouterr()
    receipt = json.loads(captured.out)
    assert [i["id"] for i in receipt["entries"] if i["status"] == "blocked"] == ["ds-1", "ds-2"]
    assert "2 artifacts are still there (blocked above)" in " ".join(captured.err.split())


def test_a_clean_delete_says_everything_is_gone(
    run_cli: CliRunner, audit_dir: Path, cleanup: FakeCleanup, capsys: StrCapture
) -> None:
    assert run_cli(["audit", "delete", str(audit_dir), "--yes"]) == 0
    assert "Everything the audit created is deleted." in capsys.readouterr().err


def test_an_endpoint_that_will_not_go_fails_the_delete_and_names_why(
    run_cli: CliRunner, audit_dir: Path, cleanup: FakeCleanup, capsys: StrCapture
) -> None:
    cleanup.undeletable = {"dep-1"}

    _delete_leaving_something(run_cli, str(audit_dir), "--yes")

    assert "deployment dep-1: blocked [not_removed] (Cannot delete a deployment" in (
        capsys.readouterr().out
    )
    assert load_state(audit_dir).halted is None  # something is left: the next delete asks again
