"""What ``audit delete`` and ``audit cancel`` say when the platform cannot see the audit or the ids."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING

import pytest
from tests.audit import _receipts as r
from tests.audit._cleanup import FakeCleanup
from tests.cli._audit_dirs import cli_state, platform_with_everything, published_dir

from dagnam._core.exceptions import APIError
from dagnam.audit.state import load_state, save_state

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)


def _platform(monkeypatch: PytestMonkeyPatch, fake: FakeCleanup) -> FakeCleanup:
    monkeypatch.setattr("dagnam.cli.audit_cleanup.client_from_env", lambda: fake)
    return fake


def _fails(run_cli: CliRunner, *argv: str) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli(list(argv))
    assert exc.value.code == 1


def _blind_published(tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> Path:
    fake = platform_with_everything(receipt=r.designed())
    fake.account_error = APIError(404, "not found")
    _platform(monkeypatch, fake)
    return published_dir(tmp_path)


def test_a_404_without_the_flag_says_nothing_changed_and_that_the_flag_is_irreversible(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    published = _blind_published(tmp_path, monkeypatch)

    _fails(run_cli, "audit", "delete", str(published), "--yes")

    err = " ".join(capsys.readouterr().err.split())
    assert "Nothing was changed" in err
    assert "--already-deleted" in err
    assert "irreversibly" in err


def test_a_404_with_the_flag_says_what_is_removed_not_that_nothing_changed(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    published = _blind_published(tmp_path, monkeypatch)

    assert run_cli(["audit", "delete", str(published), "--yes", "--already-deleted"]) == 0

    err = " ".join(capsys.readouterr().err.split())
    assert "Nothing was changed" not in err
    assert "rows and deployment keys are removed now" in err
    assert "cannot be undone" in err
    assert not (published / "workloads").exists()


def test_the_help_of_the_flag_says_it_cannot_be_undone(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "delete", "--help"])
    assert exc.value.code == 0
    assert "Irreversible" in " ".join(capsys.readouterr().out.split())


def test_an_audit_the_platform_did_not_answer_is_not_counted_as_an_artifact_still_there(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    published = _blind_published(tmp_path, monkeypatch)

    _fails(run_cli, "audit", "delete", str(published), "--yes")

    err = " ".join(capsys.readouterr().err.split())
    assert "the platform did not finish the delete" in err
    assert "artifact is still there" not in err
    assert "artifacts are still there" not in err


def _unpublished(tmp_path: Path) -> Path:
    root = tmp_path / "audit"
    save_state(root, cli_state())
    return root


def test_a_walk_that_finds_none_of_the_ids_names_the_flag_in_the_receipt_and_the_error(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    root = _unpublished(tmp_path)
    _platform(monkeypatch, FakeCleanup())  # another account's key: every read is a 404

    _fails(run_cli, "audit", "delete", str(root), "--yes")

    captured = capsys.readouterr()
    assert "--already-deleted" in captured.out  # the receipt row's own reason
    assert "--already-deleted" in " ".join(captured.err.split())
    assert load_state(root).halted is None


def test_a_cancel_that_finds_none_of_the_ids_names_the_flag_too(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    root = _unpublished(tmp_path)
    state = load_state(root)
    state.workloads["w1"][next(iter(state.workloads["w1"]))].run_status = "running"
    save_state(root, state)
    _platform(monkeypatch, FakeCleanup())

    _fails(run_cli, "audit", "cancel", str(root))

    captured = capsys.readouterr()
    assert "--already-deleted" in captured.out
    assert "--already-deleted" in " ".join(captured.err.split())


def test_an_error_with_no_missing_ids_says_nothing_about_the_flag(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    published = published_dir(tmp_path)
    fake = platform_with_everything(receipt=r.designed(r.REFUSED_ROW, status="halted"))
    _platform(monkeypatch, fake)

    _fails(run_cli, "audit", "delete", str(published), "--yes")

    assert "--already-deleted" not in capsys.readouterr().err
