"""CLI ``audit delete`` of a run that published: the platform's receipt decides and nothing else is touched."""

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
    DELETE_RECEIPT,
    kept_on_purpose_receipt,
    platform_with_everything,
    published_dir,
)

from dagnam._core.exceptions import APIError
from dagnam._types import JsonObject
from dagnam.audit.cleanup import DELETED_FILE
from dagnam.audit.state import load_state

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)


@pytest.fixture
def published(tmp_path: Path) -> Path:
    return published_dir(tmp_path)


def _platform(monkeypatch: PytestMonkeyPatch, receipt: JsonObject) -> FakeCleanup:
    """Every artifact still on the platform; the account answering ``receipt``."""
    fake = platform_with_everything(receipt=receipt)
    monkeypatch.setattr("dagnam.cli.audit_cleanup.client_from_env", lambda: fake)
    return fake


def _finish_with_leftovers(run_cli: CliRunner, *argv: str) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "delete", *argv])
    assert exc.value.code == 1


def test_delete_of_a_published_run_names_the_audit_asks_once_and_drops_the_local_rows(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    fake = _platform(monkeypatch, r.designed(*r.receipt_rows_of(DELETE_RECEIPT)))

    with mock.patch("builtins.input", return_value="no"), pytest.raises(SystemExit):
        run_cli(["audit", "delete", str(published)])
    assert "audit: audit-1 (and its published report)" in capsys.readouterr().out
    assert fake.call_log == []

    assert run_cli(["audit", "delete", str(published), "--yes"]) == 0
    assert fake.call_log == [("delete_audit", "audit-1")]  # nothing else of the client's own
    out = capsys.readouterr().out
    assert "project proj-1: deleted" in out
    receipt = json.loads((published / DELETED_FILE).read_text(encoding="utf-8"))
    assert receipt["entries"] == DELETE_RECEIPT["entries"]
    assert not (published / "workloads").exists()
    assert load_state(published).halted == {"reason": "deleted"}


def test_a_platform_that_halted_the_delete_leaves_everything_local_and_exits_one(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    fake = _platform(
        monkeypatch,
        r.designed(r.row("deployment", "dep-1", "deleted"), r.REFUSED_ROW, status="halted"),
    )

    _finish_with_leftovers(run_cli, str(published), "--yes")

    captured = capsys.readouterr()
    assert "dataset ds-1: blocked [refused] (Dataset is referenced" in captured.out
    assert "1 artifact is still there (blocked above)" in " ".join(captured.err.split())
    assert fake.call_log == [("delete_audit", "audit-1")]
    assert (published / "workloads" / "w1" / "dataset.jsonl").exists()
    assert load_state(published).halted is None
    assert (published / DELETED_FILE).exists()


def test_a_halted_delete_with_no_blocked_row_still_exits_one_and_says_who_is_not_done(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    _platform(monkeypatch, r.designed(r.row("deployment", "dep-1", "deleted"), status="halted"))

    _finish_with_leftovers(run_cli, str(published), "--yes")

    assert "the platform did not finish the delete" in " ".join(capsys.readouterr().err.split())


def test_what_the_platform_keeps_on_purpose_is_not_deleted_and_is_not_a_failed_delete(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    """A project with the owner's other work, weights another deployment serves: not the audit's."""
    fake = _platform(monkeypatch, r.designed(*r.receipt_rows_of(kept_on_purpose_receipt())))

    assert run_cli(["audit", "delete", str(published), "--yes"]) == 0

    assert fake.call_log == [("delete_audit", "audit-1")]
    assert fake.present["project"] == {"proj-1"}
    assert fake.present["model"] == {"mv-2"}
    captured = capsys.readouterr()
    assert f"project proj-1: kept [project_held] ({r.PROJECT_HELD})" in captured.out
    assert f"model_version mv-2: kept [weights_served] ({r.WEIGHTS_SERVED})" in captured.out
    last = " ".join(captured.err.split())
    assert "Everything the audit created is deleted, except 3 items kept on purpose" in last
    state = load_state(published)
    assert state.kept_ids == ["proj-1", "entry-2", "mv-2"]
    written = json.loads((published / DELETED_FILE).read_text(encoding="utf-8"))
    assert [
        f"{row['kind']} {row['id']}" for row in written["entries"] if row["status"] == "kept"
    ] == ["project proj-1", "model_entry entry-2", "model_version mv-2"]


def test_a_resource_of_the_audit_left_behind_still_fails_beside_what_was_kept(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    """Kept rows are not counted, and do not hide the audit's own that is left."""
    rows = r.receipt_rows_of(kept_on_purpose_receipt())
    _platform(
        monkeypatch,
        r.designed(*rows, r.REFUSED_ROW, r.FAILED_ROW, status="halted"),
    )

    _finish_with_leftovers(run_cli, str(published), "--yes")

    err = " ".join(capsys.readouterr().err.split())
    assert "2 artifacts are still there" in err
    assert "3 more were kept on purpose" in err


def test_a_platform_without_codes_keeps_what_its_four_reasons_say_it_kept(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    kept = r.legacy(r.row("project", "proj-1", "blocked", None, r.NOT_OURS))
    fake = _platform(monkeypatch, r.receipt(r.legacy(r.DELETED_ROW), kept))

    assert run_cli(["audit", "delete", str(published), "--yes"]) == 0

    assert fake.call_log == [("delete_audit", "audit-1")]
    assert "1 item kept on purpose" in " ".join(capsys.readouterr().err.split())


def test_a_code_from_a_newer_platform_is_shown_and_only_a_blocked_status_fails(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    odd = r.row("dataset", "ds-1", "kept", "a_code_from_the_future", "something new")
    _platform(monkeypatch, r.designed(r.row("deployment", "dep-1", "deleted"), odd))

    assert run_cli(["audit", "delete", str(published), "--yes"]) == 0

    assert "dataset ds-1: kept [a_code_from_the_future] (something new)" in capsys.readouterr().out


def test_a_status_from_a_newer_platform_is_shown_never_acted_on_and_exits_one(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    odd = r.row("dataset", "ds-1", "quarantined", "x", "ask somebody")
    fake = _platform(monkeypatch, r.designed(odd))

    _finish_with_leftovers(run_cli, str(published), "--yes")

    captured = capsys.readouterr()
    assert "dataset ds-1: quarantined [x] (ask somebody)" in captured.out
    assert "1 receipt row has a status this version of dagnam does not know" in captured.err
    assert fake.call_log == [("delete_audit", "audit-1")]


def test_a_404_is_not_a_decision_it_names_the_account_and_keeps_every_local_file(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "dk-secret-0000")
    monkeypatch.setenv("DAGNAM_API_URL", "https://api.example.test")
    fake = _platform(monkeypatch, r.designed())
    fake.account_error = APIError(404, "not found")

    _finish_with_leftovers(run_cli, str(published), "--yes")

    captured = capsys.readouterr()
    err = " ".join(captured.err.split())
    assert "https://api.example.test" in err
    assert "dk-secret-0000" not in err
    assert "may belong to another account" in err
    assert "Everything the audit created is deleted" not in err
    assert fake.call_log == [("delete_audit", "audit-1")]
    assert (published / "workloads" / "w1" / "dataset.jsonl").exists()
    assert load_state(published).halted is None

    fake.account_error = None  # the right key, afterwards, deletes it
    fake.server_receipt = r.designed(r.DELETED_ROW)
    assert run_cli(["audit", "delete", str(published), "--yes"]) == 0
    assert load_state(published).halted == {"reason": "deleted"}


def test_already_deleted_is_the_callers_word_that_a_404_means_gone(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    fake = _platform(monkeypatch, r.designed())
    fake.account_error = APIError(404, "not found")

    assert run_cli(["audit", "delete", str(published), "--yes", "--already-deleted"]) == 0

    assert fake.call_log == [("delete_audit", "audit-1")]
    assert not (published / "workloads").exists()
    assert (published / "state.json").exists()
    assert load_state(published).halted == {"reason": "deleted"}
    assert "Everything the audit created is deleted." in capsys.readouterr().err


def test_a_run_of_an_audit_deleted_this_way_is_refused(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    fake = _platform(monkeypatch, r.designed(r.DELETED_ROW))
    assert run_cli(["audit", "delete", str(published), "--yes"]) == 0
    fake.call_log.clear()

    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(published), "--yes"])

    assert exc.value.code == 1
    assert "is a deleted audit" in capsys.readouterr().err
    assert fake.call_log == []


def test_when_the_platform_does_not_answer_nothing_is_touched_and_the_exit_says_so(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    fake = _platform(monkeypatch, r.designed())
    fake.account_error = APIError(500, "server error")

    _finish_with_leftovers(run_cli, str(published), "--yes")

    captured = capsys.readouterr()
    assert "audit audit-1: blocked [not_answered]" in captured.out
    assert fake.call_log == [("delete_audit", "audit-1")]
    assert (published / "workloads" / "w1" / "dataset.jsonl").exists()
    assert load_state(published).halted is None
    assert "run it again" in captured.out


def test_an_older_platforms_leftover_fails_the_delete_and_nothing_is_retried_from_here(
    run_cli: CliRunner, published: Path, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    left = r.legacy(r.row("dataset", "ds-1", "blocked", None, "Cannot delete: busy"))
    fake = _platform(monkeypatch, r.receipt(left))

    _finish_with_leftovers(run_cli, str(published), "--yes")

    assert fake.call_log == [("delete_audit", "audit-1")]
    assert "1 artifact is still there (blocked above)" in " ".join(capsys.readouterr().err.split())
    assert (published / "workloads" / "w1" / "dataset.jsonl").exists()
    assert load_state(published).halted is None
