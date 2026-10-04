"""``audit delete`` of an audit an older dagnam published: the claim is offered before the delete."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from tests.audit import _receipts as r
from tests.audit._cleanup import FakeCleanup
from tests.cli._audit_dirs import platform_with_everything, published_dir

from dagnam._core.exceptions import APIError
from dagnam._types import JsonObject
from dagnam.audit.state import DELETED_STATE, load_state, save_state

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)


@pytest.fixture
def older(tmp_path: Path) -> Path:
    """A published audit whose resources no client ever named to the platform."""
    root = published_dir(tmp_path)
    state = load_state(root)
    state.tagged = False
    save_state(root, state)
    return root


def _platform(monkeypatch: PytestMonkeyPatch, receipt: JsonObject) -> FakeCleanup:
    fake = platform_with_everything(receipt=receipt)
    monkeypatch.setattr("dagnam.cli.audit_cleanup.client_from_env", lambda: fake)
    return fake


def test_the_claim_is_offered_once_and_a_yes_claims_before_the_delete(
    run_cli: CliRunner, older: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    fake = _platform(monkeypatch, r.designed(r.DELETED_ROW))

    with mock.patch("builtins.input", side_effect=["yes", "yes"]):
        assert run_cli(["audit", "delete", str(older)]) == 0

    assert [name for name, _ in fake.call_log][:2] == ["claim_audit_resources", "delete_audit"]
    asked = {(e["kind"], e["id"]) for e in fake.claims[0] if isinstance(e, dict)}
    assert ("dataset", "ds-1") in asked
    assert not any(kind == "model_version" for kind, _ in asked)
    assert "older dagnam" in capsys.readouterr().out
    assert load_state(older).tagged is True


def test_declining_the_claim_still_deletes_and_says_what_the_platform_kept(
    run_cli: CliRunner, older: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    fake = _platform(
        monkeypatch,
        r.designed(r.DELETED_ROW, r.row("dataset", "ds-1", "kept", "not_created_here")),
    )

    with mock.patch("builtins.input", side_effect=["yes", "no"]):
        assert run_cli(["audit", "delete", str(older)]) == 0

    assert fake.claims == []
    err = " ".join(capsys.readouterr().err.split())
    assert "The platform kept dataset ds-1" in err
    assert "cannot prove the audit created it" in err
    assert "removed in the Studio" in err


def test_no_answer_to_the_claim_question_is_a_no(
    run_cli: CliRunner, older: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    fake = _platform(monkeypatch, r.designed(r.DELETED_ROW))
    with mock.patch("builtins.input", side_effect=["yes", EOFError()]):
        assert run_cli(["audit", "delete", str(older)]) == 0
    assert fake.claims == []


def test_with_yes_the_confirmation_covers_the_claim_too(
    run_cli: CliRunner, older: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    fake = _platform(monkeypatch, r.designed(r.DELETED_ROW))
    assert run_cli(["audit", "delete", str(older), "--yes"]) == 0
    assert len(fake.claims) == 1


def test_what_the_platform_refuses_to_claim_is_deleted_here_after_the_audit_is(
    run_cli: CliRunner, older: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    fake = _platform(
        monkeypatch,
        r.designed(r.DELETED_ROW, r.row("dataset", "ds-2", "kept", "not_created_here")),
    )
    fake.claim_refuses = {"ds-2"}

    assert run_cli(["audit", "delete", str(older), "--yes"]) == 0

    assert "ds-2" not in fake.present["dataset"]
    assert ("delete_dataset", "ds-2") in fake.call_log


def test_a_claim_that_fails_as_a_whole_is_said_and_the_delete_goes_on(
    run_cli: CliRunner, older: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    fake = _platform(monkeypatch, r.designed(r.DELETED_ROW))
    fake.claim_error = APIError(500, "boom")

    assert run_cli(["audit", "delete", str(older), "--yes"]) == 0

    err = capsys.readouterr().err
    assert "The claim failed" in err
    assert "going on with the delete" in err
    assert ("delete_audit", "audit-1") in fake.call_log
    assert load_state(older).tagged is False  # asked again if the directory is ever claimed again


def test_a_platform_without_the_claim_route_still_deletes_a_0_15_directory(
    run_cli: CliRunner, older: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    """The claim route answers 404 (a platform that predates it): the delete reaches the audit."""
    fake = _platform(monkeypatch, r.designed(r.DELETED_ROW))
    fake.claim_error = APIError(404, "Not Found")

    assert run_cli(["audit", "delete", str(older), "--yes"]) == 0

    assert [name for name, _ in fake.call_log][:2] == ["claim_audit_resources", "delete_audit"]
    assert load_state(older).halted == DELETED_STATE


def test_a_deleted_directory_is_not_offered_a_claim(
    run_cli: CliRunner, older: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    state = load_state(older)
    state.halted = dict(DELETED_STATE)
    save_state(older, state)
    fake = _platform(monkeypatch, r.designed(r.DELETED_ROW))
    fake.claim_error = APIError(500, "boom")

    assert run_cli(["audit", "delete", str(older), "--yes"]) == 0

    assert fake.claims == []
    assert [name for name, _ in fake.call_log] == []  # a deleted audit is not asked about


def test_an_audit_this_client_published_is_never_offered_a_claim(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    fresh = published_dir(tmp_path)  # tagged
    fake = _platform(monkeypatch, r.designed(r.DELETED_ROW))
    assert run_cli(["audit", "delete", str(fresh), "--yes"]) == 0
    assert fake.claims == []
