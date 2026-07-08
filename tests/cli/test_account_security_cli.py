"""CLI coverage for `dagnam account change-password` and `dagnam account sessions`."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture

SECRET_OLD = "S3cret-Old-Value"
SECRET_NEW = "S3cret-New-Value"

# ------------------------------------------------------------- change-password


def _getpass_sequence(*values: str):
    it = iter(values)

    def _fake(_prompt: str) -> str:
        return next(it)

    return _fake


def test_change_password_success(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(
        change_password=mock.Mock(return_value={"message": "Password changed successfully"})
    )
    with (
        mock.patch("dagnam.account", fake),
        mock.patch("getpass.getpass", _getpass_sequence(SECRET_OLD, SECRET_NEW, SECRET_NEW)),
    ):
        run_cli(["account", "change-password"])
    fake.change_password.assert_called_once_with(SECRET_OLD, SECRET_NEW)
    out = capsys.readouterr().out
    assert "Password changed successfully." in out


def test_change_password_mismatch_aborts(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(change_password=mock.Mock())
    with (
        mock.patch("dagnam.account", fake),
        mock.patch(
            "getpass.getpass", _getpass_sequence(SECRET_OLD, SECRET_NEW, "totally-different")
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        run_cli(["account", "change-password"])
    assert exc_info.value.code == 1
    assert "do not match" in capsys.readouterr().err
    fake.change_password.assert_not_called()


def test_change_password_empty_current_aborts(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(change_password=mock.Mock())
    with (
        mock.patch("dagnam.account", fake),
        mock.patch("getpass.getpass", _getpass_sequence("")),
        pytest.raises(SystemExit) as exc_info,
    ):
        run_cli(["account", "change-password"])
    assert exc_info.value.code == 1
    assert "Current password cannot be empty" in capsys.readouterr().err
    fake.change_password.assert_not_called()


def test_change_password_empty_new_aborts(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(change_password=mock.Mock())
    with (
        mock.patch("dagnam.account", fake),
        mock.patch("getpass.getpass", _getpass_sequence(SECRET_OLD, "")),
        pytest.raises(SystemExit) as exc_info,
    ):
        run_cli(["account", "change-password"])
    assert exc_info.value.code == 1
    assert "New password cannot be empty" in capsys.readouterr().err
    fake.change_password.assert_not_called()


def test_change_password_apierror_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import AuthError

    fake = SimpleNamespace(
        change_password=mock.Mock(side_effect=AuthError("Current password is incorrect"))
    )
    with (
        mock.patch("dagnam.account", fake),
        mock.patch("getpass.getpass", _getpass_sequence("wrong-current", SECRET_NEW, SECRET_NEW)),
    ):
        assert run_cli(["account", "change-password"]) == 1
    assert "Current password is incorrect" in capsys.readouterr().err


def test_change_password_never_prints_secret(run_cli: CliRunner, capsys: StrCapture) -> None:
    """The whole flow (success path) must never echo either password value."""
    fake = SimpleNamespace(
        change_password=mock.Mock(return_value={"message": "Password changed successfully"})
    )
    with (
        mock.patch("dagnam.account", fake),
        mock.patch("getpass.getpass", _getpass_sequence(SECRET_OLD, SECRET_NEW, SECRET_NEW)),
    ):
        run_cli(["account", "change-password"])
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert SECRET_OLD not in combined
    assert SECRET_NEW not in combined


def test_change_password_never_prints_secret_on_mismatch(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    """The abort path (mismatched confirmation) must also never echo a secret."""
    fake = SimpleNamespace(change_password=mock.Mock())
    with (
        mock.patch("dagnam.account", fake),
        mock.patch(
            "getpass.getpass", _getpass_sequence(SECRET_OLD, SECRET_NEW, "nope-does-not-match")
        ),
        pytest.raises(SystemExit),
    ):
        run_cli(["account", "change-password"])
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert SECRET_OLD not in combined
    assert SECRET_NEW not in combined
    assert "nope-does-not-match" not in combined


def test_change_password_never_prints_secret_on_api_error(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    """The API-error path must also never echo a secret, even via --debug off."""
    from dagnam._core.exceptions import AuthError

    fake = SimpleNamespace(
        change_password=mock.Mock(side_effect=AuthError("Current password is incorrect"))
    )
    with (
        mock.patch("dagnam.account", fake),
        mock.patch("getpass.getpass", _getpass_sequence(SECRET_OLD, SECRET_NEW, SECRET_NEW)),
    ):
        run_cli(["account", "change-password"])
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert SECRET_OLD not in combined
    assert SECRET_NEW not in combined


# ------------------------------------------------------------------ sessions list


def test_sessions_list_table(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = [
        {
            "id": "s1",
            "device_info": {"browser": "Chrome", "os": "macOS"},
            "last_active_at": "2026-05-11T03:01:26",
            "created_at": "2026-05-01T00:00:00",
            "is_current": True,
        },
        {
            "id": "s2",
            "device_info": {},
            "last_active_at": None,
            "created_at": None,
            "is_current": False,
        },
    ]
    fake = SimpleNamespace(list_sessions=mock.Mock(return_value=payload))
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "sessions", "list"])
    fake.list_sessions.assert_called_once_with()
    out = capsys.readouterr().out
    assert "s1" in out
    assert "Chrome, macOS" in out
    assert "s2" in out


def test_sessions_list_empty_table(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(list_sessions=mock.Mock(return_value=[]))
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "sessions", "list"])
    assert "No active sessions." in capsys.readouterr().out


def test_sessions_list_json(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = [{"id": "s1", "device_info": {}, "is_current": False}]
    fake = SimpleNamespace(list_sessions=mock.Mock(return_value=payload))
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "sessions", "list", "--json"])
    assert json.loads(capsys.readouterr().out) == payload


def test_sessions_list_writes_output_file(run_cli: CliRunner, tmp_path: object) -> None:
    from pathlib import Path

    payload = [{"id": "s1", "device_info": {}, "is_current": False}]
    fake = SimpleNamespace(list_sessions=mock.Mock(return_value=payload))
    out_path = Path(str(tmp_path)) / "sessions.json"
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "sessions", "list", "--output", str(out_path)])
    assert json.loads(out_path.read_text()) == payload


def test_sessions_list_apierror_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(list_sessions=mock.Mock(side_effect=APIError(401, "invalid key")))
    with mock.patch("dagnam.account", fake):
        assert run_cli(["account", "sessions", "list"]) == 1
    assert "invalid key" in capsys.readouterr().err


# --------------------------------------------------------------- sessions revoke


def test_sessions_revoke_calls_client(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(revoke_session=mock.Mock(return_value=None))
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "sessions", "revoke", "s1"])
    fake.revoke_session.assert_called_once_with("s1")
    assert "Revoked session s1." in capsys.readouterr().out


def test_sessions_revoke_not_found_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(revoke_session=mock.Mock(side_effect=APIError(404, "Session not found")))
    with mock.patch("dagnam.account", fake):
        assert run_cli(["account", "sessions", "revoke", "ghost"]) == 1
    assert "Session not found" in capsys.readouterr().err


def test_sessions_revoke_empty_id_exits_1_no_http_call(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    fake = SimpleNamespace(revoke_session=mock.Mock(return_value=None))
    with (
        mock.patch("dagnam.account", fake),
        pytest.raises(SystemExit) as exc_info,
    ):
        run_cli(["account", "sessions", "revoke", ""])
    assert exc_info.value.code == 1
    assert "Session id cannot be empty" in capsys.readouterr().err
    fake.revoke_session.assert_not_called()


def test_sessions_revoke_whitespace_id_exits_1_no_http_call(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    fake = SimpleNamespace(revoke_session=mock.Mock(return_value=None))
    with (
        mock.patch("dagnam.account", fake),
        pytest.raises(SystemExit) as exc_info,
    ):
        run_cli(["account", "sessions", "revoke", "   "])
    assert exc_info.value.code == 1
    assert "Session id cannot be empty" in capsys.readouterr().err
    fake.revoke_session.assert_not_called()


# ----------------------------------------------------------- sessions revoke-all


def test_sessions_revoke_all_confirmed(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(
        revoke_all_sessions=mock.Mock(return_value={"message": "All sessions revoked"})
    )
    with (
        mock.patch("dagnam.account", fake),
        mock.patch("builtins.input", return_value="yes"),
    ):
        run_cli(["account", "sessions", "revoke-all"])
    fake.revoke_all_sessions.assert_called_once_with()
    assert "All sessions revoked" in capsys.readouterr().out


def test_sessions_revoke_all_aborted(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(revoke_all_sessions=mock.Mock())
    with (
        mock.patch("dagnam.account", fake),
        mock.patch("builtins.input", return_value="no"),
        pytest.raises(SystemExit) as exc_info,
    ):
        run_cli(["account", "sessions", "revoke-all"])
    assert exc_info.value.code == 1
    assert "confirmation not received" in capsys.readouterr().err
    fake.revoke_all_sessions.assert_not_called()


def test_sessions_revoke_all_yes_flag_bypasses_prompt(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    fake = SimpleNamespace(
        revoke_all_sessions=mock.Mock(return_value={"message": "All sessions revoked"})
    )

    def _boom(_prompt: str = "") -> str:
        raise AssertionError("input() must not be called when --yes is set")

    monkeypatch.setattr("builtins.input", _boom)
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "sessions", "revoke-all", "--yes"])
    fake.revoke_all_sessions.assert_called_once_with()
    assert "All sessions revoked" in capsys.readouterr().out


def test_sessions_revoke_all_json(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = {"message": "All sessions revoked"}
    fake = SimpleNamespace(revoke_all_sessions=mock.Mock(return_value=payload))
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "sessions", "revoke-all", "--yes", "--json"])
    assert json.loads(capsys.readouterr().out) == payload
