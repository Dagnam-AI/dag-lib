"""CLI coverage for `dagnam account 2fa`.

Two properties get more attention than the happy paths, because both are the
kind that fail silently: the password is never echoed and never reaches argv,
and enrollment material is printed in full exactly once (a caller who does not
see their backup codes has lost them).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture

PASSWORD = "S3cret-Account-Value"
SECRET = "JBSWY3DPEHPK3PXP"
QR_URI = f"otpauth://totp/Dagnam:me?secret={SECRET}"
BACKUP_CODES = ["11111111", "22222222"]
ENROLLMENT = {"secret": SECRET, "qr_code_uri": QR_URI, "backup_codes": BACKUP_CODES}


def _getpass(value: str):
    def _fake(_prompt: str) -> str:
        return value

    return _fake


# ------------------------------------------------------------------- status


@pytest.mark.parametrize(("enabled", "expected"), [(True, "ENABLED"), (False, "DISABLED")])
def test_status_reports_both_states(
    run_cli: CliRunner, capsys: StrCapture, enabled: bool, expected: str
) -> None:
    fake = SimpleNamespace(two_factor_enabled=mock.Mock(return_value=enabled))
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "2fa", "status"])
    assert expected in capsys.readouterr().out


def test_status_json(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(two_factor_enabled=mock.Mock(return_value=True))
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "2fa", "status", "--json"])
    assert json.loads(capsys.readouterr().out) == {"two_factor_enabled": True}


def test_status_output_file(run_cli: CliRunner, tmp_path) -> None:
    fake = SimpleNamespace(two_factor_enabled=mock.Mock(return_value=False))
    out_path = tmp_path / "status.json"
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "2fa", "status", "--output", str(out_path)])
    assert json.loads(out_path.read_text()) == {"two_factor_enabled": False}


def test_status_apierror_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import AuthError

    fake = SimpleNamespace(two_factor_enabled=mock.Mock(side_effect=AuthError("Not authenticated")))
    with mock.patch("dagnam.account", fake):
        assert run_cli(["account", "2fa", "status"]) == 1
    assert "Not authenticated" in capsys.readouterr().err


# ------------------------------------------------------------------- enable


def test_enable_prints_every_piece_of_enrollment_material(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    """The server never returns these again, so anything omitted here is lost."""
    fake = SimpleNamespace(enable_two_factor=mock.Mock(return_value=ENROLLMENT))
    with mock.patch("dagnam.account", fake), mock.patch("getpass.getpass", _getpass(PASSWORD)):
        run_cli(["account", "2fa", "enable"])

    out = capsys.readouterr().out
    assert SECRET in out
    assert QR_URI in out
    for code in BACKUP_CODES:
        assert code in out
    fake.enable_two_factor.assert_called_once_with(PASSWORD)


def test_enable_says_2fa_is_not_active_yet(run_cli: CliRunner, capsys: StrCapture) -> None:
    """Enrollment without verification leaves 2FA off. A caller who believes
    otherwise stops here and is not protected."""
    fake = SimpleNamespace(enable_two_factor=mock.Mock(return_value=ENROLLMENT))
    with mock.patch("dagnam.account", fake), mock.patch("getpass.getpass", _getpass(PASSWORD)):
        run_cli(["account", "2fa", "enable"])

    out = capsys.readouterr().out
    assert "NOT yet active" in out
    assert "verify" in out


def test_enable_never_echoes_the_password(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(enable_two_factor=mock.Mock(return_value=ENROLLMENT))
    with mock.patch("dagnam.account", fake), mock.patch("getpass.getpass", _getpass(PASSWORD)):
        run_cli(["account", "2fa", "enable"])

    captured = capsys.readouterr()
    assert PASSWORD not in captured.out
    assert PASSWORD not in captured.err


def test_enable_empty_password_aborts_without_calling(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    fake = SimpleNamespace(enable_two_factor=mock.Mock())
    with (
        mock.patch("dagnam.account", fake),
        mock.patch("getpass.getpass", _getpass("")),
        pytest.raises(SystemExit) as exc_info,
    ):
        run_cli(["account", "2fa", "enable"])

    assert exc_info.value.code == 1
    assert "Password cannot be empty" in capsys.readouterr().err
    fake.enable_two_factor.assert_not_called()


def test_enable_apierror_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(
        enable_two_factor=mock.Mock(side_effect=APIError(400, "2FA is already enabled"))
    )
    with mock.patch("dagnam.account", fake), mock.patch("getpass.getpass", _getpass(PASSWORD)):
        assert run_cli(["account", "2fa", "enable"]) == 1
    assert "already enabled" in capsys.readouterr().err


def test_enable_json(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(enable_two_factor=mock.Mock(return_value=ENROLLMENT))
    with mock.patch("dagnam.account", fake), mock.patch("getpass.getpass", _getpass(PASSWORD)):
        run_cli(["account", "2fa", "enable", "--json"])

    assert json.loads(capsys.readouterr().out) == ENROLLMENT


def test_enable_output_file_saves_the_material(run_cli: CliRunner, tmp_path) -> None:
    """`--output` is the point of the flag here: the codes must be saved before
    the terminal scrolls."""
    fake = SimpleNamespace(enable_two_factor=mock.Mock(return_value=ENROLLMENT))
    out_path = tmp_path / "enrollment.json"
    with mock.patch("dagnam.account", fake), mock.patch("getpass.getpass", _getpass(PASSWORD)):
        run_cli(["account", "2fa", "enable", "--output", str(out_path)])

    assert json.loads(out_path.read_text()) == ENROLLMENT


def test_enable_renders_a_non_object_response_without_crashing(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    fake = SimpleNamespace(enable_two_factor=mock.Mock(return_value="unexpected"))
    with mock.patch("dagnam.account", fake), mock.patch("getpass.getpass", _getpass(PASSWORD)):
        run_cli(["account", "2fa", "enable"])
    assert "unexpected" in capsys.readouterr().out


def test_enable_omits_optional_fields_the_server_did_not_send(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    fake = SimpleNamespace(enable_two_factor=mock.Mock(return_value={"secret": "S"}))
    with mock.patch("dagnam.account", fake), mock.patch("getpass.getpass", _getpass(PASSWORD)):
        run_cli(["account", "2fa", "enable"])

    out = capsys.readouterr().out
    assert "Secret:  S" in out
    assert "QR URI" not in out
    assert "Backup codes" not in out


def test_enable_omits_an_empty_backup_code_list(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(
        enable_two_factor=mock.Mock(return_value={"secret": "S", "backup_codes": []})
    )
    with mock.patch("dagnam.account", fake), mock.patch("getpass.getpass", _getpass(PASSWORD)):
        run_cli(["account", "2fa", "enable"])
    assert "Backup codes" not in capsys.readouterr().out


# ------------------------------------------------------------------- verify


def test_verify_accepts_the_code_as_an_argument(run_cli: CliRunner, capsys: StrCapture) -> None:
    """A TOTP code is single-use and expires in seconds, so argv is acceptable
    here in a way it never is for the password -- and it is what makes the
    command scriptable."""
    fake = SimpleNamespace(verify_two_factor=mock.Mock(return_value={"message": "ok"}))
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "2fa", "verify", "123456"])

    fake.verify_two_factor.assert_called_once_with("123456")
    assert "now enabled" in capsys.readouterr().out


def test_verify_prompts_when_the_code_is_omitted(
    run_cli: CliRunner, monkeypatch: PytestMonkeyPatch
) -> None:
    fake = SimpleNamespace(verify_two_factor=mock.Mock(return_value={"message": "ok"}))
    monkeypatch.setattr("builtins.input", lambda *_a: "  654321  ")
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "2fa", "verify"])

    fake.verify_two_factor.assert_called_once_with("654321")


def test_verify_empty_code_aborts_without_calling(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    fake = SimpleNamespace(verify_two_factor=mock.Mock())
    monkeypatch.setattr("builtins.input", lambda *_a: "   ")
    with mock.patch("dagnam.account", fake), pytest.raises(SystemExit) as exc_info:
        run_cli(["account", "2fa", "verify"])

    assert exc_info.value.code == 1
    assert "code cannot be empty" in capsys.readouterr().err
    fake.verify_two_factor.assert_not_called()


def test_a_rejected_code_exits_1_and_never_claims_success(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(
        verify_two_factor=mock.Mock(side_effect=APIError(400, "Invalid verification code"))
    )
    with mock.patch("dagnam.account", fake):
        assert run_cli(["account", "2fa", "verify", "000000"]) == 1

    captured = capsys.readouterr()
    assert "Invalid verification code" in captured.err
    assert "now enabled" not in captured.out


# ------------------------------------------------------------------ disable


def test_disable_requires_a_typed_confirmation(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    """Removing a security factor is gated like any destructive action: a stray
    keypress must not be able to confirm it."""
    fake = SimpleNamespace(disable_two_factor=mock.Mock())
    monkeypatch.setattr("builtins.input", lambda *_a: "y")
    with (
        mock.patch("dagnam.account", fake),
        mock.patch("getpass.getpass", _getpass(PASSWORD)),
        pytest.raises(SystemExit) as exc_info,
    ):
        run_cli(["account", "2fa", "disable"])

    assert exc_info.value.code == 1
    fake.disable_two_factor.assert_not_called()


def test_disable_proceeds_on_an_exact_yes(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    fake = SimpleNamespace(disable_two_factor=mock.Mock(return_value={"message": "off"}))
    monkeypatch.setattr("builtins.input", lambda *_a: "yes")
    with mock.patch("dagnam.account", fake), mock.patch("getpass.getpass", _getpass(PASSWORD)):
        run_cli(["account", "2fa", "disable"])

    fake.disable_two_factor.assert_called_once_with(PASSWORD)
    assert "disabled" in capsys.readouterr().out


def test_disable_yes_flag_skips_the_prompt(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(disable_two_factor=mock.Mock(return_value={"message": "off"}))
    with mock.patch("dagnam.account", fake), mock.patch("getpass.getpass", _getpass(PASSWORD)):
        run_cli(["account", "2fa", "disable", "--yes"])

    fake.disable_two_factor.assert_called_once_with(PASSWORD)


def test_disable_empty_password_aborts_without_calling(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    fake = SimpleNamespace(disable_two_factor=mock.Mock())
    with (
        mock.patch("dagnam.account", fake),
        mock.patch("getpass.getpass", _getpass("")),
        pytest.raises(SystemExit) as exc_info,
    ):
        run_cli(["account", "2fa", "disable", "--yes"])

    assert exc_info.value.code == 1
    assert "Password cannot be empty" in capsys.readouterr().err
    fake.disable_two_factor.assert_not_called()


def test_disable_never_echoes_the_password(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(disable_two_factor=mock.Mock(return_value={"message": "off"}))
    with mock.patch("dagnam.account", fake), mock.patch("getpass.getpass", _getpass(PASSWORD)):
        run_cli(["account", "2fa", "disable", "--yes"])

    captured = capsys.readouterr()
    assert PASSWORD not in captured.out
    assert PASSWORD not in captured.err


def test_disable_apierror_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(
        disable_two_factor=mock.Mock(side_effect=APIError(400, "2FA is not enabled"))
    )
    with mock.patch("dagnam.account", fake), mock.patch("getpass.getpass", _getpass(PASSWORD)):
        assert run_cli(["account", "2fa", "disable", "--yes"]) == 1
    assert "not enabled" in capsys.readouterr().err
