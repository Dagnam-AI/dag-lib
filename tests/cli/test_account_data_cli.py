"""CLI coverage for `dagnam account export` and `dagnam account delete`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture

SECRET_PASSWORD = "S3cret-Current-Value"

# ------------------------------------------------------------------------ export


def test_export_success_default_out(run_cli: CliRunner, capsys: StrCapture) -> None:
    export_meta = {"export_id": "exp-1", "status": "pending"}
    fake = SimpleNamespace(
        export_data=mock.Mock(return_value=export_meta),
        download_export=mock.Mock(return_value=Path("./dagnam_export_u1.zip")),
    )
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "export"])
    fake.export_data.assert_called_once_with()
    fake.download_export.assert_called_once_with("exp-1", out=".")
    out = capsys.readouterr().out
    assert "Saved export to" in out


def test_export_success_explicit_out(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    export_meta = {"export_id": "exp-1", "status": "pending"}
    dest = tmp_path / "dagnam_export_u1.zip"
    fake = SimpleNamespace(
        export_data=mock.Mock(return_value=export_meta),
        download_export=mock.Mock(return_value=dest),
    )
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "export", "--out", str(tmp_path)])
    fake.download_export.assert_called_once_with("exp-1", out=str(tmp_path))
    out = capsys.readouterr().out
    assert str(dest) in out


def test_export_apierror_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(
        export_data=mock.Mock(side_effect=APIError(401, "invalid key")),
        download_export=mock.Mock(),
    )
    with mock.patch("dagnam.account", fake):
        assert run_cli(["account", "export"]) == 1
    assert "invalid key" in capsys.readouterr().err
    fake.download_export.assert_not_called()


def test_export_download_404_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(
        export_data=mock.Mock(return_value={"export_id": "exp-1"}),
        download_export=mock.Mock(side_effect=APIError(404, "not found or expired")),
    )
    with mock.patch("dagnam.account", fake):
        assert run_cli(["account", "export"]) == 1
    assert "not found or expired" in capsys.readouterr().err


# ------------------------------------------------------------------------ delete


def test_delete_confirmed(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(delete_account=mock.Mock(return_value={"message": "Account deleted"}))
    with (
        mock.patch("dagnam.account", fake),
        mock.patch("getpass.getpass", return_value=SECRET_PASSWORD),
        mock.patch("builtins.input", return_value="yes"),
    ):
        run_cli(["account", "delete"])
    fake.delete_account.assert_called_once_with(SECRET_PASSWORD)
    assert "Account deleted." in capsys.readouterr().out


def test_delete_aborted_on_typed_no(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(delete_account=mock.Mock())
    with (
        mock.patch("dagnam.account", fake),
        mock.patch("getpass.getpass", return_value=SECRET_PASSWORD),
        mock.patch("builtins.input", return_value="no"),
        pytest.raises(SystemExit) as exc_info,
    ):
        run_cli(["account", "delete"])
    assert exc_info.value.code == 1
    assert "confirmation not received" in capsys.readouterr().err
    fake.delete_account.assert_not_called()


def test_delete_yes_flag_bypasses_prompt(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    fake = SimpleNamespace(delete_account=mock.Mock(return_value={"message": "Account deleted"}))

    def _boom(_prompt: str = "") -> str:
        raise AssertionError("input() must not be called when --yes is set")

    monkeypatch.setattr("builtins.input", _boom)
    with (
        mock.patch("dagnam.account", fake),
        mock.patch("getpass.getpass", return_value=SECRET_PASSWORD),
    ):
        run_cli(["account", "delete", "--yes"])
    fake.delete_account.assert_called_once_with(SECRET_PASSWORD)
    assert "Account deleted." in capsys.readouterr().out


def test_delete_empty_password_aborts(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(delete_account=mock.Mock())
    with (
        mock.patch("dagnam.account", fake),
        mock.patch("getpass.getpass", return_value=""),
        pytest.raises(SystemExit) as exc_info,
    ):
        run_cli(["account", "delete"])
    assert exc_info.value.code == 1
    assert "Password cannot be empty" in capsys.readouterr().err
    fake.delete_account.assert_not_called()


def test_delete_apierror_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import AuthError

    fake = SimpleNamespace(delete_account=mock.Mock(side_effect=AuthError("Incorrect password")))
    with (
        mock.patch("dagnam.account", fake),
        mock.patch("getpass.getpass", return_value=SECRET_PASSWORD),
        mock.patch("builtins.input", return_value="yes"),
    ):
        assert run_cli(["account", "delete"]) == 1
    assert "Incorrect password" in capsys.readouterr().err


def test_delete_never_prints_secret(run_cli: CliRunner, capsys: StrCapture) -> None:
    """The whole flow (success path) must never echo the password value."""
    fake = SimpleNamespace(delete_account=mock.Mock(return_value={"message": "Account deleted"}))
    with (
        mock.patch("dagnam.account", fake),
        mock.patch("getpass.getpass", return_value=SECRET_PASSWORD),
        mock.patch("builtins.input", return_value="yes"),
    ):
        run_cli(["account", "delete"])
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert SECRET_PASSWORD not in combined


def test_delete_never_prints_secret_on_abort(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(delete_account=mock.Mock())
    with (
        mock.patch("dagnam.account", fake),
        mock.patch("getpass.getpass", return_value=SECRET_PASSWORD),
        mock.patch("builtins.input", return_value="no"),
        pytest.raises(SystemExit),
    ):
        run_cli(["account", "delete"])
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert SECRET_PASSWORD not in combined


def test_delete_never_prints_secret_on_api_error(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import AuthError

    fake = SimpleNamespace(delete_account=mock.Mock(side_effect=AuthError("Incorrect password")))
    with (
        mock.patch("dagnam.account", fake),
        mock.patch("getpass.getpass", return_value=SECRET_PASSWORD),
        mock.patch("builtins.input", return_value="yes"),
    ):
        run_cli(["account", "delete"])
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert SECRET_PASSWORD not in combined
