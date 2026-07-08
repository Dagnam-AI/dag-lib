"""CLI ``register`` subcommand: terminal-only account onboarding."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import pytest

from dagnam.cli import register as register_mod

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture

MINTED_KEY = "dgk_minted_secret_value"

KEY_OBJ = {
    "id": "k1",
    "name": "dagnam-cli",
    "key": MINTED_KEY,
    "key_prefix": "dgk_mint",
    "permissions": ["read", "write"],
    "expires_at": None,
    "created_at": "2026-01-01T00:00:00",
}


def _prompts(email: str, password: str, confirm: str | None = None):
    answers = iter([email])

    def _input(_prompt: str) -> str:
        return next(answers)

    passwords = iter([password, confirm if confirm is not None else password])

    def _getpass(_prompt: str) -> str:
        return next(passwords)

    return _getpass, _input


def _config_paths(tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> Path:
    config_dir = tmp_path / ".dagnam"
    config_file = config_dir / "config.json"
    monkeypatch.setattr("dagnam._core.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
    return config_file


# ------------------------------------------------------------- happy path


def test_register_persists_key_not_password(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    config_file = _config_paths(tmp_path, monkeypatch)
    fake = SimpleNamespace(register=mock.Mock(return_value=KEY_OBJ))
    getpass_func, input_func = _prompts("a@b.c", "Secret123!")
    with mock.patch("dagnam.account", fake):
        register_mod.cmd_register(
            argparse.Namespace(api_url=None), getpass_func=getpass_func, input_func=input_func
        )

    data = json.loads(config_file.read_text(encoding="utf-8"))
    assert data["api_key"] == MINTED_KEY
    dumped = json.dumps(data)
    assert "password" not in dumped
    assert "Secret123!" not in dumped
    fake.register.assert_called_once_with("a@b.c", "Secret123!", api_url=None)

    captured = capsys.readouterr()
    assert MINTED_KEY not in captured.out
    assert "Next: dagnam projects list" in captured.err


def test_config_file_is_0600_on_posix(tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    if os.name != "posix":
        pytest.skip("POSIX-only permission check")
    config_file = _config_paths(tmp_path, monkeypatch)
    fake = SimpleNamespace(register=mock.Mock(return_value=KEY_OBJ))
    getpass_func, input_func = _prompts("a@b.c", "Secret123!")
    with mock.patch("dagnam.account", fake):
        register_mod.cmd_register(
            argparse.Namespace(api_url=None), getpass_func=getpass_func, input_func=input_func
        )
    assert (config_file.stat().st_mode & 0o777) == 0o600


def test_register_saves_custom_api_url(tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    config_file = _config_paths(tmp_path, monkeypatch)
    fake = SimpleNamespace(register=mock.Mock(return_value=KEY_OBJ))
    getpass_func, input_func = _prompts("a@b.c", "Secret123!")
    with mock.patch("dagnam.account", fake):
        register_mod.cmd_register(
            argparse.Namespace(api_url="https://custom"),
            getpass_func=getpass_func,
            input_func=input_func,
        )
    data = json.loads(config_file.read_text(encoding="utf-8"))
    assert data["api_url"] == "https://custom"
    fake.register.assert_called_once_with("a@b.c", "Secret123!", api_url="https://custom")


def test_register_does_not_save_default_api_url(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    config_file = _config_paths(tmp_path, monkeypatch)
    fake = SimpleNamespace(register=mock.Mock(return_value=KEY_OBJ))
    getpass_func, input_func = _prompts("a@b.c", "Secret123!")
    with mock.patch("dagnam.account", fake):
        register_mod.cmd_register(
            argparse.Namespace(api_url="https://api.dagnam.ai"),
            getpass_func=getpass_func,
            input_func=input_func,
        )
    data = json.loads(config_file.read_text(encoding="utf-8"))
    assert "api_url" not in data


def test_register_preserves_existing_config(tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    config_file = _config_paths(tmp_path, monkeypatch)
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({"training_metrics_path": "kept"}), encoding="utf-8")
    fake = SimpleNamespace(register=mock.Mock(return_value=KEY_OBJ))
    getpass_func, input_func = _prompts("a@b.c", "Secret123!")
    with mock.patch("dagnam.account", fake):
        register_mod.cmd_register(
            argparse.Namespace(api_url=None), getpass_func=getpass_func, input_func=input_func
        )
    data = json.loads(config_file.read_text(encoding="utf-8"))
    assert data["training_metrics_path"] == "kept"
    assert data["api_key"] == MINTED_KEY


# ------------------------------------------------------- input validation


def test_register_empty_email_aborts(tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    _config_paths(tmp_path, monkeypatch)
    getpass_func, input_func = _prompts("   ", "Secret123!")
    with pytest.raises(SystemExit):
        register_mod.cmd_register(
            argparse.Namespace(api_url=None), getpass_func=getpass_func, input_func=input_func
        )


def test_register_empty_password_aborts(tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    _config_paths(tmp_path, monkeypatch)
    getpass_func, input_func = _prompts("a@b.c", "")
    with pytest.raises(SystemExit):
        register_mod.cmd_register(
            argparse.Namespace(api_url=None), getpass_func=getpass_func, input_func=input_func
        )


def test_register_password_mismatch_aborts_without_echoing(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    _config_paths(tmp_path, monkeypatch)
    getpass_func, input_func = _prompts("a@b.c", "Secret123!", "Different456!")
    with pytest.raises(SystemExit):
        register_mod.cmd_register(
            argparse.Namespace(api_url=None), getpass_func=getpass_func, input_func=input_func
        )
    captured = capsys.readouterr()
    assert "Secret123!" not in captured.out + captured.err
    assert "Different456!" not in captured.out + captured.err


# ------------------------------------------------------------- error paths


def test_register_400_email_exists_surfaces_clear_error(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    _config_paths(tmp_path, monkeypatch)
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(
        register=mock.Mock(side_effect=APIError(400, "Email already registered"))
    )
    getpass_func, input_func = _prompts("a@b.c", "Secret123!")
    with mock.patch("dagnam.account", fake), pytest.raises(SystemExit):
        register_mod.cmd_register(
            argparse.Namespace(api_url=None), getpass_func=getpass_func, input_func=input_func
        )
    err = capsys.readouterr().err
    assert "Registration failed" in err


def test_register_403_prints_email_verification_guidance(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    _config_paths(tmp_path, monkeypatch)
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(register=mock.Mock(side_effect=APIError(403, "verify your email")))
    getpass_func, input_func = _prompts("a@b.c", "Secret123!")
    with mock.patch("dagnam.account", fake), pytest.raises(SystemExit):
        register_mod.cmd_register(
            argparse.Namespace(api_url=None), getpass_func=getpass_func, input_func=input_func
        )
    err = capsys.readouterr().err
    assert "email verification" in err
    assert "dagnam login" in err


def test_register_403_without_verification_surfaces_server_reason(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    _config_paths(tmp_path, monkeypatch)
    from dagnam._core.exceptions import APIError

    # A 403 whose body does NOT name verification (today's real case: a
    # suspended/deleted account failing the login step) must surface the
    # server's actual reason, never the misleading email-verification hint.
    fake = SimpleNamespace(register=mock.Mock(side_effect=APIError(403, "Account suspended")))
    getpass_func, input_func = _prompts("a@b.c", "Secret123!")
    with mock.patch("dagnam.account", fake), pytest.raises(SystemExit):
        register_mod.cmd_register(
            argparse.Namespace(api_url=None), getpass_func=getpass_func, input_func=input_func
        )
    err = capsys.readouterr().err
    assert "Registration failed" in err
    assert "Account suspended" in err
    assert "email verification" not in err


def test_register_401_login_autherror_surfaces_clear_error(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    _config_paths(tmp_path, monkeypatch)
    from dagnam._core.exceptions import AuthError

    fake = SimpleNamespace(register=mock.Mock(side_effect=AuthError("Incorrect email or password")))
    getpass_func, input_func = _prompts("a@b.c", "Secret123!")
    with mock.patch("dagnam.account", fake), pytest.raises(SystemExit):
        register_mod.cmd_register(
            argparse.Namespace(api_url=None), getpass_func=getpass_func, input_func=input_func
        )
    err = capsys.readouterr().err
    assert "Registration failed" in err


def test_register_missing_key_field_aborts(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    _config_paths(tmp_path, monkeypatch)
    fake = SimpleNamespace(register=mock.Mock(return_value={"id": "k1"}))
    getpass_func, input_func = _prompts("a@b.c", "Secret123!")
    with mock.patch("dagnam.account", fake), pytest.raises(SystemExit):
        register_mod.cmd_register(
            argparse.Namespace(api_url=None), getpass_func=getpass_func, input_func=input_func
        )
    err = capsys.readouterr().err
    assert "API key" in err


# ------------------------------------------------------------------ wiring


def test_register_wires_top_level_command(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    _config_paths(tmp_path, monkeypatch)
    getpass_func, input_func = _prompts("a@b.c", "Secret123!")
    monkeypatch.setattr("getpass.getpass", getpass_func)
    monkeypatch.setattr("builtins.input", input_func)
    fake = SimpleNamespace(register=mock.Mock(return_value=KEY_OBJ))
    with mock.patch("dagnam.account", fake):
        exit_code = run_cli(["register"])
    assert exit_code == 0
    fake.register.assert_called_once_with("a@b.c", "Secret123!", api_url=None)
