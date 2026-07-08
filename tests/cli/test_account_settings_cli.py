"""CLI coverage for `dagnam account settings` and `dagnam account notifications`."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, StrCapture

# ---------------------------------------------------------------- settings get


def test_settings_get_table(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = {"id": "s1", "user_id": "u1", "theme": "dark", "auto_save_interval": 500}
    fake = SimpleNamespace(get_settings=mock.Mock(return_value=payload))
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "settings", "get"])
    fake.get_settings.assert_called_once_with()
    out = capsys.readouterr().out
    assert "theme" in out
    assert "dark" in out
    assert "auto_save_interval" in out


def test_settings_get_json(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = {"theme": "dark"}
    fake = SimpleNamespace(get_settings=mock.Mock(return_value=payload))
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "settings", "get", "--json"])
    assert json.loads(capsys.readouterr().out) == payload


# ---------------------------------------------------------------- settings set


def test_settings_set_parses_kv_pairs(run_cli: CliRunner) -> None:
    fake = SimpleNamespace(update_settings=mock.Mock(return_value={"theme": "dark"}))
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "settings", "set", "theme=dark", "auto_save_interval=250"])
    fake.update_settings.assert_called_once_with(theme="dark", auto_save_interval=250)


def test_settings_set_coerces_bool_field(run_cli: CliRunner) -> None:
    fake = SimpleNamespace(update_settings=mock.Mock(return_value={}))
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "settings", "set", "canvas_grid_enabled=false"])
    fake.update_settings.assert_called_once_with(canvas_grid_enabled=False)


def test_settings_set_rejects_pair_without_equals(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(update_settings=mock.Mock(return_value={}))
    with mock.patch("dagnam.account", fake), pytest.raises(SystemExit) as exc_info:
        run_cli(["account", "settings", "set", "theme"])
    assert exc_info.value.code == 1
    assert "Invalid KEY=VALUE" in capsys.readouterr().err
    fake.update_settings.assert_not_called()


def test_settings_set_rejects_bad_bool_value(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(update_settings=mock.Mock(return_value={}))
    with mock.patch("dagnam.account", fake), pytest.raises(SystemExit) as exc_info:
        run_cli(["account", "settings", "set", "canvas_grid_enabled=maybe"])
    assert exc_info.value.code == 1
    assert "boolean" in capsys.readouterr().err.lower()


def test_settings_set_rejects_bad_int_value(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(update_settings=mock.Mock(return_value={}))
    with mock.patch("dagnam.account", fake), pytest.raises(SystemExit) as exc_info:
        run_cli(["account", "settings", "set", "auto_save_interval=notanumber"])
    assert exc_info.value.code == 1
    assert "integer" in capsys.readouterr().err.lower()


def test_settings_set_rejects_reserved_field_name(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(update_settings=mock.Mock(return_value={}))
    with mock.patch("dagnam.account", fake), pytest.raises(SystemExit) as exc_info:
        run_cli(["account", "settings", "set", "client=nope"])
    assert exc_info.value.code == 1
    assert "Unsupported field name" in capsys.readouterr().err
    fake.update_settings.assert_not_called()


def test_settings_set_apierror_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(update_settings=mock.Mock(side_effect=APIError(422, "bad value")))
    with mock.patch("dagnam.account", fake):
        assert run_cli(["account", "settings", "set", "theme=dark"]) == 1
    assert "bad value" in capsys.readouterr().err


# ---------------------------------------------------------------- settings reset


def test_settings_reset(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = {"theme": "system"}
    fake = SimpleNamespace(reset_settings=mock.Mock(return_value=payload))
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "settings", "reset", "--json"])
    fake.reset_settings.assert_called_once_with()
    assert json.loads(capsys.readouterr().out) == payload


# ---------------------------------------------------------------- notifications get


def test_notifications_get_table(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = {"id": "n1", "email_enabled": True, "weekly_digest": False}
    fake = SimpleNamespace(notification_preferences=mock.Mock(return_value=payload))
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "notifications", "get"])
    fake.notification_preferences.assert_called_once_with()
    out = capsys.readouterr().out
    assert "email_enabled" in out
    assert "True" in out


def test_notifications_get_json(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = {"email_enabled": True}
    fake = SimpleNamespace(notification_preferences=mock.Mock(return_value=payload))
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "notifications", "get", "--json"])
    assert json.loads(capsys.readouterr().out) == payload


# ---------------------------------------------------------------- notifications set


def test_notifications_set_coerces_bool_fields(run_cli: CliRunner) -> None:
    fake = SimpleNamespace(update_notification_preferences=mock.Mock(return_value={}))
    with mock.patch("dagnam.account", fake):
        run_cli(
            [
                "account",
                "notifications",
                "set",
                "training_alerts=false",
                "weekly_digest=true",
            ]
        )
    fake.update_notification_preferences.assert_called_once_with(
        training_alerts=False, weekly_digest=True
    )


def test_notifications_set_rejects_bad_bool_value(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(update_notification_preferences=mock.Mock(return_value={}))
    with mock.patch("dagnam.account", fake), pytest.raises(SystemExit) as exc_info:
        run_cli(["account", "notifications", "set", "email_enabled=nope"])
    assert exc_info.value.code == 1
    assert "boolean" in capsys.readouterr().err.lower()


def test_notifications_set_apierror_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import AuthError

    fake = SimpleNamespace(
        update_notification_preferences=mock.Mock(
            side_effect=AuthError("Authentication failed: invalid or expired API key")
        )
    )
    with mock.patch("dagnam.account", fake):
        assert run_cli(["account", "notifications", "set", "email_enabled=true"]) == 1
    assert "Authentication failed" in capsys.readouterr().err
