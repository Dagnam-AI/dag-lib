"""Unit tests for dagnam.account settings / notification-preference helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from dagnam import account
from dagnam._core.client import DagnamClient


def test_get_settings_delegates() -> None:
    payload = {"theme": "dark", "auto_save_interval": 500}
    c = MagicMock(spec=DagnamClient, get_settings=MagicMock(return_value=payload))
    assert account.get_settings(client=c) == payload
    c.get_settings.assert_called_once_with()


def test_update_settings_forwards_fields_as_patch() -> None:
    payload = {"theme": "dark"}
    c = MagicMock(spec=DagnamClient, update_settings=MagicMock(return_value=payload))
    assert account.update_settings(client=c, theme="dark") == payload
    c.update_settings.assert_called_once_with({"theme": "dark"})


def test_update_settings_with_no_fields_sends_empty_patch() -> None:
    c = MagicMock(spec=DagnamClient, update_settings=MagicMock(return_value={}))
    account.update_settings(client=c)
    c.update_settings.assert_called_once_with({})


def test_reset_settings_delegates() -> None:
    payload = {"theme": "system"}
    c = MagicMock(spec=DagnamClient, reset_settings=MagicMock(return_value=payload))
    assert account.reset_settings(client=c) == payload
    c.reset_settings.assert_called_once_with()


def test_notification_preferences_delegates() -> None:
    payload = {"email_enabled": True}
    c = MagicMock(spec=DagnamClient, get_notification_prefs=MagicMock(return_value=payload))
    assert account.notification_preferences(client=c) == payload
    c.get_notification_prefs.assert_called_once_with()


def test_update_notification_preferences_forwards_fields_as_patch() -> None:
    payload = {"training_alerts": False}
    c = MagicMock(spec=DagnamClient, update_notification_prefs=MagicMock(return_value=payload))
    assert account.update_notification_preferences(client=c, training_alerts=False) == payload
    c.update_notification_prefs.assert_called_once_with({"training_alerts": False})
