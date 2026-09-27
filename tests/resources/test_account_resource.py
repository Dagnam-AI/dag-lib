"""Unit tests for dagnam.account (entitlements / usage)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dagnam import account
from dagnam._core.client import DagnamClient


def test_entitlements_delegates() -> None:
    snap = {"plan": {"code": "pro"}, "limits": []}
    c = MagicMock(spec=DagnamClient, get_entitlements=MagicMock(return_value=snap))
    assert account.entitlements(client=c) == snap
    c.get_entitlements.assert_called_once_with()


def test_storage_quota_delegates() -> None:
    quota = {"used_bytes": 1, "limit_bytes": 100}
    c = MagicMock(spec=DagnamClient, get_storage_quota=MagicMock(return_value=quota))
    assert account.storage_quota(client=c) == quota
    c.get_storage_quota.assert_called_once_with()


def test_api_key_usage_stringifies_id() -> None:
    usage = {"usage_count": 7}
    c = MagicMock(spec=DagnamClient, get_api_key_usage=MagicMock(return_value=usage))
    assert account.api_key_usage("key_1", client=c) == usage
    c.get_api_key_usage.assert_called_once_with("key_1")


@pytest.mark.parametrize("name", ["create_api_key", "list_api_keys", "revoke_api_key"])
def test_key_management_functions_are_gone(name: str) -> None:
    assert not hasattr(account, name)
    assert name not in account.__all__
    assert "api_key_usage" in account.__all__


SESSION_ONLY_FUNCTIONS = [
    "change_password",
    "delete_account",
    "disable_two_factor",
    "download_export",
    "enable_two_factor",
    "export_data",
    "get_profile",
    "get_settings",
    "list_sessions",
    "notification_preferences",
    "reset_settings",
    "revoke_all_sessions",
    "revoke_session",
    "two_factor_enabled",
    "update_notification_preferences",
    "update_profile",
    "update_settings",
    "upload_profile_photo",
    "verify_two_factor",
]


@pytest.mark.parametrize("name", SESSION_ONLY_FUNCTIONS)
def test_session_only_functions_are_gone(name: str) -> None:
    assert not hasattr(account, name)
    assert name not in account.__all__


def test_key_accepting_functions_stay() -> None:
    kept = {"api_key_usage", "entitlements", "get_public_profile", "register", "storage_quota"}
    assert set(account.__all__) == kept
