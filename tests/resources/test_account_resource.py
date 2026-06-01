"""Unit tests for dagnam.account (entitlements / usage)."""

from __future__ import annotations

from unittest.mock import MagicMock

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
