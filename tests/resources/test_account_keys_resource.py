"""Unit tests for dagnam.account API-key create/list/revoke helpers."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import UUID

from dagnam import account
from dagnam._core.client import DagnamClient

CREATED = {
    "id": "key-1",
    "name": "ci-key",
    "key_prefix": "dgk_abcd",
    "permissions": ["read"],
    "key": "dgk_abcdEFGH12345678SECRET",
}


def test_create_api_key_forwards_defaults() -> None:
    c = MagicMock(spec=DagnamClient, create_api_key=MagicMock(return_value=CREATED))
    assert account.create_api_key("ci-key", client=c) == CREATED
    c.create_api_key.assert_called_once_with("ci-key", None, None)


def test_create_api_key_forwards_scopes_and_expiry() -> None:
    c = MagicMock(spec=DagnamClient, create_api_key=MagicMock(return_value=CREATED))
    result = account.create_api_key("ci-key", ["read", "write"], 30, client=c)
    assert result == CREATED
    c.create_api_key.assert_called_once_with("ci-key", ["read", "write"], 30)


def test_list_api_keys_delegates() -> None:
    payload = [{k: v for k, v in CREATED.items() if k != "key"}]
    c = MagicMock(spec=DagnamClient, list_api_keys=MagicMock(return_value=payload))
    assert account.list_api_keys(client=c) == payload
    c.list_api_keys.assert_called_once_with()


def test_revoke_api_key_delegates() -> None:
    c = MagicMock(spec=DagnamClient, revoke_api_key=MagicMock(return_value=None))
    assert account.revoke_api_key("key-1", client=c) is None
    c.revoke_api_key.assert_called_once_with("key-1")


def test_revoke_api_key_stringifies_uuid() -> None:
    key_id = UUID("12345678-1234-5678-1234-567812345678")
    c = MagicMock(spec=DagnamClient, revoke_api_key=MagicMock(return_value=None))
    account.revoke_api_key(key_id, client=c)
    c.revoke_api_key.assert_called_once_with(str(key_id))
