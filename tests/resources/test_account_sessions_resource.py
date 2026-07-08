"""Unit tests for dagnam.account change-password / session-management helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from dagnam import account
from dagnam._core.client import DagnamClient


def test_change_password_delegates() -> None:
    payload = {"message": "Password changed successfully"}
    c = MagicMock(spec=DagnamClient, change_password=MagicMock(return_value=payload))
    assert account.change_password("old-secret", "new-secret", client=c) == payload
    c.change_password.assert_called_once_with("old-secret", "new-secret")


def test_list_sessions_delegates() -> None:
    payload = [{"id": "s1"}, {"id": "s2"}]
    c = MagicMock(spec=DagnamClient, list_sessions=MagicMock(return_value=payload))
    assert account.list_sessions(client=c) == payload
    c.list_sessions.assert_called_once_with()


def test_revoke_session_delegates() -> None:
    c = MagicMock(spec=DagnamClient, revoke_session=MagicMock(return_value=None))
    assert account.revoke_session("s1", client=c) is None
    c.revoke_session.assert_called_once_with("s1")


def test_revoke_all_sessions_delegates() -> None:
    payload = {"message": "All sessions revoked"}
    c = MagicMock(spec=DagnamClient, revoke_all_sessions=MagicMock(return_value=payload))
    assert account.revoke_all_sessions(client=c) == payload
    c.revoke_all_sessions.assert_called_once_with()
