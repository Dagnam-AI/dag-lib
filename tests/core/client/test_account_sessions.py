"""Wire-level coverage for change-password + session management sync methods.

Covers ``AccountClientMixin.change_password/list_sessions/revoke_session/
revoke_all_sessions``, plus the ``raise_for_generic``/``_expect_array`` error
mapping these methods share with the rest of the account surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError, AuthError, QuotaExceededError, ResponseError

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

API = "https://api.test"
CHANGE_PASSWORD = f"{API}/api/v1/users/me/change-password"
SESSIONS = f"{API}/api/v1/users/me/sessions"
REVOKE_ALL_SESSIONS = f"{API}/api/v1/users/me/revoke-all-sessions"


# --------------------------------------------------------------- change_password


def test_change_password_sends_body(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(CHANGE_PASSWORD, json={"message": "Password changed successfully"})
    result = client.change_password("OldPassw0rd", "NewPassw0rd")
    assert result["message"] == "Password changed successfully"
    assert rmock.last_request.json() == {
        "current_password": "OldPassw0rd",
        "new_password": "NewPassw0rd",
    }
    assert rmock.last_request.headers["Authorization"] == "Bearer k"


def test_change_password_401_raises_autherror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(CHANGE_PASSWORD, status_code=401, json={"detail": "Current password is incorrect"})
    with pytest.raises(AuthError):
        client.change_password("wrong", "NewPassw0rd")


def test_change_password_402_raises_quotaexceedederror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(CHANGE_PASSWORD, status_code=402, json={"message": "Plan limit reached"})
    with pytest.raises(QuotaExceededError):
        client.change_password("OldPassw0rd", "NewPassw0rd")


def test_change_password_404_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(CHANGE_PASSWORD, status_code=404, json={"detail": "not found"})
    with pytest.raises(APIError):
        client.change_password("OldPassw0rd", "NewPassw0rd")


def test_change_password_409_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(CHANGE_PASSWORD, status_code=409, json={"detail": "conflict"})
    with pytest.raises(APIError):
        client.change_password("OldPassw0rd", "NewPassw0rd")


def test_change_password_422_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(CHANGE_PASSWORD, status_code=422, json={"detail": "Password must be 8+ chars"})
    with pytest.raises(APIError):
        client.change_password("OldPassw0rd", "short")


def test_change_password_empty_body_raises_response_error(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(CHANGE_PASSWORD, status_code=200, content=b"")
    with pytest.raises(ResponseError, match="Expected JSON object"):
        client.change_password("OldPassw0rd", "NewPassw0rd")


# ----------------------------------------------------------------- list_sessions


def test_list_sessions_returns_array(client: DagnamClient, rmock: RequestsMocker) -> None:
    payload = [
        {"id": "s1", "device_info": {"browser": "Chrome"}, "is_current": True},
        {"id": "s2", "device_info": {"browser": "Firefox"}, "is_current": False},
    ]
    rmock.get(SESSIONS, json=payload)
    result = client.list_sessions()
    assert result == payload
    assert rmock.last_request.headers["Authorization"] == "Bearer k"


def test_list_sessions_401_raises_autherror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(SESSIONS, status_code=401, text="nope")
    with pytest.raises(AuthError):
        client.list_sessions()


def test_list_sessions_non_array_body_raises_typeerror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.get(SESSIONS, json={"not": "a list"})
    with pytest.raises(TypeError, match="Expected JSON array"):
        client.list_sessions()


def test_list_sessions_empty_body_raises_typeerror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.get(SESSIONS, status_code=200, content=b"")
    with pytest.raises(TypeError, match="Expected JSON array"):
        client.list_sessions()


# --------------------------------------------------------------- revoke_session


def test_revoke_session_sends_delete(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(f"{SESSIONS}/s1", status_code=204)
    result = client.revoke_session("s1")
    assert result is None
    assert rmock.last_request.headers["Authorization"] == "Bearer k"


def test_revoke_session_quotes_id(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(f"{SESSIONS}/a%2Fb", status_code=204)
    client.revoke_session("a/b")


def test_revoke_session_401_raises_autherror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(f"{SESSIONS}/s1", status_code=401, text="nope")
    with pytest.raises(AuthError):
        client.revoke_session("s1")


def test_revoke_session_404_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(f"{SESSIONS}/ghost", status_code=404, json={"detail": "Session not found"})
    with pytest.raises(APIError):
        client.revoke_session("ghost")


def test_revoke_session_409_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(f"{SESSIONS}/s1", status_code=409, json={"detail": "conflict"})
    with pytest.raises(APIError):
        client.revoke_session("s1")


def test_revoke_session_422_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(f"{SESSIONS}/s1", status_code=422, json={"detail": "bad id"})
    with pytest.raises(APIError):
        client.revoke_session("s1")


# ----------------------------------------------------------- revoke_all_sessions


def test_revoke_all_sessions_sends_post(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(REVOKE_ALL_SESSIONS, json={"message": "All sessions revoked"})
    result = client.revoke_all_sessions()
    assert result["message"] == "All sessions revoked"
    assert rmock.last_request.method == "POST"
    assert rmock.last_request.headers["Authorization"] == "Bearer k"


def test_revoke_all_sessions_401_raises_autherror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(REVOKE_ALL_SESSIONS, status_code=401, text="nope")
    with pytest.raises(AuthError):
        client.revoke_all_sessions()


def test_revoke_all_sessions_402_raises_quotaexceedederror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(REVOKE_ALL_SESSIONS, status_code=402, json={"message": "Plan limit reached"})
    with pytest.raises(QuotaExceededError):
        client.revoke_all_sessions()


def test_revoke_all_sessions_409_raises_apierror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(REVOKE_ALL_SESSIONS, status_code=409, json={"detail": "conflict"})
    with pytest.raises(APIError):
        client.revoke_all_sessions()
