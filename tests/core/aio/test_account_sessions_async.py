"""Wire-level coverage for async change-password + session management methods.

Async mirror of ``tests/core/client/test_account_sessions.py``: exercises
``AsyncAccountMixin.change_password/list_sessions/revoke_session/
revoke_all_sessions`` plus the shared error-mapping helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import APIError, AuthError, QuotaExceededError

if TYPE_CHECKING:
    from tests.typing_helpers import RespxMockRouter

CHANGE_PASSWORD = "/api/v1/users/me/change-password"
SESSIONS = "/api/v1/users/me/sessions"
REVOKE_ALL_SESSIONS = "/api/v1/users/me/revoke-all-sessions"

pytestmark = pytest.mark.anyio


# --------------------------------------------------------------- change_password


async def test_change_password_sends_body(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    route = mock.post(CHANGE_PASSWORD).mock(
        return_value=httpx.Response(200, json={"message": "Password changed successfully"})
    )
    result = await client.change_password("OldPassw0rd", "NewPassw0rd")
    assert result["message"] == "Password changed successfully"
    assert route.calls[0].request.read() == (
        b'{"current_password":"OldPassw0rd","new_password":"NewPassw0rd"}'
    )


async def test_change_password_401_raises_autherror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(CHANGE_PASSWORD).mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await client.change_password("wrong", "NewPassw0rd")


async def test_change_password_402_raises_quotaexceedederror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(CHANGE_PASSWORD).mock(
        return_value=httpx.Response(402, json={"message": "Plan limit"})
    )
    with pytest.raises(QuotaExceededError):
        await client.change_password("OldPassw0rd", "NewPassw0rd")


async def test_change_password_404_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(CHANGE_PASSWORD).mock(return_value=httpx.Response(404, json={"detail": "nf"}))
    with pytest.raises(APIError):
        await client.change_password("OldPassw0rd", "NewPassw0rd")


async def test_change_password_409_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(CHANGE_PASSWORD).mock(return_value=httpx.Response(409, json={"detail": "conflict"}))
    with pytest.raises(APIError):
        await client.change_password("OldPassw0rd", "NewPassw0rd")


async def test_change_password_422_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(CHANGE_PASSWORD).mock(
        return_value=httpx.Response(422, json={"detail": "Password too short"})
    )
    with pytest.raises(APIError):
        await client.change_password("OldPassw0rd", "short")


async def test_change_password_empty_body_raises_typeerror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(CHANGE_PASSWORD).mock(return_value=httpx.Response(200, content=b""))
    with pytest.raises(TypeError, match="Expected JSON object"):
        await client.change_password("OldPassw0rd", "NewPassw0rd")


# ----------------------------------------------------------------- list_sessions


async def test_list_sessions_returns_array(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    payload = [
        {"id": "s1", "device_info": {"browser": "Chrome"}, "is_current": True},
        {"id": "s2", "device_info": {"browser": "Firefox"}, "is_current": False},
    ]
    mock.get(SESSIONS).mock(return_value=httpx.Response(200, json=payload))
    result = await client.list_sessions()
    assert result == payload


async def test_list_sessions_401_raises_autherror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get(SESSIONS).mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await client.list_sessions()


async def test_list_sessions_non_array_body_raises_typeerror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get(SESSIONS).mock(return_value=httpx.Response(200, json={"not": "a list"}))
    with pytest.raises(TypeError, match="Expected JSON array"):
        await client.list_sessions()


async def test_list_sessions_empty_body_raises_typeerror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get(SESSIONS).mock(return_value=httpx.Response(200, content=b""))
    with pytest.raises(TypeError, match="Expected JSON array"):
        await client.list_sessions()


# --------------------------------------------------------------- revoke_session


async def test_revoke_session_sends_delete(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.delete(f"{SESSIONS}/s1").mock(return_value=httpx.Response(204))
    result = await client.revoke_session("s1")
    assert result is None


async def test_revoke_session_quotes_id(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.delete(f"{SESSIONS}/a%2Fb").mock(return_value=httpx.Response(204))
    await client.revoke_session("a/b")


async def test_revoke_session_401_raises_autherror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.delete(f"{SESSIONS}/s1").mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await client.revoke_session("s1")


async def test_revoke_session_404_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.delete(f"{SESSIONS}/ghost").mock(
        return_value=httpx.Response(404, json={"detail": "not found"})
    )
    with pytest.raises(APIError):
        await client.revoke_session("ghost")


async def test_revoke_session_409_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.delete(f"{SESSIONS}/s1").mock(
        return_value=httpx.Response(409, json={"detail": "conflict"})
    )
    with pytest.raises(APIError):
        await client.revoke_session("s1")


async def test_revoke_session_422_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.delete(f"{SESSIONS}/s1").mock(return_value=httpx.Response(422, json={"detail": "bad id"}))
    with pytest.raises(APIError):
        await client.revoke_session("s1")


# ----------------------------------------------------------- revoke_all_sessions


async def test_revoke_all_sessions_sends_post(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post(REVOKE_ALL_SESSIONS).mock(
        return_value=httpx.Response(200, json={"message": "All sessions revoked"})
    )
    result = await client.revoke_all_sessions()
    assert result["message"] == "All sessions revoked"
    assert route.called


async def test_revoke_all_sessions_401_raises_autherror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(REVOKE_ALL_SESSIONS).mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await client.revoke_all_sessions()


async def test_revoke_all_sessions_402_raises_quotaexceedederror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(REVOKE_ALL_SESSIONS).mock(
        return_value=httpx.Response(402, json={"message": "Plan limit"})
    )
    with pytest.raises(QuotaExceededError):
        await client.revoke_all_sessions()


async def test_revoke_all_sessions_409_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(REVOKE_ALL_SESSIONS).mock(
        return_value=httpx.Response(409, json={"detail": "conflict"})
    )
    with pytest.raises(APIError):
        await client.revoke_all_sessions()
