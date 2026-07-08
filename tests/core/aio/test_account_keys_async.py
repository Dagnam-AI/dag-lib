"""Wire-level coverage for async API-key create/list/revoke methods.

Async mirror of ``tests/core/client/test_account_keys.py``: exercises
``AsyncAccountMixin.create_api_key/list_api_keys/revoke_api_key`` plus the
shared error-mapping helpers.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import APIError, AuthError, QuotaExceededError

if TYPE_CHECKING:
    from tests.typing_helpers import RespxMockRouter

API_KEYS = "/api/v1/users/me/api-keys"

pytestmark = pytest.mark.anyio

CREATED = {
    "id": "key-1",
    "name": "ci-key",
    "key_prefix": "dgk_abcd",
    "permissions": ["read"],
    "usage_count": 0,
    "last_used_at": None,
    "expires_at": None,
    "created_at": "2026-01-01T00:00:00",
    "key": "dgk_abcdEFGH12345678SECRET",
}


# ----------------------------------------------------------------- create_api_key


async def test_create_api_key_sends_name_only_when_no_scopes(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post(API_KEYS).mock(return_value=httpx.Response(201, json=CREATED))
    result = await client.create_api_key("ci-key")
    assert result == CREATED
    assert json.loads(route.calls[0].request.read()) == {"name": "ci-key"}


async def test_create_api_key_maps_scopes_to_permissions(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post(API_KEYS).mock(return_value=httpx.Response(201, json=CREATED))
    await client.create_api_key("ci-key", ["read", "write"])
    assert json.loads(route.calls[0].request.read()) == {
        "name": "ci-key",
        "permissions": ["read", "write"],
    }


async def test_create_api_key_sends_expires_in_days_when_set(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post(API_KEYS).mock(return_value=httpx.Response(201, json=CREATED))
    await client.create_api_key("ci-key", ["read"], expires_in_days=30)
    assert json.loads(route.calls[0].request.read()) == {
        "name": "ci-key",
        "permissions": ["read"],
        "expires_in_days": 30,
    }


async def test_create_api_key_omits_expires_in_days_when_none(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post(API_KEYS).mock(return_value=httpx.Response(201, json=CREATED))
    await client.create_api_key("ci-key")
    body = json.loads(route.calls[0].request.read())
    assert "expires_in_days" not in body


async def test_create_api_key_400_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(API_KEYS).mock(return_value=httpx.Response(400, json={"detail": "invalid scope"}))
    with pytest.raises(APIError):
        await client.create_api_key("ci-key", ["not-a-real-scope"])


async def test_create_api_key_401_raises_autherror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(API_KEYS).mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await client.create_api_key("ci-key")


async def test_create_api_key_402_raises_quotaexceedederror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(API_KEYS).mock(
        return_value=httpx.Response(402, json={"message": "Plan limit reached"})
    )
    with pytest.raises(QuotaExceededError):
        await client.create_api_key("ci-key")


# ------------------------------------------------------------------- list_api_keys


async def test_list_api_keys_returns_array(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    payload = [{k: v for k, v in CREATED.items() if k != "key"}]
    mock.get(API_KEYS).mock(return_value=httpx.Response(200, json=payload))
    result = await client.list_api_keys()
    assert result == payload


async def test_list_api_keys_401_raises_autherror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get(API_KEYS).mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await client.list_api_keys()


# ------------------------------------------------------------------ revoke_api_key


async def test_revoke_api_key_sends_delete(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.delete(f"{API_KEYS}/key-1").mock(return_value=httpx.Response(204))
    result = await client.revoke_api_key("key-1")
    assert result is None


async def test_revoke_api_key_quotes_id(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.delete(f"{API_KEYS}/a%2Fb").mock(return_value=httpx.Response(204))
    await client.revoke_api_key("a/b")


async def test_revoke_api_key_401_raises_autherror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.delete(f"{API_KEYS}/key-1").mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await client.revoke_api_key("key-1")


async def test_revoke_api_key_404_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.delete(f"{API_KEYS}/ghost").mock(
        return_value=httpx.Response(404, json={"detail": "API key not found"})
    )
    with pytest.raises(APIError):
        await client.revoke_api_key("ghost")
