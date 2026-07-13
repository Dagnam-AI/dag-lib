"""Wire-level coverage for the async account / usage client mixin.

Async mirror of ``tests/core/client/test_sync_account.py``: exercises the shared
``_account_get`` transport helper (empty-body and non-JSON fallbacks, error
mapping) and the three public read methods, plus path-segment quoting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import APIError, AuthError

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, RespxMockRouter

API = "https://api.test"

ENTITLEMENTS = "/api/v1/users/me/entitlements"
QUOTA = "/api/v1/datasets/storage/quota"
USAGE = "/api/v1/users/me/api-keys/key1/usage"

pytestmark = pytest.mark.anyio


async def test_async_account_full_surface(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get(ENTITLEMENTS).mock(
        return_value=httpx.Response(200, json={"plan": "pro", "limits": {}})
    )
    mock.get(QUOTA).mock(
        return_value=httpx.Response(200, json={"used_bytes": 10, "limit_bytes": 100})
    )
    mock.get(USAGE).mock(return_value=httpx.Response(200, json={"calls": 5}))

    assert (await client.get_entitlements())["plan"] == "pro"
    assert (await client.get_storage_quota())["limit_bytes"] == 100
    assert await client.get_api_key_usage("key1") == {"calls": 5}


async def test_async_api_key_usage_quotes_path_segment(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.get("/api/v1/users/me/api-keys/a%2Fb/usage").mock(
        return_value=httpx.Response(200, json={"calls": 0})
    )
    await client.get_api_key_usage("a/b")
    assert len(route.calls) == 1
    assert "a%2Fb" in str(route.calls[0].request.url)


async def test_async_account_empty_body_raises_type_error(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    # _account_get returns None on an empty body; ensure_json_object then rejects it.
    mock.get(ENTITLEMENTS).mock(return_value=httpx.Response(204))
    with pytest.raises(TypeError, match="Expected JSON object"):
        await client.get_entitlements()


async def test_async_account_non_json_body_raises_type_error(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    # Non-JSON body falls back to resp.text (a str); ensure_json_object rejects it.
    mock.get(ENTITLEMENTS).mock(
        return_value=httpx.Response(200, text="not-json", headers={"Content-Type": "text/plain"})
    )
    with pytest.raises(TypeError, match="Expected JSON object"):
        await client.get_entitlements()


async def test_async_account_auth_error(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get(QUOTA).mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await client.get_storage_quota()


async def test_async_account_500_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get(ENTITLEMENTS).mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(APIError):
        await client.get_entitlements()


# ---------------------------------------------------------------- transient retry (Plan 03)


async def test_async_get_entitlements_retries_transient(
    client: AsyncDagnamClient, mock: RespxMockRouter, monkeypatch: PytestMonkeyPatch
) -> None:
    async def _no_sleep(_d: float) -> None: ...

    monkeypatch.setattr(client, "_async_sleep", _no_sleep)
    monkeypatch.setattr(client, "_rng", lambda: 1.0)
    mock.get(ENTITLEMENTS).mock(
        side_effect=[
            httpx.Response(503, json={}),
            httpx.Response(200, json={"plan": "pro"}),
        ]
    )
    assert (await client.get_entitlements())["plan"] == "pro"


async def test_async_get_entitlements_401_not_retried(
    client: AsyncDagnamClient, mock: RespxMockRouter, monkeypatch: PytestMonkeyPatch
) -> None:
    async def _no_sleep(_d: float) -> None: ...

    monkeypatch.setattr(client, "_async_sleep", _no_sleep)
    route = mock.get(ENTITLEMENTS).mock(return_value=httpx.Response(401, json={}))
    with pytest.raises(AuthError):
        await client.get_entitlements()
    assert route.call_count == 1
