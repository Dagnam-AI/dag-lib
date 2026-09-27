"""Wire-level coverage for the async ``get_public_profile`` client method."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import APIError

if TYPE_CHECKING:
    from tests.typing_helpers import RespxMockRouter

pytestmark = pytest.mark.anyio


async def test_get_public_profile(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/users/ada/profile").mock(
        return_value=httpx.Response(200, json={"display_name": "Ada"})
    )
    result = await client.get_public_profile("ada")
    assert result["display_name"] == "Ada"


async def test_get_public_profile_quotes_username(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/users/a%2Fb/profile").mock(
        return_value=httpx.Response(200, json={"display_name": "x"})
    )
    result = await client.get_public_profile("a/b")
    assert result["display_name"] == "x"


async def test_get_public_profile_404_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/users/ghost/profile").mock(
        return_value=httpx.Response(404, json={"detail": "not found"})
    )
    with pytest.raises(APIError):
        await client.get_public_profile("ghost")
