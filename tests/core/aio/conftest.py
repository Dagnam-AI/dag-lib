"""Shared fixtures for the async client tests (respx-backed)."""

from __future__ import annotations

import pytest
import respx

from dagnam._core.aio import AsyncDagnamClient

API = "https://api.test"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    async with AsyncDagnamClient(API, "k") as c:
        yield c


@pytest.fixture
def mock():
    with respx.mock(base_url=API, assert_all_called=False) as r:
        yield r
