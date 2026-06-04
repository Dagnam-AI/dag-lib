"""Async checkpoints client mixin."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import (
    APIError,
    AuthError,
    CheckpointNotFoundError,
    TrainingJobNotFoundError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import RespxMockRouter

API = "https://api.test"

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------- checkpoints


async def test_async_list_checkpoints(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/training/jobs/job1/checkpoints").mock(
        return_value=httpx.Response(200, json=[{"id": "c1"}])
    )
    assert await client.list_checkpoints("job1") == [{"id": "c1"}]


async def test_async_list_checkpoints_404(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/training/jobs/job1/checkpoints").mock(return_value=httpx.Response(404))
    with pytest.raises(TrainingJobNotFoundError):
        await client.list_checkpoints("job1")


async def test_async_download_checkpoint(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    url = "/api/v1/training/jobs/job1/checkpoints/ck1/download"
    mock.get(url).mock(
        return_value=httpx.Response(200, content=b"weights", headers={"x-checksum-sha256": "abc"})
    )
    dest = tmp_path / "ck.bin"
    written, checksum = await client.download_checkpoint("job1", "ck1", dest)
    assert written.read_bytes() == b"weights"
    assert checksum == "abc"


async def test_async_download_checkpoint_401(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    url = "/api/v1/training/jobs/job1/checkpoints/ck1/download"
    mock.get(url).mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await client.download_checkpoint("job1", "ck1", tmp_path / "x")


async def test_async_download_checkpoint_404(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    url = "/api/v1/training/jobs/job1/checkpoints/ck1/download"
    mock.get(url).mock(return_value=httpx.Response(404))
    with pytest.raises(CheckpointNotFoundError):
        await client.download_checkpoint("job1", "ck1", tmp_path / "x")


async def test_async_download_checkpoint_500(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    url = "/api/v1/training/jobs/job1/checkpoints/ck1/download"
    mock.get(url).mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(APIError):
        await client.download_checkpoint("job1", "ck1", tmp_path / "x")
