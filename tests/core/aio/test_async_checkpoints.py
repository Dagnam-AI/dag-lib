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


async def test_async_download_checkpoint_307_redirect_to_presigned(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    """A 307 to a presigned URL is followed; the API key is NOT forwarded."""
    url = "/api/v1/training/jobs/job1/checkpoints/ck1/download"
    presigned = "https://bucket.s3.example.com/ck1?sig=xyz"
    mock.get(url).mock(return_value=httpx.Response(307, headers={"location": presigned}))
    presigned_route = mock.get(presigned).mock(
        return_value=httpx.Response(
            200, content=b"weights", headers={"x-checksum-sha256": "sha-from-s3"}
        )
    )
    dest = tmp_path / "ck.bin"
    written, checksum = await client.download_checkpoint("job1", "ck1", dest)
    assert written.read_bytes() == b"weights"
    assert checksum == "sha-from-s3"
    presigned_req = presigned_route.calls[-1].request
    assert "authorization" not in presigned_req.headers


async def test_async_download_checkpoint_308_redirect(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    """A 308 redirect is followed identically to a 307."""
    url = "/api/v1/training/jobs/job1/checkpoints/ck1/download"
    presigned = "https://bucket.s3.example.com/ck1?sig=abc"
    mock.get(url).mock(return_value=httpx.Response(308, headers={"location": presigned}))
    mock.get(presigned).mock(return_value=httpx.Response(200, content=b"bytes"))
    dest = tmp_path / "ck.bin"
    written, checksum = await client.download_checkpoint("job1", "ck1", dest)
    assert written.read_bytes() == b"bytes"
    assert checksum is None


async def test_async_download_checkpoint_redirect_checksum_from_original(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    """The checksum header on the redirect response itself is honored."""
    url = "/api/v1/training/jobs/job1/checkpoints/ck1/download"
    presigned = "https://bucket.s3.example.com/ck1?sig=def"
    mock.get(url).mock(
        return_value=httpx.Response(
            307, headers={"location": presigned, "x-checksum-sha256": "sha-from-api"}
        )
    )
    mock.get(presigned).mock(return_value=httpx.Response(200, content=b"weights"))
    dest = tmp_path / "ck.bin"
    _written, checksum = await client.download_checkpoint("job1", "ck1", dest)
    assert checksum == "sha-from-api"


async def test_async_download_checkpoint_redirect_missing_location(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    """A redirect with no Location header surfaces as an APIError."""
    url = "/api/v1/training/jobs/job1/checkpoints/ck1/download"
    mock.get(url).mock(return_value=httpx.Response(307))
    with pytest.raises(APIError):
        await client.download_checkpoint("job1", "ck1", tmp_path / "x")


async def test_async_download_checkpoint_presigned_connecterror(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    """A connection failure fetching the presigned URL maps to APIError."""
    url = "/api/v1/training/jobs/job1/checkpoints/ck1/download"
    presigned = "https://bucket.s3.example.com/ck1?sig=xyz"
    mock.get(url).mock(return_value=httpx.Response(307, headers={"location": presigned}))
    mock.get(presigned).mock(side_effect=httpx.ConnectError("nope"))
    with pytest.raises(APIError, match="Connection failed"):
        await client.download_checkpoint("job1", "ck1", tmp_path / "x")


async def test_async_download_checkpoint_presigned_timeout(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    """A timeout fetching the presigned URL maps to APIError."""
    url = "/api/v1/training/jobs/job1/checkpoints/ck1/download"
    presigned = "https://bucket.s3.example.com/ck1?sig=xyz"
    mock.get(url).mock(return_value=httpx.Response(307, headers={"location": presigned}))
    mock.get(presigned).mock(side_effect=httpx.TimeoutException("slow"))
    with pytest.raises(APIError, match="Request timed out"):
        await client.download_checkpoint("job1", "ck1", tmp_path / "x")


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
