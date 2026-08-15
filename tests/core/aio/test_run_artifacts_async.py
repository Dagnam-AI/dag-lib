"""Wire-level coverage for the async run-scoped artifact-push client methods.

Async mirror of ``tests/core/client/test_run_artifacts.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import APIError, TrainingJobNotFoundError

if TYPE_CHECKING:
    from pathlib import Path

    from tests.typing_helpers import RespxMockRouter

pytestmark = pytest.mark.anyio


# ----------------------------------------------------------------- initiate


async def test_async_initiate_run_artifacts_posts_the_declared_file_set(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post("/api/v1/training/jobs/j1/artifacts").mock(
        return_value=httpx.Response(200, json={"version_id": "v1", "status": "draft"})
    )
    result = await client.initiate_run_artifacts(
        "j1", {"files": [{"filename": "model.safetensors", "size_bytes": 7}]}
    )
    assert result["version_id"] == "v1"
    assert b'"model.safetensors"' in route.calls[0].request.content


async def test_async_initiate_run_artifacts_maps_404_to_job_not_found(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/training/jobs/j1/artifacts").mock(return_value=httpx.Response(404))
    with pytest.raises(TrainingJobNotFoundError):
        await client.initiate_run_artifacts("j1", {"files": []})


async def test_async_initiate_run_artifacts_surfaces_a_conflict(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/training/jobs/j1/artifacts").mock(
        return_value=httpx.Response(409, json={"detail": "a concurrent push is in progress"})
    )
    with pytest.raises(APIError, match="concurrent push"):
        await client.initiate_run_artifacts("j1", {"files": []})


# ------------------------------------------------------------------- upload


async def test_async_upload_run_artifact_posts_multipart(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"\x00\x01\x02")
    route = mock.post("/api/v1/training/jobs/j1/artifacts/a1/upload").mock(
        return_value=httpx.Response(204)
    )
    assert await client.upload_run_artifact("j1", "a1", weights) is True
    assert b'name="file"' in route.calls[0].request.content


async def test_async_upload_run_artifact_returns_false_when_already_committed(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"\x00")
    mock.post("/api/v1/training/jobs/j1/artifacts/a1/upload").mock(
        return_value=httpx.Response(409, json={"detail": "artifact a1 is already verified"})
    )
    assert await client.upload_run_artifact("j1", "a1", weights) is False


async def test_async_upload_run_artifact_maps_404_to_job_not_found(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"\x00")
    mock.post("/api/v1/training/jobs/j1/artifacts/nope/upload").mock(
        return_value=httpx.Response(404)
    )
    with pytest.raises(TrainingJobNotFoundError):
        await client.upload_run_artifact("j1", "nope", weights)


# --------------------------------------------------------- complete/finalize


async def test_async_complete_run_artifact_posts_the_digest(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post("/api/v1/training/jobs/j1/artifacts/a1/complete").mock(
        return_value=httpx.Response(200, json={"id": "a1", "verification_status": "verified"})
    )
    result = await client.complete_run_artifact("j1", "a1", {"sha256": "abc", "size_bytes": 3})
    assert result["verification_status"] == "verified"
    assert b'"abc"' in route.calls[0].request.content


async def test_async_finalize_run_artifacts_commits_the_version(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/training/jobs/j1/artifacts:finalize").mock(
        return_value=httpx.Response(200, json={"version_id": "v1", "status": "ready"})
    )
    assert (await client.finalize_run_artifacts("j1"))["status"] == "ready"


async def test_async_finalize_run_artifacts_surfaces_an_incomplete_version(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/training/jobs/j1/artifacts:finalize").mock(
        return_value=httpx.Response(
            422, json={"detail": "cannot finalize a version with an artifact that is not verified"}
        )
    )
    with pytest.raises(APIError, match="not verified"):
        await client.finalize_run_artifacts("j1")
