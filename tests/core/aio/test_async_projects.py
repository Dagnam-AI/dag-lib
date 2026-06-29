"""Async projects client mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import (
    ProjectNotFoundError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import RespxMockRouter

API = "https://api.test"

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------- projects


async def test_async_projects_full_surface(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/projects").mock(return_value=httpx.Response(200, json={"items": []}))
    mock.get("/api/v1/projects/p1").mock(return_value=httpx.Response(200, json={"id": "p1"}))
    mock.post("/api/v1/projects").mock(return_value=httpx.Response(200, json={"id": "p1"}))
    mock.put("/api/v1/projects/p1").mock(return_value=httpx.Response(200, json={"id": "p1"}))
    mock.delete("/api/v1/projects/p1").mock(return_value=httpx.Response(204))
    mock.post("/api/v1/projects/p1/duplicate").mock(return_value=httpx.Response(200, json={}))
    mock.post("/api/v1/projects/p1/save").mock(return_value=httpx.Response(200, json={}))
    mock.post("/api/v1/projects/import").mock(return_value=httpx.Response(200, json={}))
    mock.post("/api/v1/projects/p1/import").mock(return_value=httpx.Response(200, json={}))
    mock.post("/api/v1/projects/bulk-delete").mock(return_value=httpx.Response(200, json={}))
    mock.post("/api/v1/projects/p1/datasets").mock(return_value=httpx.Response(200, json={}))
    mock.get("/api/v1/projects/p1/datasets").mock(return_value=httpx.Response(200, json={}))
    mock.delete("/api/v1/projects/p1/datasets/d1").mock(return_value=httpx.Response(204))

    await client.list_projects()
    await client.get_project("p1")
    await client.create_project({})
    await client.update_project("p1", {})
    await client.delete_project("p1")
    await client.duplicate_project("p1", title="copy")
    await client.duplicate_project("p1")
    await client.save_architecture("p1", {})
    await client.import_dag({})
    await client.import_dag_existing("p1", {})
    await client.bulk_delete_projects(["p1"])
    await client.link_dataset("p1", "d1", "train")
    await client.get_project_datasets("p1")
    await client.unlink_dataset("p1", "d1")


async def test_async_get_project_404(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/projects/missing").mock(return_value=httpx.Response(404))
    with pytest.raises(ProjectNotFoundError):
        await client.get_project("missing")


async def test_async_projects_text_response(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/projects").mock(
        return_value=httpx.Response(200, text="plain", headers={"Content-Type": "text/plain"})
    )
    assert await client.list_projects() == "plain"


async def test_async_projects_empty_response(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/projects").mock(return_value=httpx.Response(204))
    assert await client.list_projects() is None


# ---------------------------------------------------------------- project versions


async def test_async_project_versions(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/projects/p1/versions").mock(
        return_value=httpx.Response(200, json={"items": []})
    )
    mock.get("/api/v1/projects/p1/versions/v1").mock(
        return_value=httpx.Response(200, json={"id": "v1"})
    )
    mock.get("/api/v1/projects/p1/versions/compare").mock(
        return_value=httpx.Response(200, json={"version_a": {}, "version_b": {}})
    )
    mock.post("/api/v1/projects/p1/restore/v1").mock(
        return_value=httpx.Response(201, json={"id": "v2"})
    )
    mock.delete("/api/v1/projects/p1/versions/v1").mock(return_value=httpx.Response(204))
    mock.get("/api/v1/projects/p1/latest").mock(return_value=httpx.Response(200, json={"id": "v2"}))

    assert "items" in await client.list_project_versions("p1")
    assert (await client.get_project_version("p1", "v1"))["id"] == "v1"
    assert "version_a" in await client.compare_project_versions("p1", "v1", "v2")
    assert (await client.restore_project_version("p1", "v1"))["id"] == "v2"
    assert await client.delete_project_version("p1", "v1") is None
    assert (await client.get_latest_project_version("p1"))["id"] == "v2"


async def test_async_compare_project_versions_query(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.get("/api/v1/projects/p1/versions/compare").mock(
        return_value=httpx.Response(200, json={"version_a": {}, "version_b": {}})
    )
    await client.compare_project_versions("p1", "va", "vb")
    params = route.calls[0].request.url.params
    assert params["version_a"] == "va"
    assert params["version_b"] == "vb"


async def test_async_get_project_version_404(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/projects/p1/versions/missing").mock(return_value=httpx.Response(404))
    with pytest.raises(ProjectNotFoundError):
        await client.get_project_version("p1", "missing")
