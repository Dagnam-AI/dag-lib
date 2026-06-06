"""Async datasets client mixin."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import (
    DatasetNotFoundError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import RespxMockRouter

API = "https://api.test"

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------- datasets


async def test_async_list_datasets_with_and_without_search(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.get("/api/v1/datasets/browse").mock(return_value=httpx.Response(200, json=[]))
    await client.list_datasets(search="cifar")
    assert "search=cifar" in str(route.calls[0].request.url)
    await client.list_datasets()


async def test_async_get_dataset_meta(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/datasets/ds1/meta").mock(return_value=httpx.Response(200, json={"id": "ds1"}))
    assert await client.get_dataset_meta("ds1") == {"id": "ds1"}


async def test_async_dataset_404(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/datasets/missing/meta").mock(return_value=httpx.Response(404))
    with pytest.raises(DatasetNotFoundError):
        await client.get_dataset_meta("missing")


async def test_async_list_system_datasets(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/datasets/system").mock(
        return_value=httpx.Response(200, json=[{"id": "iris"}])
    )
    assert await client.list_system_datasets() == [{"id": "iris"}]


async def test_async_get_system_dataset_meta(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/datasets/system/iris").mock(
        return_value=httpx.Response(200, json={"id": "iris"})
    )
    assert await client.get_system_dataset_meta("iris") == {"id": "iris"}


async def test_async_download_dataset(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get("/api/v1/datasets/ds1/download").mock(
        return_value=httpx.Response(
            200,
            content=b"data",
            headers={"content-disposition": 'attachment; filename="ds.bin"'},
        )
    )
    out = await client.download_dataset("ds1", tmp_path)
    assert out.read_bytes() == b"data"


async def test_async_download_system_dataset(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get("/api/v1/datasets/system/iris/download").mock(
        return_value=httpx.Response(
            200,
            content=b"iris",
            headers={"content-disposition": 'attachment; filename="iris.csv"'},
        )
    )
    out = await client.download_system_dataset("iris", tmp_path)
    assert out.read_bytes() == b"iris"


async def test_async_upload_dataset(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    fp = tmp_path / "x.csv"
    fp.write_text("a,b\n1,2")
    mock.post("/api/v1/datasets/upload").mock(return_value=httpx.Response(200, json={"id": "ds1"}))
    result = await client.upload_dataset(
        fp,
        name="x",
        dataset_type="tabular",
        format="csv",
        description="desc",
        license="MIT",
    )
    assert result == {"id": "ds1"}


async def test_async_upload_dataset_from_url(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/datasets/upload-url").mock(
        return_value=httpx.Response(200, json={"task_id": "t1"})
    )
    result = await client.upload_dataset_from_url(
        "https://x/data.csv",
        name="n",
        dataset_type="t",
        format="csv",
        description="d",
    )
    assert result == {"task_id": "t1"}


async def test_async_upload_dataset_minimal(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    fp = tmp_path / "x.csv"
    fp.write_text("a,b\n1,2")
    mock.post("/api/v1/datasets/upload").mock(return_value=httpx.Response(200, json={"id": "ds1"}))
    result = await client.upload_dataset(
        fp,
        name="x",
        dataset_type="tabular",
        format="csv",
    )
    assert result == {"id": "ds1"}


async def test_async_upload_dataset_from_url_minimal(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/datasets/upload-url").mock(
        return_value=httpx.Response(200, json={"task_id": "t1"})
    )
    result = await client.upload_dataset_from_url(
        "https://x/data.csv",
        name="n",
        dataset_type="t",
        format="csv",
    )
    assert result == {"task_id": "t1"}


async def test_async_get_dataset_task_status(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/datasets/tasks/t1").mock(
        return_value=httpx.Response(200, json={"status": "done"})
    )
    assert await client.get_dataset_task_status("t1") == {"status": "done"}
