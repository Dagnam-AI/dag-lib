"""Async datasets client mixin."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import (
    APIError,
    DatasetNotFoundError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, RespxMockRouter

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


async def test_async_download_dataset_streams_to_disk(
    client: AsyncDagnamClient,
    mock: RespxMockRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The download must stream via httpx (client.stream) and write chunks to
    # disk, never buffer the entire body into memory (an OOM risk for large
    # datasets).
    mock.get("/api/v1/datasets/ds1/download").mock(
        return_value=httpx.Response(
            200,
            content=b"streamed-bytes",
            headers={"content-disposition": 'attachment; filename="ds.bin"'},
        )
    )
    stream_used = {"called": False}
    real_stream = client._client.stream

    def spy_stream(*args: Any, **kwargs: Any) -> Any:
        stream_used["called"] = True
        return real_stream(*args, **kwargs)

    monkeypatch.setattr(client._client, "stream", spy_stream)
    out = await client.download_dataset("ds1", tmp_path)

    assert out.read_bytes() == b"streamed-bytes"
    assert stream_used["called"] is True


async def test_async_download_dataset_raises_on_error_status(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get("/api/v1/datasets/missing/download").mock(return_value=httpx.Response(404))
    with pytest.raises(DatasetNotFoundError):
        await client.download_dataset("missing", tmp_path)


async def test_async_download_dataset_connect_error(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get("/api/v1/datasets/ds1/download").mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(APIError, match="Connection failed"):
        await client.download_dataset("ds1", tmp_path)


async def test_async_download_dataset_timeout(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get("/api/v1/datasets/ds1/download").mock(side_effect=httpx.ConnectTimeout("slow"))
    with pytest.raises(APIError, match="Request timed out"):
        await client.download_dataset("ds1", tmp_path)


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


# ---------------------------------------------------------------- transient retry (Plan 03)


async def test_async_get_dataset_meta_retries_transient(
    client: AsyncDagnamClient, mock: RespxMockRouter, monkeypatch: PytestMonkeyPatch
) -> None:
    async def _no_sleep(_d: float) -> None: ...

    monkeypatch.setattr(client, "_async_sleep", _no_sleep)
    monkeypatch.setattr(client, "_rng", lambda: 1.0)
    mock.get("/api/v1/datasets/ds1/meta").mock(
        side_effect=[
            httpx.Response(503, json={}),
            httpx.Response(200, json={"id": "ds1"}),
        ]
    )
    assert await client.get_dataset_meta("ds1") == {"id": "ds1"}


async def test_async_get_dataset_meta_404_not_retried(
    client: AsyncDagnamClient, mock: RespxMockRouter, monkeypatch: PytestMonkeyPatch
) -> None:
    async def _no_sleep(_d: float) -> None: ...

    monkeypatch.setattr(client, "_async_sleep", _no_sleep)
    route = mock.get("/api/v1/datasets/missing/meta").mock(
        return_value=httpx.Response(404, json={})
    )
    with pytest.raises(DatasetNotFoundError):
        await client.get_dataset_meta("missing")
    assert route.call_count == 1


async def test_async_delete_dataset_retries_transient(
    client: AsyncDagnamClient, mock: RespxMockRouter, monkeypatch: PytestMonkeyPatch
) -> None:
    """DELETE is idempotent, so a transient status retries."""

    async def _no_sleep(_d: float) -> None: ...

    monkeypatch.setattr(client, "_async_sleep", _no_sleep)
    monkeypatch.setattr(client, "_rng", lambda: 1.0)
    mock.delete("/api/v1/datasets/ds1").mock(
        side_effect=[httpx.Response(503, json={}), httpx.Response(204)]
    )
    assert await client.delete_dataset("ds1") is None
