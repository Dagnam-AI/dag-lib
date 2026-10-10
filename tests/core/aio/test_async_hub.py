"""Async hub client mixin."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import (
    APIError,
    HubModelNotFoundError,
    ResponseError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, RespxMockRouter

API = "https://api.test"

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------- hub


async def test_async_list_hub_models(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/hub/models").mock(return_value=httpx.Response(200, json={"items": []}))
    assert await client.list_hub_models() == {"items": []}


async def test_async_get_hub_model_404(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/hub/models/missing").mock(return_value=httpx.Response(404))
    with pytest.raises(HubModelNotFoundError):
        await client.get_hub_model("missing")


async def test_async_create_hub_model(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.post("/api/v1/hub/models").mock(return_value=httpx.Response(200, json={"id": "m1"}))
    assert await client.create_hub_model({}) == {"id": "m1"}


async def test_async_update_hub_model(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.put("/api/v1/hub/models/m1").mock(return_value=httpx.Response(200, json={"id": "m1"}))
    assert await client.update_hub_model("m1", {}) == {"id": "m1"}


async def test_async_delete_hub_model(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.delete("/api/v1/hub/models/m1").mock(return_value=httpx.Response(204))
    assert await client.delete_hub_model("m1") is None


async def test_async_hub_misc(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/hub/models/m1/files").mock(return_value=httpx.Response(200, json={}))
    mock.post("/api/v1/hub/models/m1/download").mock(return_value=httpx.Response(200, json={}))
    mock.get("/api/v1/hub/models/m1/versions").mock(return_value=httpx.Response(200, json=[]))
    mock.post("/api/v1/hub/models/m1/versions").mock(return_value=httpx.Response(200, json={}))
    mock.post("/api/v1/hub/models/m1/star").mock(return_value=httpx.Response(200, json={}))
    mock.delete("/api/v1/hub/models/m1/star").mock(return_value=httpx.Response(200, json={}))
    mock.post("/api/v1/hub/models/m1/fork").mock(return_value=httpx.Response(200, json={}))
    mock.get("/api/v1/hub/models/m1/reviews").mock(return_value=httpx.Response(200, json={}))
    mock.post("/api/v1/hub/models/m1/reviews").mock(return_value=httpx.Response(200, json={}))
    mock.post("/api/v1/hub/models/m1/use-in-studio").mock(return_value=httpx.Response(200, json={}))
    mock.get("/api/v1/hub/categories").mock(return_value=httpx.Response(200, json=[]))
    mock.get("/api/v1/hub/featured").mock(return_value=httpx.Response(200, json=[]))
    mock.get("/api/v1/hub/trending").mock(return_value=httpx.Response(200, json=[]))
    mock.get("/api/v1/hub/models/starred").mock(return_value=httpx.Response(200, json={}))

    await client.list_hub_model_files("m1")
    await client.download_hub_model("m1", file_id="f1")
    await client.download_hub_model("m1")
    await client.list_hub_model_versions("m1")
    await client.create_hub_model_version("m1", {})
    await client.star_hub_model("m1")
    await client.unstar_hub_model("m1")
    await client.fork_hub_model("m1")
    await client.list_hub_model_reviews("m1")
    await client.add_hub_model_review("m1", {})
    await client.use_hub_model_in_studio("m1")
    await client.list_hub_categories()
    await client.get_hub_featured()
    await client.get_hub_trending()
    await client.list_hub_starred()


async def test_async_hub_text_response(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/hub/categories").mock(
        return_value=httpx.Response(200, text="plain", headers={"Content-Type": "text/plain"})
    )
    assert await client.list_hub_categories() == "plain"


async def test_async_hub_empty_response(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/hub/categories").mock(return_value=httpx.Response(204))
    assert await client.list_hub_categories() is None


# ---------------------------------------------------------------- download


DOWNLOAD = "/api/v1/hub/models/m1/download"
CONTENT_F1 = "/api/v1/hub/models/m1/files/f1/content"
CONTENT_F2 = "/api/v1/hub/models/m1/files/f2/content"


def _links(*entries: tuple[str, str, str]) -> dict[str, object]:
    """The platform's download answer: one signed content link per file."""
    return {
        "files": [
            {"file_id": fid, "file_name": name, "size": 1, "url": url} for fid, name, url in entries
        ],
        "download_all_url": None,
    }


async def test_async_download_hub_model_posts(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post(DOWNLOAD).mock(return_value=httpx.Response(200, json=_links()))
    assert await client.download_hub_model("m1", file_id="f1") == _links()
    assert route.calls[0].request.url.query == b"file_id=f1"


async def test_async_download_hub_model_files_asks_for_a_fresh_link_per_file(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    """A link lasts minutes: each file's link is issued just before its fetch, not in one batch."""
    links = mock.post(DOWNLOAD).mock(
        side_effect=[
            httpx.Response(
                200,
                json=_links(
                    ("f1", "model.safetensors", f"{CONTENT_F1}?token=batch1"),
                    ("f2", "head.onnx", f"{CONTENT_F2}?token=batch2"),
                ),
            ),
            httpx.Response(
                200, json=_links(("f1", "model.safetensors", f"{CONTENT_F1}?token=fresh1"))
            ),
            httpx.Response(
                200, json=_links(("f2", "head.onnx", f"{API}{CONTENT_F2}?token=fresh2"))
            ),
        ]
    )
    first = mock.get(CONTENT_F1).mock(return_value=httpx.Response(200, content=b"weights"))
    second = mock.get(CONTENT_F2).mock(return_value=httpx.Response(200, content=b"onnx"))
    saved = await client.download_hub_model_files("m1", tmp_path)
    assert saved == [tmp_path / "model.safetensors", tmp_path / "head.onnx"]
    assert saved[0].read_bytes() == b"weights"
    assert saved[1].read_bytes() == b"onnx"
    assert [call.request.url.query for call in links.calls] == [b"", b"file_id=f1", b"file_id=f2"]
    assert first.calls[0].request.url.query == b"token=fresh1"
    assert second.calls[0].request.url.query == b"token=fresh2"
    # The second link is asked for only after the first file is on disk.
    assert links.calls[2].request.url.query == b"file_id=f2"
    assert first.call_count == 1
    assert second.call_count == 1
    assert first.calls[0].request.headers["authorization"] == "Bearer k"
    assert second.calls[0].request.headers["authorization"] == "Bearer k"


async def test_async_download_hub_model_files_issues_the_next_link_after_the_previous_file_is_saved(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    """The ordering the fresh-link rule exists for: file two's link is requested once file one is on disk."""
    on_disk_when_asked: list[bool] = []

    entries = {
        "f1": ("f1", "model.safetensors", f"{CONTENT_F1}?token=t1"),
        "f2": ("f2", "head.onnx", f"{CONTENT_F2}?token=t2"),
    }

    def issue(request: httpx.Request) -> httpx.Response:
        asked = request.url.params.get("file_id")
        if asked == "f2":
            on_disk_when_asked.append((tmp_path / "model.safetensors").exists())
        chosen = [entries[asked]] if asked else list(entries.values())
        return httpx.Response(200, json=_links(*chosen))

    mock.post(DOWNLOAD).mock(side_effect=issue)
    mock.get(CONTENT_F1).mock(return_value=httpx.Response(200, content=b"weights"))
    mock.get(CONTENT_F2).mock(return_value=httpx.Response(200, content=b"onnx"))
    await client.download_hub_model_files("m1", tmp_path)
    assert on_disk_when_asked == [True]


async def test_async_download_hub_model_files_foreign_absolute_link_is_fetched_without_api_key(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    """An absolute link on another host (object storage) never receives the key."""
    store_url = "https://bucket.example.com/m1/f1?X-Amz-Signature=abc"
    mock.post(DOWNLOAD).mock(
        return_value=httpx.Response(200, json=_links(("f1", "model.safetensors", store_url)))
    )
    store = mock.get(store_url).mock(return_value=httpx.Response(200, content=b"weights"))
    saved = await client.download_hub_model_files("m1", tmp_path)
    assert saved == [tmp_path / "model.safetensors"]
    assert saved[0].read_bytes() == b"weights"
    assert "authorization" not in store.calls[0].request.headers


@pytest.mark.parametrize("status", [302, 307])
async def test_async_download_hub_model_files_follows_redirect_without_api_key(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path, status: int
) -> None:
    presigned = "https://bucket.example.com/m1/f1?X-Amz-Signature=abc"
    mock.post(DOWNLOAD).mock(
        return_value=httpx.Response(
            200, json=_links(("f1", "model.safetensors", f"{CONTENT_F1}?token=t1"))
        )
    )
    mock.get(CONTENT_F1).mock(return_value=httpx.Response(status, headers={"location": presigned}))
    store = mock.get(presigned).mock(return_value=httpx.Response(200, content=b"weights"))
    saved = await client.download_hub_model_files("m1", tmp_path, file_id="f1")
    assert saved == [tmp_path / "model.safetensors"]
    assert saved[0].read_bytes() == b"weights"
    assert "authorization" not in store.calls[0].request.headers


@pytest.mark.parametrize(
    ("hostile", "landed"),
    [
        ("../../etc/passwd", "passwd"),
        ("/etc/passwd", "passwd"),
        ("dir/weights.safetensors", "weights.safetensors"),
        ("..", "f1"),
    ],
)
async def test_async_download_hub_model_files_hostile_name_lands_inside_dest_dir(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path, hostile: str, landed: str
) -> None:
    mock.post(DOWNLOAD).mock(
        return_value=httpx.Response(200, json=_links(("f1", hostile, f"{CONTENT_F1}?token=t1")))
    )
    mock.get(CONTENT_F1).mock(return_value=httpx.Response(200, content=b"x"))
    saved = await client.download_hub_model_files("m1", tmp_path, file_id="f1")
    assert saved == [tmp_path / landed]
    assert saved[0].read_bytes() == b"x"


async def test_async_download_hub_model_files_hostile_file_id_fallback_lands_inside_dest_dir(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.post(DOWNLOAD).mock(
        return_value=httpx.Response(200, json=_links(("../../x", "..", f"{CONTENT_F1}?token=t1")))
    )
    mock.get(CONTENT_F1).mock(return_value=httpx.Response(200, content=b"x"))
    saved = await client.download_hub_model_files("m1", tmp_path, file_id="f1")
    assert saved == [tmp_path / "x"]
    assert not (tmp_path.parent.parent / "x").exists()


async def test_async_download_hub_model_files_joins_a_relative_redirect_location(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.post(DOWNLOAD).mock(
        return_value=httpx.Response(
            200, json=_links(("f1", "model.safetensors", f"{CONTENT_F1}?token=t1"))
        )
    )
    mock.get(CONTENT_F1).mock(
        return_value=httpx.Response(302, headers={"location": "/storage/m1/f1?sig=x"})
    )
    store = mock.get("/storage/m1/f1").mock(return_value=httpx.Response(200, content=b"weights"))
    saved = await client.download_hub_model_files("m1", tmp_path, file_id="f1")
    assert saved[0].read_bytes() == b"weights"
    assert str(store.calls[0].request.url) == f"{API}/storage/m1/f1?sig=x"
    assert "authorization" not in store.calls[0].request.headers


async def test_async_download_hub_model_files_content_404_raises_not_found(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.post(DOWNLOAD).mock(
        return_value=httpx.Response(
            200, json=_links(("f1", "model.safetensors", f"{CONTENT_F1}?token=t1"))
        )
    )
    mock.get(CONTENT_F1).mock(return_value=httpx.Response(404, json={"detail": "File not found"}))
    with pytest.raises(HubModelNotFoundError):
        await client.download_hub_model_files("m1", tmp_path)


async def test_async_download_hub_model_files_redirect_target_404_raises(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    presigned = "https://bucket.example.com/m1/f1?X-Amz-Signature=abc"
    mock.post(DOWNLOAD).mock(
        return_value=httpx.Response(
            200, json=_links(("f1", "model.safetensors", f"{CONTENT_F1}?token=t1"))
        )
    )
    mock.get(CONTENT_F1).mock(return_value=httpx.Response(302, headers={"location": presigned}))
    mock.get(presigned).mock(return_value=httpx.Response(404, text="gone"))
    with pytest.raises(HubModelNotFoundError):
        await client.download_hub_model_files("m1", tmp_path)


async def test_async_download_hub_model_files_redirect_without_location_raises(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.post(DOWNLOAD).mock(
        return_value=httpx.Response(
            200, json=_links(("f1", "model.safetensors", f"{CONTENT_F1}?token=t1"))
        )
    )
    mock.get(CONTENT_F1).mock(return_value=httpx.Response(302))
    with pytest.raises(APIError):
        await client.download_hub_model_files("m1", tmp_path)


async def test_async_download_hub_model_files_link_missing_raises(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.post(DOWNLOAD).mock(
        return_value=httpx.Response(200, json={"files": [{"file_id": "f1", "file_name": "w"}]})
    )
    with pytest.raises(ResponseError, match="content link"):
        await client.download_hub_model_files("m1", tmp_path)


async def test_async_download_hub_model_files_connect_error_and_timeout(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.post(DOWNLOAD).mock(
        return_value=httpx.Response(
            200, json=_links(("f1", "model.safetensors", f"{CONTENT_F1}?token=secret"))
        )
    )
    content = mock.get(CONTENT_F1).mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(APIError, match="Connection failed"):
        await client.download_hub_model_files("m1", tmp_path)
    content.mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(APIError, match="Request timed out"):
        await client.download_hub_model_files("m1", tmp_path)


# ---------------------------------------------------------------- file upload


async def test_async_hub_upload_model_file(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    f = tmp_path / "weights.bin"
    f.write_bytes(b"\x00\x01\x02")
    route = mock.post("/api/v1/hub/models/m1/files").mock(
        return_value=httpx.Response(
            201, json={"id": "f1", "model_id": "m1", "file_name": "weights.bin"}
        )
    )
    out = await client.upload_model_file("m1", str(f))
    assert out["id"] == "f1"
    assert b'name="file"' in route.calls[0].request.content
    assert route.calls[0].request.extensions["timeout"] == {
        "connect": 10.0,
        "read": 900.0,
        "write": 900.0,
        "pool": 900.0,
    }


async def test_async_hub_upload_model_file_404(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    f = tmp_path / "weights.bin"
    f.write_bytes(b"\x00")
    mock.post("/api/v1/hub/models/missing/files").mock(return_value=httpx.Response(404))
    with pytest.raises(HubModelNotFoundError):
        await client.upload_model_file("missing", str(f))


# ---------------------------------------------------------------- finalize


async def test_async_finalize_hub_model(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.post("/api/v1/hub/models/m1/finalize").mock(
        return_value=httpx.Response(200, json={"id": "m1", "status": "published"})
    )
    out = await client.finalize_hub_model("m1")
    assert out["status"] == "published"


async def test_async_finalize_hub_model_404(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/hub/models/m1/finalize").mock(return_value=httpx.Response(404))
    with pytest.raises(HubModelNotFoundError):
        await client.finalize_hub_model("m1")


# --------------------------------------------------------------------------- transient retry


async def test_async_get_hub_model_retries_transient(
    client: AsyncDagnamClient, mock: RespxMockRouter, monkeypatch: PytestMonkeyPatch
) -> None:
    async def _no_sleep(_d: float) -> None: ...

    monkeypatch.setattr(client, "_async_sleep", _no_sleep)
    monkeypatch.setattr(client, "_rng", lambda: 1.0)
    mock.get("/api/v1/hub/models/m1").mock(
        side_effect=[
            httpx.Response(503, json={}),
            httpx.Response(200, json={"id": "m1"}),
        ]
    )
    model = await client.get_hub_model("m1")
    assert model["id"] == "m1"


async def test_async_get_hub_model_404_not_retried(
    client: AsyncDagnamClient, mock: RespxMockRouter, monkeypatch: PytestMonkeyPatch
) -> None:
    async def _no_sleep(_d: float) -> None: ...

    monkeypatch.setattr(client, "_async_sleep", _no_sleep)
    route = mock.get("/api/v1/hub/models/missing").mock(return_value=httpx.Response(404, json={}))
    with pytest.raises(HubModelNotFoundError):
        await client.get_hub_model("missing")
    assert route.call_count == 1
