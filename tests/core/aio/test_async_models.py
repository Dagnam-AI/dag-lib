"""Wire-level coverage for the async AsyncModelsMixin (model registry).

Async mirror of ``tests/core/client/test_sync_models.py``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import (
    APIError,
    AuthError,
    ModelError,
    ModelNotFoundError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, RespxMockRouter

API = "https://api.test"

pytestmark = pytest.mark.anyio


# ------------------------------------------------------------------- entries


async def test_async_create_model_entry(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.post("/api/v1/models").mock(
        return_value=httpx.Response(201, json={"id": "m1", "slug": "tiny-chat"})
    )
    result = await client.create_model_entry({"name": "tiny-chat", "slug": "tiny-chat"})
    assert result == {"id": "m1", "slug": "tiny-chat"}


async def test_async_create_model_entry_sends_idempotency_key(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post("/api/v1/models").mock(return_value=httpx.Response(201, json={"id": "m1"}))
    await client.create_model_entry({"name": "x", "slug": "x"})
    assert route.calls[-1].request.headers.get("Idempotency-Key")


async def test_async_create_model_entry_409_duplicate_slug(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/models").mock(return_value=httpx.Response(409, text="dup"))
    with pytest.raises(ModelError):
        await client.create_model_entry({"name": "x", "slug": "dup"})


async def test_async_create_model_entry_404_project_not_found(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/models").mock(return_value=httpx.Response(404, text="Project not found"))
    with pytest.raises(ModelError) as exc_info:
        await client.create_model_entry({"name": "x", "slug": "y", "project_id": "missing"})
    assert not isinstance(exc_info.value, ModelNotFoundError)


async def test_async_get_model_entry(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/models/m1").mock(return_value=httpx.Response(200, json={"id": "m1"}))
    assert await client.get_model_entry("m1") == {"id": "m1"}


async def test_async_get_model_entry_404(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/models/missing").mock(return_value=httpx.Response(404))
    with pytest.raises(ModelNotFoundError):
        await client.get_model_entry("missing")


async def test_async_get_model_entry_401(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/models/m1").mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await client.get_model_entry("m1")


async def test_async_get_model_entry_500(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/models/m1").mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(APIError):
        await client.get_model_entry("m1")


async def test_async_get_model_entry_non_json_body_raises(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/models/m1").mock(
        return_value=httpx.Response(200, text="not json", headers={"Content-Type": "text/plain"})
    )
    with pytest.raises(TypeError):
        await client.get_model_entry("m1")


async def test_async_list_model_entries_returns_array(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/models").mock(
        return_value=httpx.Response(200, json=[{"id": "m1"}, {"id": "m2"}])
    )
    assert await client.list_model_entries() == [{"id": "m1"}, {"id": "m2"}]


async def test_async_list_model_entries_query_params(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.get("/api/v1/models").mock(return_value=httpx.Response(200, json=[]))
    await client.list_model_entries(search="tiny")
    assert route.calls[-1].request.url.params["search"] == "tiny"


async def test_async_update_model_entry(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.patch("/api/v1/models/m1").mock(return_value=httpx.Response(200, json={"id": "m1"}))
    assert await client.update_model_entry("m1", {"name": "y"}) == {"id": "m1"}


async def test_async_update_model_entry_404(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.patch("/api/v1/models/missing").mock(return_value=httpx.Response(404))
    with pytest.raises(ModelNotFoundError):
        await client.update_model_entry("missing", {"name": "y"})


async def test_async_delete_model_entry(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.delete("/api/v1/models/m1").mock(return_value=httpx.Response(204))
    assert await client.delete_model_entry("m1") is None


async def test_async_delete_model_entry_404(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.delete("/api/v1/models/missing").mock(return_value=httpx.Response(404))
    with pytest.raises(ModelNotFoundError):
        await client.delete_model_entry("missing")


# ------------------------------------------------------------------ versions


async def test_async_create_model_version(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.post("/api/v1/models/m1/versions").mock(
        return_value=httpx.Response(201, json={"id": "v1"})
    )
    assert await client.create_model_version("m1", {"origin": "trained"}) == {"id": "v1"}


async def test_async_create_model_version_sends_idempotency_key(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post("/api/v1/models/m1/versions").mock(
        return_value=httpx.Response(201, json={"id": "v1"})
    )
    await client.create_model_version("m1", {"origin": "trained"})
    assert route.calls[-1].request.headers.get("Idempotency-Key")


async def test_async_create_model_version_404(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/models/missing/versions").mock(return_value=httpx.Response(404))
    with pytest.raises(ModelNotFoundError):
        await client.create_model_version("missing", {"origin": "trained"})


async def test_async_list_model_versions_returns_array(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/models/m1/versions").mock(
        return_value=httpx.Response(200, json=[{"id": "v1"}])
    )
    assert await client.list_model_versions("m1") == [{"id": "v1"}]


async def test_async_get_model_version(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/model-versions/v1").mock(
        return_value=httpx.Response(200, json={"id": "v1", "status": "ready"})
    )
    assert await client.get_model_version("v1") == {"id": "v1", "status": "ready"}


async def test_async_get_model_version_404(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/model-versions/missing").mock(return_value=httpx.Response(404))
    with pytest.raises(ModelNotFoundError):
        await client.get_model_version("missing")


async def test_async_get_model_version_lineage(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/model-versions/v1/lineage").mock(
        return_value=httpx.Response(200, json={"version_id": "v1", "edges": []})
    )
    result = await client.get_model_version_lineage("v1")
    assert result == {"version_id": "v1", "edges": []}


async def test_async_get_model_version_lineage_404(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/model-versions/missing/lineage").mock(return_value=httpx.Response(404))
    with pytest.raises(ModelNotFoundError):
        await client.get_model_version_lineage("missing")


# ----------------------------------------------------------------- artifacts


async def test_async_initiate_model_artifact(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/model-versions/v1/artifacts:initiate").mock(
        return_value=httpx.Response(
            201, json={"artifact_id": "a1", "upload_method": "POST", "upload_url": "/x"}
        )
    )
    result = await client.initiate_model_artifact(
        "v1", {"logical_key": "weights", "size_bytes": 10}
    )
    assert result["artifact_id"] == "a1"


async def test_async_initiate_model_artifact_sends_idempotency_key(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post("/api/v1/model-versions/v1/artifacts:initiate").mock(
        return_value=httpx.Response(
            201, json={"artifact_id": "a1", "upload_method": "POST", "upload_url": "/x"}
        )
    )
    await client.initiate_model_artifact("v1", {"logical_key": "weights", "size_bytes": 10})
    assert route.calls[-1].request.headers.get("Idempotency-Key")


async def test_async_initiate_model_artifact_404_not_draft(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/model-versions/v1/artifacts:initiate").mock(return_value=httpx.Response(404))
    with pytest.raises(ModelNotFoundError):
        await client.initiate_model_artifact("v1", {"logical_key": "weights", "size_bytes": 10})


async def test_async_complete_model_artifact(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    sha = hashlib.sha256(b"weights").hexdigest()
    mock.post("/api/v1/model-versions/v1/artifacts/a1/complete").mock(
        return_value=httpx.Response(200, json={"id": "a1", "verification_status": "verified"})
    )
    result = await client.complete_model_artifact("v1", "a1", {"sha256": sha, "size_bytes": 7})
    assert result["verification_status"] == "verified"


async def test_async_complete_model_artifact_404(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/model-versions/v1/artifacts/missing/complete").mock(
        return_value=httpx.Response(404)
    )
    with pytest.raises(ModelNotFoundError):
        await client.complete_model_artifact("v1", "missing", {"sha256": "x", "size_bytes": 1})


async def test_async_finalize_model_version(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/model-versions/v1/finalize").mock(
        return_value=httpx.Response(200, json={"id": "v1", "status": "ready"})
    )
    result = await client.finalize_model_version("v1")
    assert result["status"] == "ready"


async def test_async_finalize_model_version_422_invalid_state(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/model-versions/v1/finalize").mock(
        return_value=httpx.Response(422, text="cannot finalize a version with zero artifacts")
    )
    with pytest.raises(ModelError):
        await client.finalize_model_version("v1")


async def test_async_finalize_model_version_404(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/model-versions/missing/finalize").mock(return_value=httpx.Response(404))
    with pytest.raises(ModelNotFoundError):
        await client.finalize_model_version("missing")


async def test_async_get_task_contract(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/task-contracts/chat/versions/v1").mock(
        return_value=httpx.Response(200, json={"key": "chat", "version": "v1"})
    )
    assert await client.get_task_contract("chat", "v1") == {"key": "chat", "version": "v1"}


async def test_async_get_task_contract_404(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    """No model/version id is in play for a task-contract lookup -> bare ModelError."""
    mock.get("/api/v1/task-contracts/missing/versions/v1").mock(return_value=httpx.Response(404))
    with pytest.raises(ModelError) as exc_info:
        await client.get_task_contract("missing", "v1")
    assert not isinstance(exc_info.value, ModelNotFoundError)


async def test_async_list_model_version_artifacts_returns_array(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/model-versions/v1/artifacts").mock(
        return_value=httpx.Response(200, json=[{"id": "a1"}, {"id": "a2"}])
    )
    assert await client.list_model_version_artifacts("v1") == [{"id": "a1"}, {"id": "a2"}]


async def test_async_list_model_version_artifacts_404(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/model-versions/missing/artifacts").mock(return_value=httpx.Response(404))
    with pytest.raises(ModelNotFoundError):
        await client.list_model_version_artifacts("missing")


# --------------------------------------------------------- direct file upload


async def test_async_upload_model_artifact_direct(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    f = tmp_path / "weights.bin"
    f.write_bytes(b"\x00\x01\x02")
    route = mock.post("/api/v1/model-versions/v1/artifacts/a1/upload").mock(
        return_value=httpx.Response(204)
    )
    await client.upload_model_artifact_direct("/api/v1/model-versions/v1/artifacts/a1/upload", f)
    assert b'name="file"' in route.calls[0].request.content


async def test_async_upload_model_artifact_direct_404(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    f = tmp_path / "weights.bin"
    f.write_bytes(b"\x00")
    mock.post("/api/v1/model-versions/v1/artifacts/missing/upload").mock(
        return_value=httpx.Response(404)
    )
    with pytest.raises(ModelError):
        await client.upload_model_artifact_direct(
            "/api/v1/model-versions/v1/artifacts/missing/upload", f
        )


# ------------------------------------------------------- artifact download


async def test_async_download_model_artifact_direct(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    url = "/api/v1/model-versions/v1/artifacts/a1/download"
    mock.get(url).mock(
        return_value=httpx.Response(200, content=b"weights", headers={"x-checksum-sha256": "abc"})
    )
    dest = tmp_path / "a1.bin"
    written, checksum = await client.download_model_artifact("v1", "a1", dest)
    assert written.read_bytes() == b"weights"
    assert checksum == "abc"


async def test_async_download_model_artifact_307_redirect_to_presigned(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    """A 307 to a presigned URL is followed; the API key is NOT forwarded."""
    url = "/api/v1/model-versions/v1/artifacts/a1/download"
    presigned = "https://bucket.s3.example.com/a1?sig=xyz"
    mock.get(url).mock(return_value=httpx.Response(307, headers={"location": presigned}))
    presigned_route = mock.get(presigned).mock(
        return_value=httpx.Response(
            200, content=b"weights", headers={"x-checksum-sha256": "sha-from-s3"}
        )
    )
    dest = tmp_path / "a1.bin"
    written, checksum = await client.download_model_artifact("v1", "a1", dest)
    assert written.read_bytes() == b"weights"
    assert checksum == "sha-from-s3"
    presigned_req = presigned_route.calls[-1].request
    assert "authorization" not in presigned_req.headers


async def test_async_download_model_artifact_308_redirect(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    url = "/api/v1/model-versions/v1/artifacts/a1/download"
    presigned = "https://bucket.s3.example.com/a1?sig=abc"
    mock.get(url).mock(return_value=httpx.Response(308, headers={"location": presigned}))
    mock.get(presigned).mock(return_value=httpx.Response(200, content=b"bytes"))
    dest = tmp_path / "a1.bin"
    written, checksum = await client.download_model_artifact("v1", "a1", dest)
    assert written.read_bytes() == b"bytes"
    assert checksum is None


async def test_async_download_model_artifact_redirect_checksum_from_original(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    """The checksum header on the redirect response itself is honored."""
    url = "/api/v1/model-versions/v1/artifacts/a1/download"
    presigned = "https://bucket.s3.example.com/a1?sig=def"
    mock.get(url).mock(
        return_value=httpx.Response(
            307, headers={"location": presigned, "x-checksum-sha256": "sha-from-api"}
        )
    )
    mock.get(presigned).mock(return_value=httpx.Response(200, content=b"weights"))
    dest = tmp_path / "a1.bin"
    _written, checksum = await client.download_model_artifact("v1", "a1", dest)
    assert checksum == "sha-from-api"


async def test_async_download_model_artifact_redirect_missing_location(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    url = "/api/v1/model-versions/v1/artifacts/a1/download"
    mock.get(url).mock(return_value=httpx.Response(307))
    with pytest.raises(APIError):
        await client.download_model_artifact("v1", "a1", tmp_path / "x")


async def test_async_download_model_artifact_presigned_connecterror(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    url = "/api/v1/model-versions/v1/artifacts/a1/download"
    presigned = "https://bucket.s3.example.com/a1?sig=xyz"
    mock.get(url).mock(return_value=httpx.Response(307, headers={"location": presigned}))
    mock.get(presigned).mock(side_effect=httpx.ConnectError("nope"))
    with pytest.raises(APIError, match="Connection failed"):
        await client.download_model_artifact("v1", "a1", tmp_path / "x")


async def test_async_download_model_artifact_presigned_timeout(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    url = "/api/v1/model-versions/v1/artifacts/a1/download"
    presigned = "https://bucket.s3.example.com/a1?sig=xyz"
    mock.get(url).mock(return_value=httpx.Response(307, headers={"location": presigned}))
    mock.get(presigned).mock(side_effect=httpx.TimeoutException("slow"))
    with pytest.raises(APIError, match="Request timed out"):
        await client.download_model_artifact("v1", "a1", tmp_path / "x")


async def test_async_download_model_artifact_401(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    url = "/api/v1/model-versions/v1/artifacts/a1/download"
    mock.get(url).mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await client.download_model_artifact("v1", "a1", tmp_path / "x")


async def test_async_download_model_artifact_404(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    url = "/api/v1/model-versions/v1/artifacts/missing/download"
    mock.get(url).mock(return_value=httpx.Response(404))
    with pytest.raises(ModelNotFoundError):
        await client.download_model_artifact("v1", "missing", tmp_path / "x")


async def test_async_download_model_artifact_500(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    url = "/api/v1/model-versions/v1/artifacts/a1/download"
    mock.get(url).mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(APIError):
        await client.download_model_artifact("v1", "a1", tmp_path / "x")


async def test_async_download_model_artifact_presigned_non_success_status(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    url = "/api/v1/model-versions/v1/artifacts/a1/download"
    presigned = "https://bucket.s3.example.com/a1?sig=xyz"
    mock.get(url).mock(return_value=httpx.Response(307, headers={"location": presigned}))
    mock.get(presigned).mock(return_value=httpx.Response(404))
    with pytest.raises(ModelNotFoundError):
        await client.download_model_artifact("v1", "a1", tmp_path / "x")


# --------------------------------------------------------------------------- transient retry


async def test_async_get_model_entry_retries_transient(
    client: AsyncDagnamClient, mock: RespxMockRouter, monkeypatch: PytestMonkeyPatch
) -> None:
    async def _no_sleep(_d: float) -> None: ...

    monkeypatch.setattr(client, "_async_sleep", _no_sleep)
    monkeypatch.setattr(client, "_rng", lambda: 1.0)
    mock.get("/api/v1/models/m1").mock(
        side_effect=[httpx.Response(503, json={}), httpx.Response(200, json={"id": "m1"})]
    )
    model = await client.get_model_entry("m1")
    assert model["id"] == "m1"


async def test_async_get_model_entry_404_not_retried(
    client: AsyncDagnamClient, mock: RespxMockRouter, monkeypatch: PytestMonkeyPatch
) -> None:
    async def _no_sleep(_d: float) -> None: ...

    monkeypatch.setattr(client, "_async_sleep", _no_sleep)
    route = mock.get("/api/v1/models/missing").mock(return_value=httpx.Response(404, json={}))
    with pytest.raises(ModelNotFoundError):
        await client.get_model_entry("missing")
    assert route.call_count == 1
