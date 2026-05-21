"""Coverage for the async client mixins using respx.

respx intercepts httpx calls at the transport layer, so all async client
methods exercise the real request-construction path without hitting a network.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.aio.base import (
    BaseAsyncDagnamClient,
    _parse_cd,
    _raise_for_job,
    _sanitize_filename,
)
from dagnam._core.exceptions import (
    APIError,
    AuthError,
    CheckpointNotFoundError,
    DatasetNotFoundError,
    DeploymentNotFoundError,
    HubModelNotFoundError,
    ProjectNotFoundError,
    TrainingJobNotFoundError,
)

API = "https://api.test"

pytestmark = pytest.mark.anyio


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


# ---------------------------------------------------------------- base helpers


def test_parse_cd_quoted() -> None:
    assert _parse_cd('attachment; filename="x.csv"') == "x.csv"


def test_parse_cd_unquoted() -> None:
    assert _parse_cd("attachment; filename=x.csv") == "x.csv"


def test_parse_cd_default() -> None:
    assert _parse_cd(None) == "data"
    assert _parse_cd("inline") == "data"


def test_sanitize_filename_rejects_unsafe() -> None:
    for bad in ("../x", "C:foo", "", ".", "..", "CON.txt"):
        with pytest.raises(ValueError):
            _sanitize_filename(bad)


def test_raise_for_job_2xx_returns():
    r = httpx.Response(200)
    _raise_for_job(r, "job1")


def test_raise_for_job_401():
    with pytest.raises(AuthError):
        _raise_for_job(httpx.Response(401), "job1")


def test_raise_for_job_404():
    with pytest.raises(TrainingJobNotFoundError):
        _raise_for_job(httpx.Response(404), "job1")


def test_raise_for_job_500():
    with pytest.raises(APIError):
        _raise_for_job(httpx.Response(500, text="boom"), "job1")


async def test_base_request_connection_error_wraps():
    base = BaseAsyncDagnamClient(API, "k")
    try:
        with respx.mock(base_url=API) as r:
            r.get("/x").mock(side_effect=httpx.ConnectError("nope"))
            with pytest.raises(APIError, match="Connection failed"):
                await base._request("GET", "/x")
    finally:
        await base._client.aclose()


async def test_base_request_timeout_wraps():
    base = BaseAsyncDagnamClient(API, "k")
    try:
        with respx.mock(base_url=API) as r:
            r.get("/x").mock(side_effect=httpx.ReadTimeout("slow"))
            with pytest.raises(APIError, match="Request timed out"):
                await base._request("GET", "/x")
    finally:
        await base._client.aclose()


# ---------------------------------------------------------------- hub


async def test_async_list_hub_models(client, mock):
    mock.get("/api/v1/hub/models").mock(return_value=httpx.Response(200, json={"items": []}))
    assert await client.list_hub_models() == {"items": []}


async def test_async_get_hub_model_404(client, mock):
    mock.get("/api/v1/hub/models/missing").mock(return_value=httpx.Response(404))
    with pytest.raises(HubModelNotFoundError):
        await client.get_hub_model("missing")


async def test_async_create_hub_model(client, mock):
    mock.post("/api/v1/hub/models").mock(return_value=httpx.Response(200, json={"id": "m1"}))
    assert await client.create_hub_model({}) == {"id": "m1"}


async def test_async_update_hub_model(client, mock):
    mock.put("/api/v1/hub/models/m1").mock(return_value=httpx.Response(200, json={"id": "m1"}))
    assert await client.update_hub_model("m1", {}) == {"id": "m1"}


async def test_async_delete_hub_model(client, mock):
    mock.delete("/api/v1/hub/models/m1").mock(return_value=httpx.Response(204))
    assert await client.delete_hub_model("m1") is None


async def test_async_hub_misc(client, mock):
    mock.get("/api/v1/hub/models/m1/files").mock(return_value=httpx.Response(200, json={}))
    mock.get("/api/v1/hub/models/m1/download").mock(return_value=httpx.Response(200, json={}))
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
    mock.get("/api/v1/hub/starred").mock(return_value=httpx.Response(200, json={}))

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


async def test_async_hub_text_response(client, mock):
    mock.get("/api/v1/hub/categories").mock(
        return_value=httpx.Response(200, text="plain", headers={"Content-Type": "text/plain"})
    )
    assert await client.list_hub_categories() == "plain"


async def test_async_hub_empty_response(client, mock):
    mock.get("/api/v1/hub/categories").mock(return_value=httpx.Response(204))
    assert await client.list_hub_categories() is None


# ---------------------------------------------------------------- projects


async def test_async_projects_full_surface(client, mock):
    mock.get("/api/v1/projects").mock(return_value=httpx.Response(200, json={"items": []}))
    mock.get("/api/v1/projects/p1").mock(return_value=httpx.Response(200, json={"id": "p1"}))
    mock.post("/api/v1/projects").mock(return_value=httpx.Response(200, json={"id": "p1"}))
    mock.put("/api/v1/projects/p1").mock(return_value=httpx.Response(200, json={"id": "p1"}))
    mock.delete("/api/v1/projects/p1").mock(return_value=httpx.Response(204))
    mock.post("/api/v1/projects/p1/duplicate").mock(return_value=httpx.Response(200, json={}))
    mock.post("/api/v1/projects/p1/architecture").mock(return_value=httpx.Response(200, json={}))
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


async def test_async_get_project_404(client, mock):
    mock.get("/api/v1/projects/missing").mock(return_value=httpx.Response(404))
    with pytest.raises(ProjectNotFoundError):
        await client.get_project("missing")


async def test_async_projects_text_response(client, mock):
    mock.get("/api/v1/projects").mock(
        return_value=httpx.Response(200, text="plain", headers={"Content-Type": "text/plain"})
    )
    assert await client.list_projects() == "plain"


async def test_async_projects_empty_response(client, mock):
    mock.get("/api/v1/projects").mock(return_value=httpx.Response(204))
    assert await client.list_projects() is None


# ---------------------------------------------------------------- deployments


async def test_async_deployments_full_surface(client, mock):
    mock.get("/api/v1/deployments").mock(return_value=httpx.Response(200, json={"items": []}))
    mock.get("/api/v1/deployments/dep1").mock(return_value=httpx.Response(200, json={"id": "dep1"}))
    mock.post("/api/v1/deployments").mock(return_value=httpx.Response(200, json={}))
    mock.put("/api/v1/deployments/dep1").mock(return_value=httpx.Response(200, json={}))
    mock.delete("/api/v1/deployments/dep1").mock(return_value=httpx.Response(204))
    mock.post("/api/v1/deployments/dep1/pause").mock(return_value=httpx.Response(200, json={}))
    mock.post("/api/v1/deployments/dep1/resume").mock(return_value=httpx.Response(200, json={}))
    mock.put("/api/v1/deployments/dep1/scale").mock(return_value=httpx.Response(200, json={}))
    mock.post("/api/v1/deployments/dep1/rollback").mock(return_value=httpx.Response(200, json={}))
    mock.get("/api/v1/deployments/dep1/metrics").mock(return_value=httpx.Response(200, json={}))
    mock.get("/api/v1/deployments/dep1/logs").mock(return_value=httpx.Response(200, json={}))
    mock.get("/api/v1/deployments/dep1/health").mock(return_value=httpx.Response(200, json={}))

    await client.list_deployments(
        status_filter="active",
        platform="aws",
        project_id="p1",
        search="q",
    )
    await client.list_deployments()  # minimal
    await client.get_deployment("dep1")
    await client.create_deployment({})
    await client.update_deployment("dep1", {})
    await client.delete_deployment("dep1")
    await client.pause_deployment("dep1")
    await client.resume_deployment("dep1")
    await client.scale_deployment("dep1", 5)
    await client.rollback_deployment("dep1", "ck")
    await client.get_deployment_metrics("dep1")
    await client.get_deployment_logs(
        "dep1",
        level="ERROR",
        search="oom",
        start_time="2025-01-01",
        end_time="2025-01-02",
    )
    await client.get_deployment_health_full("dep1")


async def test_async_get_deployment_404(client, mock):
    mock.get("/api/v1/deployments/missing").mock(return_value=httpx.Response(404))
    with pytest.raises(DeploymentNotFoundError):
        await client.get_deployment("missing")


# ---------------------------------------------------------------- inference


async def test_async_predict(client, mock):
    mock.post("/api/v1/inference/dep1/predict").mock(
        return_value=httpx.Response(200, json={"y": 1})
    )
    assert await client.predict("dep1", {"x": 1}) == {"y": 1}


async def test_async_predict_404(client, mock):
    mock.post("/api/v1/inference/missing/predict").mock(return_value=httpx.Response(404))
    with pytest.raises(DeploymentNotFoundError):
        await client.predict("missing", {})


async def test_async_predict_batch(client, mock):
    mock.post("/api/v1/inference/dep1/predict/batch").mock(
        return_value=httpx.Response(200, json=[{"y": 1}, {"y": 2}])
    )
    assert await client.predict_batch("dep1", [{"x": 1}, {"x": 2}]) == [{"y": 1}, {"y": 2}]


async def test_async_deployment_health(client, mock):
    mock.get("/api/v1/inference/dep1/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )
    assert await client.deployment_health("dep1") == {"status": "healthy"}


# ---------------------------------------------------------------- datasets


async def test_async_list_datasets_with_and_without_search(client, mock):
    route = mock.get("/api/v1/datasets/browse").mock(return_value=httpx.Response(200, json=[]))
    await client.list_datasets(search="cifar")
    assert "search=cifar" in str(route.calls[0].request.url)
    await client.list_datasets()


async def test_async_get_dataset_meta(client, mock):
    mock.get("/api/v1/datasets/ds1/meta").mock(return_value=httpx.Response(200, json={"id": "ds1"}))
    assert await client.get_dataset_meta("ds1") == {"id": "ds1"}


async def test_async_dataset_404(client, mock):
    mock.get("/api/v1/datasets/missing/meta").mock(return_value=httpx.Response(404))
    with pytest.raises(DatasetNotFoundError):
        await client.get_dataset_meta("missing")


async def test_async_list_system_datasets(client, mock):
    mock.get("/api/v1/datasets/system").mock(
        return_value=httpx.Response(200, json=[{"id": "iris"}])
    )
    assert await client.list_system_datasets() == [{"id": "iris"}]


async def test_async_get_system_dataset_meta(client, mock):
    mock.get("/api/v1/datasets/system/iris").mock(
        return_value=httpx.Response(200, json={"id": "iris"})
    )
    assert await client.get_system_dataset_meta("iris") == {"id": "iris"}


async def test_async_download_dataset(client, mock, tmp_path: Path):
    mock.get("/api/v1/datasets/ds1/download").mock(
        return_value=httpx.Response(
            200,
            content=b"data",
            headers={"content-disposition": 'attachment; filename="ds.bin"'},
        )
    )
    out = await client.download_dataset("ds1", tmp_path)
    assert out.read_bytes() == b"data"


async def test_async_download_system_dataset(client, mock, tmp_path: Path):
    mock.get("/api/v1/datasets/system/iris/download").mock(
        return_value=httpx.Response(
            200,
            content=b"iris",
            headers={"content-disposition": 'attachment; filename="iris.csv"'},
        )
    )
    out = await client.download_system_dataset("iris", tmp_path)
    assert out.read_bytes() == b"iris"


async def test_async_upload_dataset(client, mock, tmp_path: Path):
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


async def test_async_upload_dataset_from_url(client, mock):
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


async def test_async_get_dataset_task_status(client, mock):
    mock.get("/api/v1/datasets/tasks/t1").mock(
        return_value=httpx.Response(200, json={"status": "done"})
    )
    assert await client.get_dataset_task_status("t1") == {"status": "done"}


# ---------------------------------------------------------------- checkpoints


async def test_async_list_checkpoints(client, mock):
    mock.get("/api/v1/training/jobs/job1/checkpoints").mock(
        return_value=httpx.Response(200, json=[{"id": "c1"}])
    )
    assert await client.list_checkpoints("job1") == [{"id": "c1"}]


async def test_async_list_checkpoints_404(client, mock):
    mock.get("/api/v1/training/jobs/job1/checkpoints").mock(return_value=httpx.Response(404))
    with pytest.raises(TrainingJobNotFoundError):
        await client.list_checkpoints("job1")


async def test_async_download_checkpoint(client, mock, tmp_path: Path):
    url = "/api/v1/training/jobs/job1/checkpoints/ck1/download"
    mock.get(url).mock(
        return_value=httpx.Response(200, content=b"weights", headers={"x-checksum-sha256": "abc"})
    )
    dest = tmp_path / "ck.bin"
    written, checksum = await client.download_checkpoint("job1", "ck1", dest)
    assert written.read_bytes() == b"weights"
    assert checksum == "abc"


async def test_async_download_checkpoint_401(client, mock, tmp_path: Path):
    url = "/api/v1/training/jobs/job1/checkpoints/ck1/download"
    mock.get(url).mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await client.download_checkpoint("job1", "ck1", tmp_path / "x")


async def test_async_download_checkpoint_404(client, mock, tmp_path: Path):
    url = "/api/v1/training/jobs/job1/checkpoints/ck1/download"
    mock.get(url).mock(return_value=httpx.Response(404))
    with pytest.raises(CheckpointNotFoundError):
        await client.download_checkpoint("job1", "ck1", tmp_path / "x")


async def test_async_download_checkpoint_500(client, mock, tmp_path: Path):
    url = "/api/v1/training/jobs/job1/checkpoints/ck1/download"
    mock.get(url).mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(APIError):
        await client.download_checkpoint("job1", "ck1", tmp_path / "x")


# ---------------------------------------------------------------- codegen


async def test_async_generate_code_default(client, mock):
    route = mock.post("/api/v1/projects/p1/generate-code").mock(
        return_value=httpx.Response(200, json={"task_id": "t1"})
    )
    await client.generate_code(
        "p1", framework="tf", version_id="v2", options={"s": 1}, async_mode=True
    )
    body = route.calls[0].request.read()
    assert b"tf" in body
    assert b"v2" in body
    assert "async_mode=true" in str(route.calls[0].request.url)


async def test_async_generate_code_explicit_payload(client, mock):
    mock.post("/api/v1/projects/p1/generate-code").mock(return_value=httpx.Response(200, json={}))
    await client.generate_code("p1", payload={"custom": True})


async def test_async_preview_code(client, mock):
    mock.get("/api/v1/projects/p1/code-preview").mock(return_value=httpx.Response(200, json={}))
    await client.preview_code("p1", "pytorch", version_id="v1")
    await client.preview_code("p1", "pytorch")


async def test_async_validate_code(client, mock):
    mock.post("/api/v1/projects/p1/validate").mock(return_value=httpx.Response(200, json={}))
    await client.validate_code("p1", version_id="v1")
    await client.validate_code("p1")
    await client.validate_architecture("p1")


async def test_async_download_code_returns_bytes(client, mock):
    mock.get("/api/v1/projects/p1/download-code").mock(
        return_value=httpx.Response(200, content=b"<code>")
    )
    out = await client.download_code("p1", framework="pytorch", version_id="v1")
    assert out == b"<code>"


async def test_async_download_code_to_file(client, mock, tmp_path: Path):
    mock.get("/api/v1/projects/p1/download-code").mock(
        return_value=httpx.Response(200, content=b"<code>")
    )
    dest = tmp_path / "out.zip"
    out = await client.download_code("p1", dest_path=dest)
    assert out == dest
    assert dest.read_bytes() == b"<code>"


async def test_async_download_code_zip_alias(client, mock):
    mock.get("/api/v1/projects/p1/download-code").mock(
        return_value=httpx.Response(200, content=b"x")
    )
    assert await client.download_code_zip("p1", "pytorch") == b"x"


async def test_async_codegen_text_response(client, mock):
    mock.get("/api/v1/projects/p1/code-preview").mock(
        return_value=httpx.Response(200, text="plain", headers={"Content-Type": "text/plain"})
    )
    assert await client.preview_code("p1", "pytorch") == "plain"


async def test_async_codegen_empty_response(client, mock):
    mock.get("/api/v1/projects/p1/code-preview").mock(return_value=httpx.Response(204))
    assert await client.preview_code("p1", "pytorch") is None
