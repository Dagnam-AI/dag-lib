"""Wire-level coverage for sync client mixins: hub, projects, deployments, datasets, base."""

from __future__ import annotations
from pathlib import Path
from tests.typing_helpers import PytestMonkeyPatch, RequestsMocker


from typing import ClassVar

import pytest
import requests
import requests_mock as rm_module

from dagnam._core.client import DagnamClient
from dagnam._core.client.base import (
    is_success_response,
    parse_content_disposition_filename,
    safe_error_body_from_response,
    _sanitize_filename,
)
from dagnam._core.exceptions import (
    APIError,
    DatasetNotFoundError,
    DeploymentNotFoundError,
    HubError,
    HubModelNotFoundError,
    ProjectNotFoundError,
)

API = "https://api.test"


@pytest.fixture
def client() -> DagnamClient:
    return DagnamClient(API, "k")


@pytest.fixture
def rmock():
    with rm_module.Mocker() as m:
        yield m


# ---------------------------------------------------------------- base helpers


def testparse_content_disposition_filename_quoted() -> None:
    assert parse_content_disposition_filename('attachment; filename="cool.csv"') == "cool.csv"


def testparse_content_disposition_filename_unquoted() -> None:
    assert parse_content_disposition_filename("attachment; filename=cool.csv") == "cool.csv"


def testparse_content_disposition_filename_default_when_none() -> None:
    assert parse_content_disposition_filename(None) == "data"


def testparse_content_disposition_filename_default_when_no_filename_param() -> None:
    assert parse_content_disposition_filename("inline") == "data"


def test_sanitize_filename_rejects_path_separator() -> None:
    with pytest.raises(ValueError, match="Unsafe filename"):
        _sanitize_filename("../etc/passwd")


def test_sanitize_filename_rejects_windows_reserved() -> None:
    with pytest.raises(ValueError, match="Unsafe filename"):
        _sanitize_filename("CON.txt")


def test_sanitize_filename_rejects_drive_letter() -> None:
    with pytest.raises(ValueError, match="Unsafe filename"):
        _sanitize_filename("C:nasty")


def test_sanitize_filename_rejects_empty_and_dots() -> None:
    for bad in ("", ".", ".."):
        with pytest.raises(ValueError):
            _sanitize_filename(bad)


def testis_success_response_from_status_code() -> None:
    class _R:
        status_code = 204
        ok = False

    assert is_success_response(_R())


def testis_success_response_falls_back_to_ok_attr() -> None:
    class _R:
        status_code = None
        ok = True

    assert is_success_response(_R())


def test_safe_error_body_from_response_delegates_to_common(client: DagnamClient) -> None:
    class _R:
        headers: ClassVar[dict[str, str]] = {"Content-Type": "text/plain"}
        content = b"err"
        text = "err"

    assert safe_error_body_from_response(_R()) == "err"


# ---------------------------------------------------------------- hub client


def test_list_hub_models(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/hub/models", json={"items": []})
    assert client.list_hub_models(category="vision") == {"items": []}
    assert rmock.last_request.qs == {"category": ["vision"]}


def test_get_hub_model(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/hub/models/m1", json={"id": "m1"})
    assert client.get_hub_model("m1") == {"id": "m1"}


def test_get_hub_model_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/hub/models/missing", status_code=404)
    with pytest.raises(HubModelNotFoundError):
        client.get_hub_model("missing")


def test_create_hub_model(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/hub/models", json={"id": "m1"})
    assert client.create_hub_model({"name": "x"}) == {"id": "m1"}


def test_update_hub_model(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.put(f"{API}/api/v1/hub/models/m1", json={"id": "m1"})
    assert client.update_hub_model("m1", {"name": "y"}) == {"id": "m1"}


def test_delete_hub_model_empty_body(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(f"{API}/api/v1/hub/models/m1", status_code=204, text="")
    assert client.delete_hub_model("m1") is None


def test_list_hub_model_files(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/hub/models/m1/files", json={"files": []})
    assert client.list_hub_model_files("m1") == {"files": []}


def test_download_hub_model_with_file_id(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/hub/models/m1/download", json={"url": "u"})
    assert client.download_hub_model("m1", file_id="f1") == {"url": "u"}
    assert rmock.last_request.qs == {"file_id": ["f1"]}


def test_download_hub_model_without_file_id(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/hub/models/m1/download", json={"url": "u"})
    client.download_hub_model("m1")
    assert rmock.last_request.qs == {}


def test_list_hub_model_versions(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/hub/models/m1/versions", json=[{"v": 1}])
    assert client.list_hub_model_versions("m1") == [{"v": 1}]


def test_create_hub_model_version(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/hub/models/m1/versions", json={"v": 1})
    assert client.create_hub_model_version("m1", {"v": 1}) == {"v": 1}


def test_star_and_unstar(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/hub/models/m1/star", json={"starred": True})
    rmock.delete(f"{API}/api/v1/hub/models/m1/star", json={"starred": False})
    assert client.star_hub_model("m1") == {"starred": True}
    assert client.unstar_hub_model("m1") == {"starred": False}


def test_fork_hub_model(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/hub/models/m1/fork", json={"new_id": "m2"})
    assert client.fork_hub_model("m1") == {"new_id": "m2"}


def test_list_hub_model_reviews(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/hub/models/m1/reviews", json={"items": []})
    client.list_hub_model_reviews("m1", page=2, limit=50)
    assert rmock.last_request.qs == {"page": ["2"], "limit": ["50"]}


def test_add_hub_model_review(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/hub/models/m1/reviews", json={"id": "r1"})
    assert client.add_hub_model_review("m1", {"rating": 5}) == {"id": "r1"}


def test_use_hub_model_in_studio(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/hub/models/m1/use-in-studio", json={"ok": True})
    assert client.use_hub_model_in_studio("m1") == {"ok": True}


def test_hub_categories_featured_trending_starred(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/hub/categories", json=["a"])
    rmock.get(f"{API}/api/v1/hub/featured", json=["f"])
    rmock.get(f"{API}/api/v1/hub/trending", json=["t"])
    rmock.get(f"{API}/api/v1/hub/starred", json={"items": []})
    assert client.list_hub_categories() == ["a"]
    assert client.get_hub_featured() == ["f"]
    assert client.get_hub_trending(days=14) == ["t"]
    client.list_hub_starred(sort_by="name", page=3, limit=10)
    assert rmock.last_request.qs == {"sort_by": ["name"], "page": ["3"], "limit": ["10"]}


def test_hub_text_body_returned_when_not_json(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(
        f"{API}/api/v1/hub/categories", text="plain text", headers={"Content-Type": "text/plain"}
    )
    assert client.list_hub_categories() == "plain text"


def test_hub_500_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/hub/categories", status_code=500, text="boom")
    with pytest.raises(APIError):
        client.list_hub_categories()


def test_hub_400_raises_huberror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/hub/models", status_code=400, text="bad")
    with pytest.raises(HubError):
        client.create_hub_model({})


def test_hub_connectionerror_wrapped(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "request", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.list_hub_categories()


def test_hub_timeout_wrapped(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "request", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.list_hub_categories()


# ---------------------------------------------------------------- projects client


def test_list_projects(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/projects", json={"items": []})
    client.list_projects(search="x")
    assert rmock.last_request.qs == {"search": ["x"]}


def test_get_project(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/projects/p1", json={"id": "p1"})
    assert client.get_project("p1") == {"id": "p1"}


def test_get_project_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/projects/missing", status_code=404)
    with pytest.raises(ProjectNotFoundError):
        client.get_project("missing")


def test_create_update_delete_project(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/projects", json={"id": "p1"})
    rmock.put(f"{API}/api/v1/projects/p1", json={"id": "p1", "title": "x"})
    rmock.delete(f"{API}/api/v1/projects/p1", status_code=204, text="")
    assert client.create_project({}) == {"id": "p1"}
    assert client.update_project("p1", {"title": "x"})["title"] == "x"
    assert client.delete_project("p1") is None


def test_duplicate_project_with_and_without_title(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/projects/p1/duplicate", json={"id": "p2"})
    client.duplicate_project("p1", title="copy")
    assert rmock.last_request.json() == {"title": "copy"}
    client.duplicate_project("p1")
    # No body when title is None
    assert rmock.last_request.text in (None, "")


def test_save_architecture(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/projects/p1/architecture", json={"version_id": "v1"})
    assert client.save_architecture("p1", {"nodes": []}) == {"version_id": "v1"}


def test_import_dag(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/projects/import", json={"id": "p1"})
    assert client.import_dag({"dag": "..."}) == {"id": "p1"}


def test_import_dag_existing(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/projects/p1/import", json={"id": "p1"})
    assert client.import_dag_existing("p1", {"dag": "..."}) == {"id": "p1"}


def test_bulk_delete_projects(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/projects/bulk-delete", json={"deleted": 2})
    assert client.bulk_delete_projects(["p1", "p2"]) == {"deleted": 2}
    assert rmock.last_request.json() == {"project_ids": ["p1", "p2"]}


def test_link_unlink_dataset(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/projects/p1/datasets", json={"linked": True})
    rmock.delete(f"{API}/api/v1/projects/p1/datasets/d1", status_code=204, text="")
    rmock.get(f"{API}/api/v1/projects/p1/datasets", json={"items": []})
    assert client.link_dataset("p1", "d1", "train") == {"linked": True}
    assert rmock.last_request.json() == {"dataset_id": "d1", "role": "train"}
    assert client.get_project_datasets("p1") == {"items": []}
    assert client.unlink_dataset("p1", "d1") is None


def test_projects_500_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/projects", status_code=500, text="boom")
    with pytest.raises(APIError):
        client.list_projects()


def test_projects_text_response(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/projects", text="plain", headers={"Content-Type": "text/plain"})
    assert client.list_projects() == "plain"


def test_projects_connectionerror_wrapped(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "request", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.list_projects()


def test_projects_timeout_wrapped(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "request", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.list_projects()


# ---------------------------------------------------------------- deployments client


def test_list_deployments_all_params(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/deployments", json={"items": []})
    client.list_deployments(
        page=2,
        limit=10,
        status_filter="active",
        platform="aws",
        project_id="p1",
        search="foo",
    )
    qs = rmock.last_request.qs
    assert qs["page"] == ["2"]
    assert qs["status"] == ["active"]
    assert qs["platform"] == ["aws"]
    assert qs["project_id"] == ["p1"]
    assert qs["search"] == ["foo"]


def test_list_deployments_minimal(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/deployments", json={"items": []})
    client.list_deployments()
    # Only page+limit
    assert set(rmock.last_request.qs.keys()) == {"page", "limit"}


def test_get_deployment(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/deployments/dep1", json={"id": "dep1"})
    assert client.get_deployment("dep1") == {"id": "dep1"}


def test_get_deployment_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/deployments/missing", status_code=404)
    with pytest.raises(DeploymentNotFoundError):
        client.get_deployment("missing")


def test_create_deployment(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/deployments", json={"id": "dep1"})
    assert client.create_deployment({"name": "x"}) == {"id": "dep1"}


def test_update_deployment(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.put(f"{API}/api/v1/deployments/dep1", json={"id": "dep1"})
    assert client.update_deployment("dep1", {}) == {"id": "dep1"}


def test_delete_deployment_empty_body(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(f"{API}/api/v1/deployments/dep1", status_code=204, text="")
    assert client.delete_deployment("dep1") is None


def test_pause_resume(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/deployments/dep1/pause", json={"paused": True})
    rmock.post(f"{API}/api/v1/deployments/dep1/resume", json={"paused": False})
    assert client.pause_deployment("dep1") == {"paused": True}
    assert client.resume_deployment("dep1") == {"paused": False}


def test_scale_deployment(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.put(f"{API}/api/v1/deployments/dep1/scale", json={"instances": 5})
    client.scale_deployment("dep1", 5)
    assert rmock.last_request.qs == {"num_instances": ["5"]}


def test_rollback_deployment(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/deployments/dep1/rollback", json={"ok": True})
    client.rollback_deployment("dep1", "ckpt/path")
    assert rmock.last_request.qs == {"checkpoint_path": ["ckpt/path"]}


def test_get_deployment_metrics(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/deployments/dep1/metrics", json={"qps": 1})
    client.get_deployment_metrics("dep1", time_range="1h")
    assert rmock.last_request.qs == {"time_range": ["1h"]}


def test_get_deployment_logs_all_params(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/deployments/dep1/logs", json={"items": []})
    client.get_deployment_logs(
        "dep1",
        level="ERROR",
        search="oom",
        start_time="2025-01-01",
        end_time="2025-01-02",
        page=2,
        limit=50,
    )
    qs = rmock.last_request.qs
    for key in ("level", "search", "start_time", "end_time", "page", "limit"):
        assert key in qs


def test_get_deployment_health_full(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/deployments/dep1/health", json={"status": "healthy"})
    assert client.get_deployment_health_full("dep1") == {"status": "healthy"}


def test_open_deployment_stream_sets_headers(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/deployments/dep1/stream", text="data: x\n\n")
    resp = client.open_deployment_stream("dep1", last_event_id="cursor")
    assert resp.status_code == 200
    req = rmock.last_request
    assert req.headers["Accept"] == "text/event-stream"
    assert req.headers["Last-Event-ID"] == "cursor"
    assert req.qs == {"api_key": ["k"]}


def test_open_deployment_stream_without_cursor(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/deployments/dep1/stream", text="")
    resp = client.open_deployment_stream("dep1")
    assert "Last-Event-ID" not in resp.request.headers


def test_open_deployment_stream_connectionerror(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.open_deployment_stream("dep1")


def test_open_deployment_stream_timeout(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.open_deployment_stream("dep1")


def test_deployments_connectionerror_wrapped(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "request", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.list_deployments()


def test_deployments_timeout_wrapped(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "request", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.list_deployments()


def test_deployments_text_response(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/deployments", text="plain", headers={"Content-Type": "text/plain"})
    assert client.list_deployments() == "plain"


# ---------------------------------------------------------------- datasets client


def test_list_datasets_with_search(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/datasets/browse", json=[{"id": "ds1"}])
    client.list_datasets(type="vision", search="cifar")
    qs = rmock.last_request.qs
    assert qs == {"type": ["vision"], "search": ["cifar"]}


def test_list_datasets_no_search(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/datasets/browse", json=[])
    client.list_datasets()
    qs = rmock.last_request.qs
    assert qs == {"type": ["all"]}


def test_list_datasets_connectionerror(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.list_datasets()


def test_list_datasets_timeout(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.list_datasets()


def test_get_dataset_meta_with_version(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/datasets/ds1/meta", json={"id": "ds1"})
    client.get_dataset_meta("ds1", version="v2")
    assert rmock.last_request.qs == {"version": ["v2"]}


def test_get_dataset_meta_connectionerror(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.get_dataset_meta("ds1")


def test_get_dataset_meta_timeout(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.get_dataset_meta("ds1")


def test_list_system_datasets(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/datasets/system", json=[{"id": "iris"}])
    assert client.list_system_datasets() == [{"id": "iris"}]


def test_list_system_datasets_connectionerror(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.list_system_datasets()


def test_list_system_datasets_timeout(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.list_system_datasets()


def test_get_system_dataset_meta(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/datasets/system/iris", json={"id": "iris"})
    client.get_system_dataset_meta("iris", version="2.0")
    assert rmock.last_request.qs == {"version": ["2.0"]}


def test_get_system_dataset_meta_no_version(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/datasets/system/iris", json={"id": "iris"})
    client.get_system_dataset_meta("iris")
    assert rmock.last_request.qs == {}


def test_get_system_dataset_meta_connectionerror(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.get_system_dataset_meta("ds1")


def test_get_system_dataset_meta_timeout(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.get_system_dataset_meta("ds1")


def test_download_system_dataset_writes_file(client: DagnamClient, rmock: RequestsMocker, tmp_path: Path) -> None:
    rmock.get(
        f"{API}/api/v1/datasets/system/iris/download",
        content=b"hello",
        headers={"Content-Disposition": 'attachment; filename="iris.csv"'},
    )
    out = client.download_system_dataset("iris", tmp_path)
    assert out.name == "iris.csv"
    assert out.read_bytes() == b"hello"


def test_upload_dataset_streams_file(client: DagnamClient, rmock: RequestsMocker, tmp_path: Path) -> None:
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2")
    rmock.post(f"{API}/api/v1/datasets/upload", json={"id": "ds1"})
    result = client.upload_dataset(
        f,
        name="x",
        dataset_type="tabular",
        format="csv",
        description="desc",
        license="MIT",
    )
    assert result == {"id": "ds1"}


def test_upload_dataset_connectionerror(client: DagnamClient, monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    f = tmp_path / "data.csv"
    f.write_text("x")

    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.upload_dataset(f, name="x", dataset_type="t", format="csv")


def test_upload_dataset_timeout(client: DagnamClient, monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    f = tmp_path / "data.csv"
    f.write_text("x")

    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.upload_dataset(f, name="x", dataset_type="t", format="csv")


def test_upload_dataset_from_url(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/datasets/upload-url", json={"task_id": "t1"})
    result = client.upload_dataset_from_url(
        "https://example/data.csv",
        name="x",
        dataset_type="tabular",
        format="csv",
        description="d",
    )
    assert result == {"task_id": "t1"}
    body = rmock.last_request.json()
    assert body["url"] == "https://example/data.csv"
    assert body["description"] == "d"


def test_upload_dataset_from_url_connectionerror(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.upload_dataset_from_url("u", name="x", dataset_type="t", format="csv")


def test_upload_dataset_from_url_timeout(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.upload_dataset_from_url("u", name="x", dataset_type="t", format="csv")


def test_get_dataset_task_status(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/datasets/tasks/t1", json={"status": "done"})
    assert client.get_dataset_task_status("t1") == {"status": "done"}


def test_get_dataset_task_status_connectionerror(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.get_dataset_task_status("t1")


def test_get_dataset_task_status_timeout(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.get_dataset_task_status("t1")


def test_download_dataset_full(client: DagnamClient, rmock: RequestsMocker, tmp_path: Path) -> None:
    rmock.get(
        f"{API}/api/v1/datasets/ds1/download",
        content=b"abc",
        headers={"Content-Disposition": 'attachment; filename="ds.bin"'},
    )
    out = client.download_dataset("ds1", tmp_path)
    assert out.read_bytes() == b"abc"


def test_download_dataset_resume_206(client: DagnamClient, rmock: RequestsMocker, tmp_path: Path) -> None:
    # Pre-existing partial file
    part = tmp_path / "ds.bin.part"
    part.write_bytes(b"abc")
    rmock.get(
        f"{API}/api/v1/datasets/ds1/download",
        content=b"def",
        status_code=206,
    )
    out = client.download_dataset("ds1", tmp_path, filename="ds.bin", resume=True)
    assert out.read_bytes() == b"abcdef"


def test_download_dataset_resume_falls_back_to_full(client: DagnamClient, rmock: RequestsMocker, tmp_path: Path) -> None:
    part = tmp_path / "ds.bin.part"
    part.write_bytes(b"old")
    rmock.get(
        f"{API}/api/v1/datasets/ds1/download",
        content=b"new",
        status_code=200,
    )
    out = client.download_dataset("ds1", tmp_path, filename="ds.bin", resume=True)
    # Server ignored Range; we restart with fresh body.
    assert out.read_bytes() == b"new"


def test_download_dataset_resume_disabled_cleans_partial(client: DagnamClient, rmock: RequestsMocker, tmp_path: Path) -> None:
    part = tmp_path / "ds.bin.part"
    part.write_bytes(b"stale")
    rmock.get(
        f"{API}/api/v1/datasets/ds1/download",
        content=b"fresh",
    )
    out = client.download_dataset("ds1", tmp_path, filename="ds.bin", resume=False)
    assert out.read_bytes() == b"fresh"


def test_download_dataset_with_presigned_url(client: DagnamClient, tmp_path: Path) -> None:
    presigned = "https://presigned.test/blob"
    with rm_module.Mocker() as m:
        m.get(
            presigned,
            content=b"blob",
            headers={"Content-Disposition": 'attachment; filename="blob.bin"'},
        )
        out = client.download_dataset("ds1", tmp_path, download_url=presigned)
        assert out.name == "blob.bin"
        # Presigned path must not send auth headers
        assert "Authorization" not in m.last_request.headers


def test_download_dataset_connectionerror(client: DagnamClient, monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.download_dataset("ds1", tmp_path)


def test_download_dataset_timeout(client: DagnamClient, monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.download_dataset("ds1", tmp_path)


def test_download_dataset_404(client: DagnamClient, rmock: RequestsMocker, tmp_path: Path) -> None:
    rmock.get(f"{API}/api/v1/datasets/missing/download", status_code=404)
    with pytest.raises(DatasetNotFoundError):
        client.download_dataset("missing", tmp_path)


def test_download_system_dataset_connectionerror(client: DagnamClient, monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.download_system_dataset("iris", tmp_path)


def test_download_system_dataset_timeout(client: DagnamClient, monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.download_system_dataset("iris", tmp_path)
