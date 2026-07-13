"""Wire-level coverage for the sync projects client mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import requests

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import (
    APIError,
    ProjectNotFoundError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

API = "https://api.test"


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


def test_duplicate_project_with_and_without_title(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/projects/p1/duplicate", json={"id": "p2"})
    client.duplicate_project("p1", title="copy")
    assert rmock.last_request.json() == {"title": "copy"}
    client.duplicate_project("p1")
    # No body when title is None
    assert rmock.last_request.text in (None, "")


def test_save_architecture(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/projects/p1/save", json={"version_id": "v1"})
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
    client._sleep = lambda _s: None  # 500 is transient on a GET → retried; don't sleep
    with pytest.raises(APIError):
        client.list_projects()


def test_projects_retries_transient(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(
        f"{API}/api/v1/projects",
        [{"status_code": 503}, {"status_code": 200, "json": {"items": []}}],
    )
    client._sleep = lambda _s: None
    client._rng = lambda: 1.0
    assert client.list_projects() == {"items": []}
    assert rmock.call_count == 2


def test_projects_404_not_retried(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/projects/p1", status_code=404)
    client._sleep = lambda _s: None
    with pytest.raises(ProjectNotFoundError):
        client.get_project("p1")
    assert rmock.call_count == 1


def test_projects_text_response(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/projects", text="plain", headers={"Content-Type": "text/plain"})
    assert client.list_projects() == "plain"


def test_projects_connectionerror_wrapped(client: DagnamClient, rmock: RequestsMocker) -> None:
    client._sleep = lambda _s: None
    rmock.get(f"{API}/api/v1/projects", exc=requests.ConnectionError("nope"))
    with pytest.raises(APIError, match="Request failed"):
        client.list_projects()


def test_projects_timeout_wrapped(client: DagnamClient, rmock: RequestsMocker) -> None:
    client._sleep = lambda _s: None
    rmock.get(f"{API}/api/v1/projects", exc=requests.Timeout("slow"))
    with pytest.raises(APIError, match="Request failed"):
        client.list_projects()


# ---------------------------------------------------------------- project versions


def test_project_versions_surface(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/projects/p1/versions", json={"items": [], "total": 0})
    rmock.get(f"{API}/api/v1/projects/p1/versions/v1", json={"id": "v1", "version_number": "1.0.0"})
    rmock.get(f"{API}/api/v1/projects/p1/versions/compare", json={"version_a": {}, "version_b": {}})
    rmock.post(
        f"{API}/api/v1/projects/p1/restore/v1",
        json={"id": "v2", "is_current": True},
        status_code=201,
    )
    rmock.delete(f"{API}/api/v1/projects/p1/versions/v1", status_code=204)
    rmock.get(f"{API}/api/v1/projects/p1/latest", json={"id": "v2", "is_current": True})

    assert "items" in client.list_project_versions("p1")
    assert client.get_project_version("p1", "v1")["version_number"] == "1.0.0"
    assert "version_a" in client.compare_project_versions("p1", "v1", "v2")
    assert client.restore_project_version("p1", "v1")["is_current"] is True
    assert client.delete_project_version("p1", "v1") is None
    assert client.get_latest_project_version("p1")["is_current"] is True


def test_list_project_versions_passes_pagination(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.get(f"{API}/api/v1/projects/p1/versions", json={"items": []})
    client.list_project_versions("p1", page=2, limit=5)
    qs = rmock.last_request.qs
    assert qs["page"] == ["2"]
    assert qs["limit"] == ["5"]


def test_compare_project_versions_passes_query(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/projects/p1/versions/compare", json={"version_a": {}, "version_b": {}})
    client.compare_project_versions("p1", "va", "vb")
    qs = rmock.last_request.qs
    assert qs["version_a"] == ["va"]
    assert qs["version_b"] == ["vb"]


def test_get_project_version_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/projects/p1/versions/missing", status_code=404)
    with pytest.raises(ProjectNotFoundError):
        client.get_project_version("p1", "missing")
