"""Wire-level coverage for the sync hub client mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import requests

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import (
    APIError,
    HubError,
    HubModelNotFoundError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, RequestsMocker

API = "https://api.test"


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


def test_hub_categories_featured_trending_starred(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
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
