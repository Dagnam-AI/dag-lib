"""Unit tests for dagnam.hub module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dagnam import hub
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import HubModelNotFoundError


def _client(**overrides) -> MagicMock:
    return MagicMock(spec=DagnamClient, **overrides)


class TestDiscovery:
    def test_search_delegates_all_filters(self):
        c = _client(hub_search=MagicMock(return_value={"items": [], "total": 0}))
        out = hub.search(
            search="llama",
            task_type="text-generation",
            framework="pytorch",
            license="mit",
            tags=["nlp"],
            is_official=True,
            is_verified=False,
            sort_by="recent",
            page=2,
            limit=10,
            client=c,
        )
        c.hub_search.assert_called_once_with(
            search="llama",
            task_type="text-generation",
            framework="pytorch",
            license="mit",
            tags=["nlp"],
            is_official=True,
            is_verified=False,
            sort_by="recent",
            page=2,
            limit=10,
        )
        assert out["total"] == 0

    def test_categories_delegates(self):
        c = _client(hub_categories=MagicMock(return_value=["nlp", "cv"]))
        assert hub.categories(client=c) == ["nlp", "cv"]
        c.hub_categories.assert_called_once()

    def test_featured_delegates(self):
        c = _client(hub_featured=MagicMock(return_value=[{"id": "m1"}]))
        assert hub.featured(client=c) == [{"id": "m1"}]

    def test_trending_passes_days(self):
        c = _client(hub_trending=MagicMock(return_value=[]))
        hub.trending(days=30, client=c)
        c.hub_trending.assert_called_once_with(days=30)


class TestModelCRUD:
    def test_get_delegates(self):
        c = _client(hub_get=MagicMock(return_value={"id": "m1", "name": "test"}))
        out = hub.get("m1", client=c)
        c.hub_get.assert_called_once_with("m1")
        assert out["name"] == "test"

    def test_create_builds_payload(self):
        c = _client(hub_create=MagicMock(return_value={"id": "m1"}))
        hub.create(
            name="my-model",
            description="desc",
            task_type="text-generation",
            framework="pytorch",
            tags=["nlp"],
            client=c,
        )
        payload = c.hub_create.call_args.args[0]
        assert payload["name"] == "my-model"
        assert payload["tags"] == ["nlp"]
        assert payload["license"] == "mit"

    def test_create_omits_none_optionals(self):
        c = _client(hub_create=MagicMock(return_value={"id": "m1"}))
        hub.create(
            name="m",
            description="d",
            task_type="t",
            framework="f",
            client=c,
        )
        payload = c.hub_create.call_args.args[0]
        assert "tags" not in payload
        assert "metadata" not in payload

    def test_update_delegates(self):
        c = _client(hub_update=MagicMock(return_value={"id": "m1"}))
        hub.update("m1", description="new", client=c)
        c.hub_update.assert_called_once_with("m1", {"description": "new"})

    def test_delete_delegates(self):
        c = _client(hub_delete=MagicMock(return_value=None))
        hub.delete("m1", client=c)
        c.hub_delete.assert_called_once_with("m1")


class TestFilesAndVersions:
    def test_list_files_delegates(self):
        c = _client(hub_list_files=MagicMock(return_value={"files": []}))
        hub.list_files("m1", client=c)
        c.hub_list_files.assert_called_once_with("m1")

    def test_download_with_file_id(self):
        c = _client(hub_download=MagicMock(return_value={"url": "https://x"}))
        hub.download("m1", file_id="f1", client=c)
        c.hub_download.assert_called_once_with("m1", file_id="f1")

    def test_list_versions_delegates(self):
        c = _client(hub_list_versions=MagicMock(return_value=[]))
        hub.list_versions("m1", client=c)
        c.hub_list_versions.assert_called_once_with("m1")

    def test_create_version_delegates(self):
        c = _client(hub_create_version=MagicMock(return_value={"version": "1.0"}))
        hub.create_version("m1", "1.0", changelog="init", client=c)
        c.hub_create_version.assert_called_once_with("m1", version="1.0", changelog="init")


class TestSocial:
    def test_star_delegates(self):
        c = _client(hub_star=MagicMock(return_value={"starred": True}))
        out = hub.star("m1", client=c)
        c.hub_star.assert_called_once_with("m1")
        assert out["starred"] is True

    def test_unstar_delegates(self):
        c = _client(hub_unstar=MagicMock(return_value={}))
        hub.unstar("m1", client=c)
        c.hub_unstar.assert_called_once_with("m1")

    def test_starred_passes_params(self):
        c = _client(hub_starred=MagicMock(return_value={"items": []}))
        hub.starred(sort_by="name", page=2, limit=5, client=c)
        c.hub_starred.assert_called_once_with(sort_by="name", page=2, limit=5)

    def test_fork_delegates(self):
        c = _client(hub_fork=MagicMock(return_value={"id": "m2"}))
        hub.fork("m1", client=c)
        c.hub_fork.assert_called_once_with("m1")

    def test_add_review_delegates(self):
        c = _client(hub_add_review=MagicMock(return_value={"id": "r1"}))
        hub.add_review("m1", 5, review_text="great", client=c)
        c.hub_add_review.assert_called_once_with("m1", rating=5, review_text="great")

    def test_list_reviews_delegates(self):
        c = _client(hub_list_reviews=MagicMock(return_value={"items": []}))
        hub.list_reviews("m1", page=2, limit=10, client=c)
        c.hub_list_reviews.assert_called_once_with("m1", page=2, limit=10)


class TestStudioIntegration:
    def test_use_in_studio_delegates(self):
        c = _client(hub_use_in_studio=MagicMock(return_value={"project_id": "p1"}))
        out = hub.use_in_studio("m1", client=c)
        c.hub_use_in_studio.assert_called_once_with("m1")
        assert out["project_id"] == "p1"


class TestErrorPropagation:
    def test_get_propagates_hub_model_not_found(self):
        c = _client(hub_get=MagicMock(side_effect=HubModelNotFoundError("m1")))
        with pytest.raises(HubModelNotFoundError):
            hub.get("m1", client=c)
