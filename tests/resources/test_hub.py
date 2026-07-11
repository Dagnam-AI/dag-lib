"""Unit tests for dagnam.hub module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dagnam import hub
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import HubModelNotFoundError


def _client(**overrides: object) -> MagicMock:
    client = MagicMock(spec=DagnamClient)
    client.configure_mock(**overrides)
    return client


class TestDiscovery:
    def test_search_delegates_all_filters(self) -> None:
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

    def test_categories_delegates(self) -> None:
        c = _client(hub_categories=MagicMock(return_value=["nlp", "cv"]))
        assert hub.categories(client=c) == ["nlp", "cv"]
        c.hub_categories.assert_called_once()

    def test_featured_delegates(self) -> None:
        c = _client(hub_featured=MagicMock(return_value=[{"id": "m1"}]))
        assert hub.featured(client=c) == [{"id": "m1"}]

    def test_trending_passes_days(self) -> None:
        c = _client(hub_trending=MagicMock(return_value=[]))
        hub.trending(days=30, client=c)
        c.hub_trending.assert_called_once_with(days=30)


class TestModelCRUD:
    def test_get_delegates(self) -> None:
        c = _client(hub_get=MagicMock(return_value={"id": "m1", "name": "test"}))
        out = hub.get("m1", client=c)
        c.hub_get.assert_called_once_with("m1")
        assert out["name"] == "test"

    def test_create_builds_payload(self) -> None:
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

    def test_create_omits_none_optionals(self) -> None:
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

    def test_update_delegates(self) -> None:
        c = _client(hub_update=MagicMock(return_value={"id": "m1"}))
        hub.update("m1", description="new", client=c)
        c.hub_update.assert_called_once_with("m1", {"description": "new"})

    def test_delete_delegates(self) -> None:
        c = _client(hub_delete=MagicMock(return_value=None))
        hub.delete("m1", client=c)
        c.hub_delete.assert_called_once_with("m1")


class TestFilesAndVersions:
    def test_list_files_delegates(self) -> None:
        c = _client(hub_list_files=MagicMock(return_value={"files": []}))
        hub.list_files("m1", client=c)
        c.hub_list_files.assert_called_once_with("m1")

    def test_download_with_file_id(self) -> None:
        c = _client(hub_download=MagicMock(return_value={"url": "https://x"}))
        hub.download("m1", file_id="f1", client=c)
        c.hub_download.assert_called_once_with("m1", file_id="f1")

    def test_list_versions_delegates(self) -> None:
        c = _client(hub_list_versions=MagicMock(return_value=[]))
        hub.list_versions("m1", client=c)
        c.hub_list_versions.assert_called_once_with("m1")

    def test_create_version_delegates(self) -> None:
        c = _client(hub_create_version=MagicMock(return_value={"version": "1.0"}))
        hub.create_version("m1", "1.0", changelog="init", client=c)
        c.hub_create_version.assert_called_once_with("m1", version="1.0", changelog="init")


class TestSocial:
    def test_star_delegates(self) -> None:
        c = _client(hub_star=MagicMock(return_value={"starred": True}))
        out = hub.star("m1", client=c)
        c.hub_star.assert_called_once_with("m1")
        assert out["starred"] is True

    def test_unstar_delegates(self) -> None:
        c = _client(hub_unstar=MagicMock(return_value={}))
        hub.unstar("m1", client=c)
        c.hub_unstar.assert_called_once_with("m1")

    def test_starred_passes_params(self) -> None:
        c = _client(hub_starred=MagicMock(return_value={"items": []}))
        hub.starred(sort_by="name", page=2, limit=5, client=c)
        c.hub_starred.assert_called_once_with(sort_by="name", page=2, limit=5)

    def test_fork_delegates(self) -> None:
        c = _client(hub_fork=MagicMock(return_value={"id": "m2"}))
        hub.fork("m1", client=c)
        c.hub_fork.assert_called_once_with("m1")

    def test_add_review_delegates(self) -> None:
        c = _client(hub_add_review=MagicMock(return_value={"id": "r1"}))
        hub.add_review("m1", 5, review_text="great", client=c)
        c.hub_add_review.assert_called_once_with("m1", rating=5, review_text="great")

    def test_list_reviews_delegates(self) -> None:
        c = _client(hub_list_reviews=MagicMock(return_value={"items": []}))
        hub.list_reviews("m1", page=2, limit=10, client=c)
        c.hub_list_reviews.assert_called_once_with("m1", page=2, limit=10)


class TestStudioIntegration:
    def test_use_in_studio_delegates(self) -> None:
        c = _client(hub_use_in_studio=MagicMock(return_value={"project_id": "p1"}))
        out = hub.use_in_studio("m1", client=c)
        c.hub_use_in_studio.assert_called_once_with("m1")
        assert out["project_id"] == "p1"


class TestErrorPropagation:
    def test_get_propagates_hub_model_not_found(self) -> None:
        c = _client(hub_get=MagicMock(side_effect=HubModelNotFoundError("m1")))
        with pytest.raises(HubModelNotFoundError):
            hub.get("m1", client=c)


# ---------------------------------------------------------------------------
# New-API delegation branch
#
# The tests above exercise the *legacy* client surface (``hub_search``,
# ``hub_get``, ...), reached via ``getattr(resolved, "hub_*", None)``. A spec'd
# ``MagicMock`` has none of those legacy attributes, so it falls through to the
# *new* client API (``list_hub_models``, ``get_hub_model``, ...) — the path the
# real ``DagnamClient`` takes in production. These tests pin that second branch.
# ---------------------------------------------------------------------------


class TestDiscoveryNewApi:
    def test_search_uses_list_hub_models(self) -> None:
        c = _client(list_hub_models=MagicMock(return_value={"items": [], "total": 3}))
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
        c.list_hub_models.assert_called_once_with(
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
        assert out["total"] == 3

    def test_categories_uses_list_hub_categories(self) -> None:
        c = _client(list_hub_categories=MagicMock(return_value=["nlp", "cv"]))
        assert hub.categories(client=c) == ["nlp", "cv"]
        c.list_hub_categories.assert_called_once_with()

    def test_featured_uses_get_hub_featured(self) -> None:
        c = _client(get_hub_featured=MagicMock(return_value=[{"id": "m1"}]))
        assert hub.featured(client=c) == [{"id": "m1"}]
        c.get_hub_featured.assert_called_once_with()

    def test_trending_uses_get_hub_trending(self) -> None:
        c = _client(get_hub_trending=MagicMock(return_value=[]))
        hub.trending(days=30, client=c)
        c.get_hub_trending.assert_called_once_with(days=30)


class TestModelCRUDNewApi:
    def test_get_uses_get_hub_model(self) -> None:
        c = _client(get_hub_model=MagicMock(return_value={"id": "m1", "name": "test"}))
        out = hub.get("m1", client=c)
        c.get_hub_model.assert_called_once_with("m1")
        assert out["name"] == "test"

    def test_create_uses_create_hub_model(self) -> None:
        c = _client(create_hub_model=MagicMock(return_value={"id": "m1"}))
        hub.create(
            name="my-model",
            description="desc",
            task_type="text-generation",
            framework="pytorch",
            tags=["nlp"],
            metadata={"k": "v"},
            client=c,
        )
        payload = c.create_hub_model.call_args.args[0]
        assert payload["name"] == "my-model"
        assert payload["tags"] == ["nlp"]
        assert payload["metadata"] == {"k": "v"}
        assert payload["license"] == "mit"

    def test_update_uses_update_hub_model(self) -> None:
        c = _client(update_hub_model=MagicMock(return_value={"id": "m1"}))
        hub.update("m1", description="new", client=c)
        c.update_hub_model.assert_called_once_with("m1", {"description": "new"})

    def test_delete_uses_delete_hub_model(self) -> None:
        c = _client(delete_hub_model=MagicMock(return_value=None))
        hub.delete("m1", client=c)
        c.delete_hub_model.assert_called_once_with("m1")


class TestFilesAndVersionsNewApi:
    def test_list_files_uses_list_hub_model_files(self) -> None:
        c = _client(list_hub_model_files=MagicMock(return_value={"files": []}))
        hub.list_files("m1", client=c)
        c.list_hub_model_files.assert_called_once_with("m1")

    def test_download_uses_download_hub_model(self) -> None:
        c = _client(download_hub_model=MagicMock(return_value={"url": "https://x"}))
        hub.download("m1", file_id="f1", client=c)
        c.download_hub_model.assert_called_once_with("m1", file_id="f1")

    def test_list_versions_uses_list_hub_model_versions(self) -> None:
        c = _client(list_hub_model_versions=MagicMock(return_value=[]))
        hub.list_versions("m1", client=c)
        c.list_hub_model_versions.assert_called_once_with("m1")

    def test_create_version_uses_create_hub_model_version(self) -> None:
        c = _client(create_hub_model_version=MagicMock(return_value={"version": "1.0"}))
        hub.create_version("m1", "1.0", changelog="init", client=c)
        payload = c.create_hub_model_version.call_args.args[1]
        assert payload == {"version": "1.0", "changelog": "init"}

    def test_create_version_omits_changelog_when_none(self) -> None:
        c = _client(create_hub_model_version=MagicMock(return_value={"version": "1.0"}))
        hub.create_version("m1", "1.0", client=c)
        payload = c.create_hub_model_version.call_args.args[1]
        assert payload == {"version": "1.0"}


class TestSocialNewApi:
    def test_star_uses_star_hub_model(self) -> None:
        c = _client(star_hub_model=MagicMock(return_value={"starred": True}))
        out = hub.star("m1", client=c)
        c.star_hub_model.assert_called_once_with("m1")
        assert out["starred"] is True

    def test_unstar_uses_unstar_hub_model(self) -> None:
        c = _client(unstar_hub_model=MagicMock(return_value={}))
        hub.unstar("m1", client=c)
        c.unstar_hub_model.assert_called_once_with("m1")

    def test_starred_uses_list_hub_starred(self) -> None:
        c = _client(list_hub_starred=MagicMock(return_value={"items": []}))
        hub.starred(sort_by="name", page=2, limit=5, client=c)
        c.list_hub_starred.assert_called_once_with(sort_by="name", page=2, limit=5)

    def test_fork_uses_fork_hub_model(self) -> None:
        c = _client(fork_hub_model=MagicMock(return_value={"id": "m2"}))
        hub.fork("m1", client=c)
        c.fork_hub_model.assert_called_once_with("m1")

    def test_list_reviews_uses_list_hub_model_reviews(self) -> None:
        c = _client(list_hub_model_reviews=MagicMock(return_value={"items": []}))
        hub.list_reviews("m1", page=2, limit=10, client=c)
        c.list_hub_model_reviews.assert_called_once_with("m1", page=2, limit=10)

    def test_add_review_uses_add_hub_model_review(self) -> None:
        c = _client(add_hub_model_review=MagicMock(return_value={"id": "r1"}))
        hub.add_review("m1", 5, review_text="great", client=c)
        payload = c.add_hub_model_review.call_args.args[1]
        assert payload == {"rating": 5, "review_text": "great"}

    def test_add_review_omits_text_when_none(self) -> None:
        c = _client(add_hub_model_review=MagicMock(return_value={"id": "r1"}))
        hub.add_review("m1", 4, client=c)
        payload = c.add_hub_model_review.call_args.args[1]
        assert payload == {"rating": 4}


class TestStudioIntegrationNewApi:
    def test_use_in_studio_uses_use_hub_model_in_studio(self) -> None:
        c = _client(use_hub_model_in_studio=MagicMock(return_value={"project_id": "p1"}))
        out = hub.use_in_studio("m1", client=c)
        c.use_hub_model_in_studio.assert_called_once_with("m1")
        assert out["project_id"] == "p1"


class TestPayloadValidation:
    def test_update_rejects_non_json_value(self) -> None:
        c = _client(update_hub_model=MagicMock(return_value={}))
        with pytest.raises(TypeError, match="must be JSON-compatible"):
            hub.update("m1", bad=object(), client=c)


class TestFileUpload:
    def test_upload_file_delegates(self) -> None:
        c = _client(
            upload_model_file=MagicMock(
                return_value={"id": "f1", "model_id": "m1", "file_name": "w.bin"}
            )
        )
        out = hub.upload_file("m1", "/tmp/w.bin", client=c)
        c.upload_model_file.assert_called_once_with("m1", "/tmp/w.bin")
        assert out["id"] == "f1"


class TestPublish:
    def _client(self, tmp_path):
        client = MagicMock(spec=DagnamClient)
        client.create_hub_model.return_value = {"id": "m1", "name": "n"}
        client.upload_model_file.return_value = {"id": "f1"}
        client.create_hub_model_version.return_value = {"id": "v1", "version": "1.0.0"}
        client.finalize_hub_model.return_value = {"id": "m1", "status": "published"}
        f1 = tmp_path / "weights.bin"
        f1.write_bytes(b"x")
        f2 = tmp_path / "config.json"
        f2.write_text("{}")
        return client, [str(f1), str(f2)]

    def test_publish_happy_path_create_upload_finalize(self, tmp_path) -> None:
        client, files = self._client(tmp_path)
        progress: list[tuple[str, int, int, str]] = []
        result = hub.publish(
            name="n",
            description="d",
            task_type="t",
            framework="pytorch",
            files=files,
            version="1.0.0",
            changelog="first",
            on_file_progress=lambda p, i, t, s: progress.append((p, i, t, s)),
            client=client,
        )
        client.create_hub_model.assert_called_once()
        assert client.upload_model_file.call_count == 2
        client.create_hub_model_version.assert_called_once_with(
            "m1", {"version": "1.0.0", "changelog": "first"}
        )
        client.finalize_hub_model.assert_called_once_with("m1")
        assert result["finalized"] is True
        model = result["model"]
        assert isinstance(model, dict)
        assert model["id"] == "m1"
        files = result["files"]
        assert isinstance(files, list)
        assert len(files) == 2
        states = [s for (_, _, _, s) in progress]
        assert states == ["uploading", "uploaded", "uploading", "uploaded"]

    def test_publish_retries_then_succeeds(self, tmp_path) -> None:
        from dagnam._core.exceptions import APIError

        client, files = self._client(tmp_path)
        client.upload_model_file.side_effect = [APIError(500, "blip"), {"id": "f1"}, {"id": "f2"}]
        progress: list[str] = []
        result = hub.publish(
            name="n",
            description="d",
            task_type="t",
            framework="pytorch",
            files=files,
            on_file_progress=lambda p, i, t, s: progress.append(s),
            client=client,
        )
        assert client.upload_model_file.call_count == 3
        assert result["finalized"] is True
        assert progress == ["uploading", "retrying", "uploaded", "uploading", "uploaded"]

    def test_publish_exhausted_retries_raises_uploaderror(self, tmp_path) -> None:
        from dagnam._core.exceptions import APIError, UploadError

        client, files = self._client(tmp_path)
        client.upload_model_file.side_effect = APIError(500, "down")
        progress: list[str] = []
        with pytest.raises(UploadError) as exc_info:
            hub.publish(
                name="n",
                description="d",
                task_type="t",
                framework="pytorch",
                files=files,
                max_retries_per_file=1,
                on_file_progress=lambda p, i, t, s: progress.append(s),
                client=client,
            )
        message = str(exc_info.value)
        assert "m1" in message
        assert "weights.bin" in message
        assert "failed" in progress
        client.finalize_hub_model.assert_not_called()

    def test_publish_retries_then_succeeds_without_callback(self, tmp_path) -> None:
        from dagnam._core.exceptions import APIError

        client, files = self._client(tmp_path)
        client.upload_model_file.side_effect = [APIError(500, "blip"), {"id": "f1"}, {"id": "f2"}]
        result = hub.publish(
            name="n",
            description="d",
            task_type="t",
            framework="pytorch",
            files=files,
            client=client,
        )
        assert client.upload_model_file.call_count == 3
        assert result["finalized"] is True

    def test_publish_exhausted_retries_without_callback_raises(self, tmp_path) -> None:
        from dagnam._core.exceptions import APIError, UploadError

        client, files = self._client(tmp_path)
        client.upload_model_file.side_effect = APIError(500, "down")
        with pytest.raises(UploadError):
            hub.publish(
                name="n",
                description="d",
                task_type="t",
                framework="pytorch",
                files=files,
                max_retries_per_file=1,
                client=client,
            )
        client.finalize_hub_model.assert_not_called()

    def test_publish_finalize_route_missing_falls_back(self, tmp_path) -> None:
        from dagnam._core.exceptions import HubModelNotFoundError

        client, files = self._client(tmp_path)
        client.finalize_hub_model.side_effect = HubModelNotFoundError("no such route")
        result = hub.publish(
            name="n",
            description="d",
            task_type="t",
            framework="pytorch",
            files=files,
            client=client,
        )
        assert result["finalized"] is False

    def test_publish_finalize_apierror_405_falls_back_but_500_raises(self, tmp_path) -> None:
        from dagnam._core.exceptions import APIError

        client, files = self._client(tmp_path)
        client.finalize_hub_model.side_effect = APIError(405, "method not allowed")
        assert (
            hub.publish(
                name="n",
                description="d",
                task_type="t",
                framework="pytorch",
                files=files,
                client=client,
            )["finalized"]
            is False
        )
        client.finalize_hub_model.side_effect = APIError(500, "boom")
        with pytest.raises(APIError):
            hub.publish(
                name="n",
                description="d",
                task_type="t",
                framework="pytorch",
                files=files,
                client=client,
            )

    def test_publish_missing_local_file_fails_before_create(self, tmp_path) -> None:
        client, _ = self._client(tmp_path)
        with pytest.raises(FileNotFoundError):
            hub.publish(
                name="n",
                description="d",
                task_type="t",
                framework="pytorch",
                files=[str(tmp_path / "missing.bin")],
                client=client,
            )
        client.create_hub_model.assert_not_called()

    def test_publish_no_version_skips_create_version(self, tmp_path) -> None:
        client, files = self._client(tmp_path)
        result = hub.publish(
            name="n",
            description="d",
            task_type="t",
            framework="pytorch",
            files=files,
            client=client,
        )
        client.create_hub_model_version.assert_not_called()
        assert result["version"] is None
