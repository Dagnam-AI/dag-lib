"""Unit tests for dagnam.projects module."""

from __future__ import annotations
from tests.typing_helpers import JsonObject

from unittest.mock import MagicMock

import pytest

from dagnam import projects
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import ProjectNotFoundError


def _client(**overrides: JsonObject) -> MagicMock:
    return MagicMock(spec=DagnamClient, **overrides)


class TestReadOperations:
    def test_list_passes_filters(self) -> None:
        c = _client(list_projects=MagicMock(return_value={"items": [], "total": 0}))
        projects.list(
            page=2,
            limit=10,
            framework="pytorch",
            status="active",
            visibility="public",
            tags=["nlp", "cv"],
            search="foo",
            sort_by="created_at",
            order="asc",
            client=c,
        )
        params = c.list_projects.call_args.kwargs["params"]
        assert params["page"] == 2
        assert params["framework"] == "pytorch"
        assert params["tags"] == "nlp,cv"
        assert params["search"] == "foo"

    def test_list_omits_none_filters(self) -> None:
        c = _client(list_projects=MagicMock(return_value={"items": []}))
        projects.list(client=c)
        params = c.list_projects.call_args.kwargs["params"]
        assert "framework" not in params
        assert "status" not in params
        assert "tags" not in params

    def test_get_delegates(self) -> None:
        c = _client(get_project=MagicMock(return_value={"id": "p1", "title": "T"}))
        out = projects.get("p1", client=c)
        c.get_project.assert_called_once_with("p1")
        assert out["title"] == "T"


class TestWriteOperations:
    def test_create_builds_payload(self) -> None:
        c = _client(create_project=MagicMock(return_value={"id": "p1"}))
        projects.create("My Model", framework="pytorch", tags=["nlp"], client=c)
        payload = c.create_project.call_args.args[0]
        assert payload["title"] == "My Model"
        assert payload["framework"] == "pytorch"
        assert payload["tags"] == ["nlp"]

    def test_create_omits_none_optionals(self) -> None:
        c = _client(create_project=MagicMock(return_value={"id": "p1"}))
        projects.create("M", client=c)
        payload = c.create_project.call_args.args[0]
        assert "description" not in payload
        assert "tags" not in payload

    def test_update_delegates(self) -> None:
        c = _client(update_project=MagicMock(return_value={"id": "p1"}))
        projects.update("p1", title="New", description="d", client=c)
        c.update_project.assert_called_once_with("p1", {"title": "New", "description": "d"})

    def test_delete_delegates(self) -> None:
        c = _client()
        projects.delete("p1", client=c)
        c.delete_project.assert_called_once_with("p1")

    def test_duplicate_with_title(self) -> None:
        c = _client(duplicate_project=MagicMock(return_value={"id": "p2"}))
        projects.duplicate("p1", title="Copy", client=c)
        c.duplicate_project.assert_called_once_with("p1", {"title": "Copy"})

    def test_duplicate_without_title(self) -> None:
        c = _client(duplicate_project=MagicMock(return_value={"id": "p2"}))
        projects.duplicate("p1", client=c)
        c.duplicate_project.assert_called_once_with("p1", None)

    def test_save_architecture_delegates(self) -> None:
        c = _client(save_project_architecture=MagicMock(return_value={"ok": True}))
        projects.save_architecture(
            "p1", {"nodes": []}, {"layers": []}, commit_message="v1", client=c
        )
        payload = c.save_project_architecture.call_args.args[1]
        assert payload["diagram_state"] == {"nodes": []}
        assert payload["commit_message"] == "v1"


class TestDAGImport:
    def test_import_dag_builds_payload(self) -> None:
        c = _client(import_project_dag=MagicMock(return_value={"id": "p1"}))
        projects.import_dag({"nodes": []}, "Imported", tags=["t"], client=c)
        payload = c.import_project_dag.call_args.args[0]
        assert payload["ir"] == {"nodes": []}
        assert payload["title"] == "Imported"
        assert payload["tags"] == ["t"]

    def test_import_dag_existing_delegates(self) -> None:
        c = _client(import_project_dag_existing=MagicMock(return_value={"ok": True}))
        projects.import_dag_existing("p1", {"nodes": []}, commit_message="update", client=c)
        payload = c.import_project_dag_existing.call_args.args[1]
        assert payload["ir"] == {"nodes": []}
        assert payload["commit_message"] == "update"


class TestBulkAndDatasets:
    def test_bulk_delete_delegates(self) -> None:
        c = _client(bulk_delete_projects=MagicMock(return_value={"deleted": 2}))
        projects.bulk_delete(["p1", "p2"], client=c)
        c.bulk_delete_projects.assert_called_once_with({"project_ids": ["p1", "p2"]})

    def test_link_dataset_delegates(self) -> None:
        c = _client(link_project_dataset=MagicMock(return_value={"ok": True}))
        projects.link_dataset("p1", "d1", "training", client=c)
        c.link_project_dataset.assert_called_once_with(
            "p1", {"dataset_id": "d1", "role": "training"}
        )

    def test_get_datasets_delegates(self) -> None:
        c = _client(get_project_datasets=MagicMock(return_value={"datasets": []}))
        projects.get_datasets("p1", client=c)
        c.get_project_datasets.assert_called_once_with("p1")

    def test_unlink_dataset_delegates(self) -> None:
        c = _client(unlink_project_dataset=MagicMock(return_value=None))
        projects.unlink_dataset("p1", "d1", client=c)
        c.unlink_project_dataset.assert_called_once_with("p1", "d1")


class TestErrorPropagation:
    def test_get_propagates_project_not_found(self) -> None:
        c = _client()
        c.get_project.side_effect = ProjectNotFoundError("p1")
        with pytest.raises(ProjectNotFoundError):
            projects.get("p1", client=c)
