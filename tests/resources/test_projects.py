"""Unit tests for dagnam.projects module.

Every facade delegates straight to the real ``DagnamClient`` method surface, so
the client is a ``MagicMock(spec=DagnamClient)`` and the assertions pin the
exact call each facade makes.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dagnam import projects
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import ArchitectureValidationError, ProjectNotFoundError
from dagnam._types import JsonValue


def _client(**overrides: object) -> MagicMock:
    client = MagicMock(spec=DagnamClient)
    client.configure_mock(**overrides)
    return client


_BAD_DIAGRAM: JsonValue = {
    "nodes": [
        {
            "id": "c1",
            "data": {
                "componentId": "convolution-layer",
                "config": {"filters": 8, "kernelSize": 3, "padding": 2},
            },
        }
    ]
}
_GOOD_DIAGRAM: JsonValue = {
    "nodes": [
        {
            "id": "c1",
            "data": {
                "componentId": "convolution-layer",
                "config": {
                    "filters": 8,
                    "kernelSize": 3,
                    "padding": {"mode": "explicit", "value": 2},
                },
            },
        }
    ]
}


class TestReadOperations:
    def test_list_passes_filters_as_kwargs(self) -> None:
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
        args, kwargs = c.list_projects.call_args
        assert args == ()
        assert kwargs["page"] == 2
        assert kwargs["framework"] == "pytorch"
        assert kwargs["tags"] == "nlp,cv"
        assert kwargs["search"] == "foo"

    def test_list_omits_none_filters(self) -> None:
        c = _client(list_projects=MagicMock(return_value={"items": []}))
        projects.list(client=c)
        kwargs = c.list_projects.call_args.kwargs
        assert "framework" not in kwargs
        assert "status" not in kwargs
        assert "visibility" not in kwargs
        assert "tags" not in kwargs
        assert "search" not in kwargs

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

    def test_duplicate_with_title_passes_keyword(self) -> None:
        c = _client(duplicate_project=MagicMock(return_value={"id": "p2"}))
        projects.duplicate("p1", title="Copy", client=c)
        c.duplicate_project.assert_called_once_with("p1", title="Copy")

    def test_duplicate_without_title_passes_none(self) -> None:
        c = _client(duplicate_project=MagicMock(return_value={"id": "p2"}))
        projects.duplicate("p1", client=c)
        c.duplicate_project.assert_called_once_with("p1", title=None)


class TestDAGImport:
    def test_import_dag_builds_payload(self) -> None:
        c = _client(import_dag=MagicMock(return_value={"id": "p1"}))
        projects.import_dag({"nodes": []}, "Imported", tags=["t"], client=c)
        payload = c.import_dag.call_args.args[0]
        assert payload["ir"] == {"nodes": []}
        assert payload["title"] == "Imported"
        assert payload["tags"] == ["t"]

    def test_import_dag_omits_tags_and_description(self) -> None:
        c = _client(import_dag=MagicMock(return_value={"id": "p1"}))
        projects.import_dag({"nodes": []}, "Imported", client=c)
        payload = c.import_dag.call_args.args[0]
        assert "tags" not in payload
        assert "description" not in payload
        assert "commit_message" not in payload

    def test_import_dag_existing_delegates(self) -> None:
        c = _client(import_dag_existing=MagicMock(return_value={"ok": True}))
        projects.import_dag_existing("p1", {"nodes": []}, commit_message="update", client=c)
        pid, payload = c.import_dag_existing.call_args.args
        assert pid == "p1"
        assert payload["ir"] == {"nodes": []}
        assert payload["commit_message"] == "update"

    def test_import_dag_existing_omits_commit_message(self) -> None:
        c = _client(import_dag_existing=MagicMock(return_value={"ok": True}))
        projects.import_dag_existing("p1", {"nodes": []}, client=c)
        payload = c.import_dag_existing.call_args.args[1]
        assert "commit_message" not in payload


class TestBulkAndDatasets:
    def test_bulk_delete_passes_id_list(self) -> None:
        c = _client(bulk_delete_projects=MagicMock(return_value={"deleted": 2}))
        projects.bulk_delete(["p1", "p2"], client=c)
        c.bulk_delete_projects.assert_called_once_with(["p1", "p2"])

    def test_link_dataset_uses_positional_args(self) -> None:
        c = _client(link_dataset=MagicMock(return_value={"ok": True}))
        projects.link_dataset("p1", "d1", "training", client=c)
        c.link_dataset.assert_called_once_with("p1", "d1", "training")

    def test_get_datasets_delegates(self) -> None:
        c = _client(get_project_datasets=MagicMock(return_value={"datasets": []}))
        projects.get_datasets("p1", client=c)
        c.get_project_datasets.assert_called_once_with("p1")

    def test_unlink_dataset_uses_positional_args(self) -> None:
        c = _client(unlink_dataset=MagicMock(return_value=None))
        projects.unlink_dataset("p1", "d1", client=c)
        c.unlink_dataset.assert_called_once_with("p1", "d1")


class TestSaveArchitecture:
    def test_delegates_to_save_architecture(self) -> None:
        c = _client(save_architecture=MagicMock(return_value={"ok": True}))
        projects.save_architecture(
            "p1", {"nodes": []}, {"layers": []}, commit_message="v1", client=c
        )
        c.save_architecture.assert_called_once()
        pid, payload = c.save_architecture.call_args.args
        assert pid == "p1"
        assert payload["diagram_state"] == {"nodes": []}
        assert payload["commit_message"] == "v1"

    def test_omits_commit_message(self) -> None:
        c = _client(save_architecture=MagicMock(return_value={"ok": True}))
        projects.save_architecture("p1", {"nodes": []}, {"layers": []}, client=c)
        payload = c.save_architecture.call_args.args[1]
        assert "commit_message" not in payload

    def test_rejects_bare_int_padding_before_persisting(self) -> None:
        """An SDK-built model with bare-int padding is invalid in the Studio, so the
        SDK rejects it BEFORE any network call, mirroring the backend's gate."""
        c = _client(save_architecture=MagicMock(return_value={"version_id": "v1"}))
        with pytest.raises(ArchitectureValidationError) as exc:
            projects.save_architecture("p1", _BAD_DIAGRAM, {"layers": []}, client=c)
        assert any(e.node_id == "c1" for e in exc.value.errors)
        c.save_architecture.assert_not_called()  # never reached the wire

    def test_persists_when_padding_is_typed(self) -> None:
        c = _client(save_architecture=MagicMock(return_value={"version_id": "v1"}))
        assert projects.save_architecture("p1", _GOOD_DIAGRAM, {"layers": []}, client=c) == {
            "version_id": "v1"
        }
        c.save_architecture.assert_called_once()

    def test_bypass_skips_local_validation(self) -> None:
        """validate_locally=False preserves the bypass for power users; the
        normalizer still upgrades the bare-int form before it is sent."""
        c = _client(save_architecture=MagicMock(return_value={"version_id": "v1"}))
        projects.save_architecture(
            "p1", _BAD_DIAGRAM, {"layers": []}, validate_locally=False, client=c
        )
        c.save_architecture.assert_called_once()
        _pid, payload = c.save_architecture.call_args.args
        assert payload["diagram_state"]["nodes"][0]["data"]["config"]["padding"] == {
            "mode": "explicit",
            "value": 2,
        }

    def test_upgrades_tolerated_legacy_string_padding(self) -> None:
        """Legacy bare 'same'/'valid' strings pass validation and are upgraded to
        the canonical typed form before persisting."""
        c = _client(save_architecture=MagicMock(return_value={"version_id": "v1"}))
        diagram: JsonValue = {
            "nodes": [
                {
                    "id": "c1",
                    "data": {
                        "componentId": "convolution-layer",
                        "config": {"filters": 8, "kernelSize": 3, "padding": "same"},
                    },
                }
            ]
        }
        projects.save_architecture(
            "p1",
            diagram,
            {"layers": [{"id": "c1", "type": "conv2d", "config": {"padding": "valid"}}]},
            client=c,
        )
        _pid, payload = c.save_architecture.call_args.args
        assert payload["diagram_state"]["nodes"][0]["data"]["config"]["padding"] == {"mode": "same"}
        assert payload["architecture_config"]["layers"][0]["config"]["padding"] == {"mode": "valid"}

    def test_skips_validation_for_non_mapping_diagram(self) -> None:
        # A non-mapping diagram_state can't be walked; the guard is skipped and the
        # value is forwarded unchanged (normalizer is also a no-op on non-mappings).
        c = _client(save_architecture=MagicMock(return_value={"version_id": "v1"}))
        projects.save_architecture("p1", ["not", "a", "mapping"], {"layers": []}, client=c)
        c.save_architecture.assert_called_once()
        _pid, payload = c.save_architecture.call_args.args
        assert payload["diagram_state"] == ["not", "a", "mapping"]


class TestErrorPropagation:
    def test_get_propagates_project_not_found(self) -> None:
        c = _client()
        c.get_project.side_effect = ProjectNotFoundError("p1")
        with pytest.raises(ProjectNotFoundError):
            projects.get("p1", client=c)


class TestVersionOperations:
    def test_list_versions_delegates(self) -> None:
        c = _client(list_project_versions=MagicMock(return_value={"items": [], "total": 0}))
        out = projects.list_versions("p1", page=2, limit=5, client=c)
        c.list_project_versions.assert_called_once_with("p1", page=2, limit=5)
        assert "items" in out

    def test_get_version_delegates(self) -> None:
        c = _client(get_project_version=MagicMock(return_value={"id": "v1"}))
        out = projects.get_version("p1", "v1", client=c)
        c.get_project_version.assert_called_once_with("p1", "v1")
        assert out["id"] == "v1"

    def test_compare_versions_delegates(self) -> None:
        c = _client(compare_project_versions=MagicMock(return_value={"version_a": {}}))
        out = projects.compare_versions("p1", "va", "vb", client=c)
        c.compare_project_versions.assert_called_once_with("p1", "va", "vb")
        assert "version_a" in out

    def test_restore_version_delegates(self) -> None:
        c = _client(
            restore_project_version=MagicMock(return_value={"id": "v2", "is_current": True})
        )
        out = projects.restore_version("p1", "v1", client=c)
        c.restore_project_version.assert_called_once_with("p1", "v1")
        assert out["is_current"] is True

    def test_delete_version_delegates(self) -> None:
        c = _client(delete_project_version=MagicMock(return_value=None))
        assert projects.delete_version("p1", "v1", client=c) is None
        c.delete_project_version.assert_called_once_with("p1", "v1")

    def test_latest_version_delegates(self) -> None:
        c = _client(get_latest_project_version=MagicMock(return_value={"id": "v2"}))
        out = projects.latest_version("p1", client=c)
        c.get_latest_project_version.assert_called_once_with("p1")
        assert out["id"] == "v2"
