"""Unit tests for dagnam.projects module."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast
from unittest.mock import MagicMock, Mock

import pytest

from dagnam import projects
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import ProjectNotFoundError


def _client(**overrides: object) -> MagicMock:
    client = MagicMock(spec=DagnamClient)
    client.configure_mock(**overrides)
    return client


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


# ---------------------------------------------------------------------------
# New-API delegation branch
#
# The tests above cover the *legacy* client surface. Three facades (``list``,
# ``duplicate``, ``bulk_delete``) branch on ``_is_mock_client`` and so always
# take the legacy path for a ``Mock``; the *new* client API
# (``resolved.list_projects(**params)``, ...) is only reachable with a real
# (non-Mock) client. A hand-rolled recording fake exercises those, while a
# spec'd ``MagicMock`` (via ``_client``) covers the remaining facades whose
# legacy branch is a ``getattr`` miss.
# ---------------------------------------------------------------------------


class _RecordingClient:
    """Minimal non-Mock client that records the last call per method.

    ``projects._is_mock_client`` returns ``False`` for this type (it is not a
    ``unittest.mock.Mock``), so the facades take their real new-API branch.
    """

    def __init__(self, **returns: object) -> None:
        self._returns = returns
        self.calls: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}

    def __getattr__(self, name: str) -> Callable[..., object]:
        def _method(*args: Any, **kwargs: Any) -> object:
            self.calls[name] = (args, kwargs)
            return self._returns.get(name, {})

        return _method

    def as_client(self) -> DagnamClient:
        """Return ``self`` typed as a ``DagnamClient`` for the facade ``client=``."""
        return cast("DagnamClient", self)


class TestMockGatedNewApi:
    """Facades gated by ``_is_mock_client`` — exercised with a non-Mock fake."""

    def test_list_uses_kwargs_call(self) -> None:
        c = _RecordingClient(list_projects={"items": [], "total": 0})
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
            client=c.as_client(),
        )
        args, kwargs = c.calls["list_projects"]
        assert args == ()
        assert kwargs["page"] == 2
        assert kwargs["framework"] == "pytorch"
        assert kwargs["tags"] == "nlp,cv"
        assert kwargs["search"] == "foo"

    def test_duplicate_with_title_passes_keyword(self) -> None:
        c = _RecordingClient(duplicate_project={"id": "p2"})
        projects.duplicate("p1", title="Copy", client=c.as_client())
        args, kwargs = c.calls["duplicate_project"]
        assert args == ("p1",)
        assert kwargs == {"title": "Copy"}

    def test_duplicate_without_title_passes_none(self) -> None:
        c = _RecordingClient(duplicate_project={"id": "p2"})
        projects.duplicate("p1", client=c.as_client())
        _args, kwargs = c.calls["duplicate_project"]
        assert kwargs == {"title": None}

    def test_bulk_delete_passes_id_list(self) -> None:
        c = _RecordingClient(bulk_delete_projects={"deleted": 2})
        projects.bulk_delete(["p1", "p2"], client=c.as_client())
        args, _kwargs = c.calls["bulk_delete_projects"]
        assert args == (["p1", "p2"],)


class TestMockWithoutLegacyMethodFallsThrough:
    """The ``_is_mock_client`` legacy branch is gated by ``callable(getattr(...))``.

    A ``Mock`` with no spec'd methods (``Mock(spec=[])``) is still a ``Mock`` —
    so ``_is_mock_client`` is True — but ``getattr(client, "<legacy>", None)``
    returns ``None`` (non-callable), exercising the false leg that falls through
    to the new-API call ``resolved.<legacy>(**params)``. That attribute is also
    blocked by the empty spec, so the facade raises ``AttributeError``. Pins the
    degenerate-input behavior (covers branches ``76->78``/``175->177``/``283->285``).
    """

    def test_list_falls_through_to_new_api(self) -> None:
        c = cast("DagnamClient", Mock(spec=[]))
        with pytest.raises(AttributeError):
            projects.list(client=c)

    def test_duplicate_falls_through_to_new_api(self) -> None:
        c = cast("DagnamClient", Mock(spec=[]))
        with pytest.raises(AttributeError):
            projects.duplicate("p1", client=c)

    def test_bulk_delete_falls_through_to_new_api(self) -> None:
        c = cast("DagnamClient", Mock(spec=[]))
        with pytest.raises(AttributeError):
            projects.bulk_delete(["p1"], client=c)


class TestGetattrGatedNewApi:
    """Facades whose legacy branch is a ``getattr`` miss on a spec'd mock."""

    def test_save_architecture_uses_save_architecture(self) -> None:
        c = _client(save_architecture=MagicMock(return_value={"ok": True}))
        projects.save_architecture(
            "p1", {"nodes": []}, {"layers": []}, commit_message="v1", client=c
        )
        c.save_architecture.assert_called_once()
        pid, payload = c.save_architecture.call_args.args
        assert pid == "p1"
        assert payload["diagram_state"] == {"nodes": []}
        assert payload["commit_message"] == "v1"

    def test_save_architecture_normalizes_bare_int_padding(self) -> None:
        """An SDK-built model with legacy bare-int padding is upgraded to the
        canonical typed form before it is sent, so it can never persist in a
        state the Studio would reject (closes the e2e-06 hole at the SDK side)."""
        c = _client(save_architecture=MagicMock(return_value={"ok": True}))
        projects.save_architecture(
            "p1",
            {
                "nodes": [
                    {
                        "id": "c1",
                        "data": {
                            "componentId": "convolution-layer",
                            "config": {"filters": 8, "kernelSize": 3, "padding": 2},
                        },
                    }
                ]
            },
            {
                "layers": [
                    {"id": "c1", "type": "conv2d", "config": {"padding": 2, "filters": 8, "kernelSize": 3}}
                ],
                "connections": [],
            },
            client=c,
        )
        _pid, payload = c.save_architecture.call_args.args
        assert payload["diagram_state"]["nodes"][0]["data"]["config"]["padding"] == {
            "mode": "explicit",
            "value": 2,
        }
        assert payload["architecture_config"]["layers"][0]["config"]["padding"] == {
            "mode": "explicit",
            "value": 2,
        }

    def test_import_dag_uses_import_dag(self) -> None:
        c = _client(import_dag=MagicMock(return_value={"id": "p1"}))
        projects.import_dag({"nodes": []}, "Imported", tags=["t"], client=c)
        payload = c.import_dag.call_args.args[0]
        assert payload["ir"] == {"nodes": []}
        assert payload["title"] == "Imported"
        assert payload["tags"] == ["t"]

    def test_import_dag_existing_uses_import_dag_existing(self) -> None:
        c = _client(import_dag_existing=MagicMock(return_value={"ok": True}))
        projects.import_dag_existing("p1", {"nodes": []}, commit_message="update", client=c)
        pid, payload = c.import_dag_existing.call_args.args
        assert pid == "p1"
        assert payload["ir"] == {"nodes": []}
        assert payload["commit_message"] == "update"

    def test_link_dataset_uses_positional_args(self) -> None:
        c = _client(link_dataset=MagicMock(return_value={"ok": True}))
        projects.link_dataset("p1", "d1", "training", client=c)
        c.link_dataset.assert_called_once_with("p1", "d1", "training")

    def test_unlink_dataset_uses_positional_args(self) -> None:
        c = _client(unlink_dataset=MagicMock(return_value=None))
        projects.unlink_dataset("p1", "d1", client=c)
        c.unlink_dataset.assert_called_once_with("p1", "d1")


class TestOptionalPayloadOmission:
    """Exercise the ``if <optional> is not None`` skip branches."""

    def test_save_architecture_omits_commit_message(self) -> None:
        c = _client(save_architecture=MagicMock(return_value={"ok": True}))
        projects.save_architecture("p1", {"nodes": []}, {"layers": []}, client=c)
        payload = c.save_architecture.call_args.args[1]
        assert "commit_message" not in payload

    def test_import_dag_omits_tags_and_description(self) -> None:
        c = _client(import_dag=MagicMock(return_value={"id": "p1"}))
        projects.import_dag({"nodes": []}, "Imported", client=c)
        payload = c.import_dag.call_args.args[0]
        assert "tags" not in payload
        assert "description" not in payload
        assert "commit_message" not in payload

    def test_import_dag_existing_omits_commit_message(self) -> None:
        c = _client(import_dag_existing=MagicMock(return_value={"ok": True}))
        projects.import_dag_existing("p1", {"nodes": []}, client=c)
        payload = c.import_dag_existing.call_args.args[1]
        assert "commit_message" not in payload


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
