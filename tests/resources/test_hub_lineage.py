"""``dagnam/resources/hub_lineage.py``: derivatives, design snapshots, version calls and download."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dagnam import hub
from dagnam._core.client import DagnamClient
from dagnam._types import JsonObject

DESIGN: JsonObject = {"diagram_state": {"nodes": []}, "architecture_config": {"layers": []}}


def _client(**overrides: object) -> MagicMock:
    client = MagicMock(spec=DagnamClient)
    client.configure_mock(**overrides)
    return client


class TestVersionCalls:
    def test_derivatives_passes_the_stage_list_and_paging(self) -> None:
        client = _client(list_hub_model_derivatives=MagicMock(return_value={"items": []}))
        assert hub.derivatives("m1", stage="adapter", client=client) == {"items": []}
        client.list_hub_model_derivatives.assert_called_once_with(
            "m1", stage=["adapter"], page=1, limit=20
        )

    def test_derivatives_normalizes_each_stage_and_defaults_to_all(self) -> None:
        client = _client(list_hub_model_derivatives=MagicMock(return_value={}))
        hub.derivatives("m1", stage=["fine-tuned", "merged"], page=2, limit=5, client=client)
        client.list_hub_model_derivatives.assert_called_once_with(
            "m1", stage=["finetuned", "merged"], page=2, limit=5
        )
        client.reset_mock()
        hub.derivatives("m1", client=client)
        client.list_hub_model_derivatives.assert_called_once_with(
            "m1", stage=None, page=1, limit=20
        )

    def test_derivatives_refuses_an_unknown_stage(self) -> None:
        client = _client()
        with pytest.raises(ValueError, match="Unknown stage"):
            hub.derivatives("m1", stage="bogus", client=client)
        client.list_hub_model_derivatives.assert_not_called()

    def test_get_architecture_returns_the_clients_snapshot(self) -> None:
        client = _client(get_hub_model_architecture=MagicMock(return_value=DESIGN))
        assert hub.get_architecture("m1", "v1", client=client) == DESIGN
        client.get_hub_model_architecture.assert_called_once_with("m1", "v1")

    def test_finalize_and_delete_version_delegate(self) -> None:
        client = _client(finalize_hub_model_version=MagicMock(return_value={"id": "v1"}))
        assert hub.finalize_version("m1", "v1", client=client) == {"id": "v1"}
        client.finalize_hub_model_version.assert_called_once_with("m1", "v1")
        assert hub.delete_version("m1", "v1", client=client) is None
        client.delete_hub_model_version.assert_called_once_with("m1", "v1")


class TestDownload:
    def test_download_saves_files_and_returns_their_paths(self, tmp_path: Path) -> None:
        saved = [tmp_path / "model.safetensors", tmp_path / "head.onnx"]
        c = _client(download_hub_model_files=MagicMock(return_value=saved))
        out = hub.download("m1", tmp_path, file_id="f1", client=c)
        c.download_hub_model_files.assert_called_once_with(
            "m1", tmp_path, file_id="f1", version_id=None
        )
        assert out == [str(path) for path in saved]

    def test_a_file_id_saves_no_design_and_reads_no_model(self, tmp_path: Path) -> None:
        c = _client(download_hub_model_files=MagicMock(return_value=[]))
        hub.download("m1", tmp_path, "f1", client=c)
        c.get_hub_model.assert_not_called()
        c.get_hub_model_architecture.assert_not_called()

    def test_the_latest_versions_design_is_saved_beside_the_files(self, tmp_path: Path) -> None:
        weights = tmp_path / "w.safetensors"
        c = _client(
            download_hub_model_files=MagicMock(return_value=[weights]),
            get_hub_model=MagicMock(
                return_value={
                    "name": "My Model v2!",
                    "latest_version": {"id": "v9", "version": "1.2.0", "has_architecture": True},
                }
            ),
            get_hub_model_architecture=MagicMock(return_value=DESIGN),
        )
        out = hub.download("m1", tmp_path / "out", client=c)
        design = tmp_path / "out" / "my-model-v2-1.2.0.dag.json"
        assert out == [str(weights), str(design)]
        assert json.loads(design.read_text(encoding="utf-8")) == DESIGN
        c.get_hub_model_architecture.assert_called_once_with("m1", "v9")
        c.download_hub_model_files.assert_called_once_with(
            "m1", tmp_path / "out", file_id=None, version_id=None
        )

    def test_a_named_version_is_looked_up_in_the_versions_list(self, tmp_path: Path) -> None:
        c = _client(
            download_hub_model_files=MagicMock(return_value=[]),
            get_hub_model=MagicMock(return_value={"name": "!!!", "latest_version": None}),
            list_hub_model_versions=MagicMock(
                return_value=[
                    "junk",
                    {"id": "v1", "version": "1.0.0", "has_architecture": False},
                    {"id": "v2", "version": "2.0.0", "has_architecture": True},
                ]
            ),
            get_hub_model_architecture=MagicMock(return_value=DESIGN),
        )
        out = hub.download("m1", tmp_path, version_id="v2", client=c)
        assert out == [str(tmp_path / "design-2.0.0.dag.json")]
        c.download_hub_model_files.assert_called_once_with(
            "m1", tmp_path, file_id=None, version_id="v2"
        )

    @pytest.mark.parametrize("version_id", [None, "v1", "missing"])
    def test_a_version_without_a_design_saves_none(
        self, tmp_path: Path, version_id: str | None
    ) -> None:
        c = _client(
            download_hub_model_files=MagicMock(return_value=[]),
            get_hub_model=MagicMock(
                return_value={
                    "name": "n",
                    "latest_version": {"id": "v1", "version": "1.0.0", "has_architecture": False},
                }
            ),
            list_hub_model_versions=MagicMock(
                return_value=[{"id": "v1", "version": "1.0.0", "has_architecture": False}]
            ),
        )
        assert hub.download("m1", tmp_path, version_id=version_id, client=c) == []
        c.get_hub_model_architecture.assert_not_called()

    def test_a_hostile_version_cannot_leave_the_destination(self, tmp_path: Path) -> None:
        c = _client(
            download_hub_model_files=MagicMock(return_value=[]),
            get_hub_model=MagicMock(
                return_value={
                    "name": "n",
                    "latest_version": {"id": "v1", "version": "../../x", "has_architecture": True},
                }
            ),
            get_hub_model_architecture=MagicMock(return_value=DESIGN),
        )
        (out,) = hub.download("m1", tmp_path, client=c)
        assert Path(out).parent == tmp_path

    def test_hub_download_lives_in_hub_lineage(self) -> None:
        assert hub.download.__module__ == "dagnam.resources.hub_lineage"
