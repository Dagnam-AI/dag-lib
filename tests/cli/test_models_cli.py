"""Tests for the `dagnam models` CLI subcommand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock
from unittest.mock import MagicMock, patch

from dagnam.cli.models import register_models

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, StrCapture


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_models(subparsers)
    return parser


class TestRegisterModels:
    # ------------------------------------------------------------- push

    def test_push_parses_repeatable_file_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "models",
                "push",
                "--name",
                "tiny-chat",
                "--slug",
                "tiny-chat",
                "--description",
                "d",
                "--file",
                "a.safetensors",
                "--file",
                "config.json",
            ]
        )
        assert args.file == ["a.safetensors", "config.json"]

    @patch("dagnam.cli.models.models.push")
    def test_push_invokes_resource_push(self, mock_push: MagicMock) -> None:
        mock_push.return_value = {"id": "v1", "status": "ready"}
        parser = _build_parser()
        args = parser.parse_args(
            [
                "models",
                "push",
                "--name",
                "tiny-chat",
                "--slug",
                "tiny-chat",
                "--description",
                "d",
                "--file",
                "a.safetensors",
            ]
        )
        args.func(args)
        mock_push.assert_called_once()

    def test_push_passes_all_kwargs_and_defaults(self, capsys: StrCapture) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "models",
                "push",
                "--name",
                "tiny-chat",
                "--slug",
                "tiny-chat",
                "--description",
                "d",
                "--file",
                "a.safetensors",
            ]
        )
        with mock.patch(
            "dagnam.cli.models.models.push",
            return_value={"id": "v1", "status": "ready"},
        ) as mock_push:
            args.func(args)
        mock_push.assert_called_once_with(
            name="tiny-chat",
            slug="tiny-chat",
            description="d",
            files=["a.safetensors"],
            origin="imported",
            license="mit",
            visibility="private",
        )
        # I6: the default (no --json flag) output must be real JSON, not a
        # Python dict repr (single-quoted) -- json.loads raises on the latter.
        assert json.loads(capsys.readouterr().out) == {"id": "v1", "status": "ready"}

    def test_push_json_flag_prints_json(self, capsys: StrCapture) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "models",
                "push",
                "--name",
                "tiny-chat",
                "--slug",
                "tiny-chat",
                "--description",
                "d",
                "--file",
                "a.safetensors",
                "--origin",
                "finetuned",
                "--license",
                "apache-2.0",
                "--visibility",
                "public",
                "--json",
            ]
        )
        with mock.patch(
            "dagnam.cli.models.models.push",
            return_value={"id": "v1", "status": "ready"},
        ) as mock_push:
            args.func(args)
        mock_push.assert_called_once_with(
            name="tiny-chat",
            slug="tiny-chat",
            description="d",
            files=["a.safetensors"],
            origin="finetuned",
            license="apache-2.0",
            visibility="public",
        )
        assert json.loads(capsys.readouterr().out) == {"id": "v1", "status": "ready"}

    # --------------------------------------------------------------- get

    def test_get_invokes_client_and_prints(
        self, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        monkeypatch.setenv("DAGNAM_API_KEY", "k")
        parser = _build_parser()
        args = parser.parse_args(["models", "get", "m1"])
        with mock.patch(
            "dagnam._core.client.DagnamClient.get_model_entry",
            return_value={"id": "m1", "name": "Tiny"},
        ) as mock_get:
            args.func(args)
        mock_get.assert_called_once_with("m1")
        # I6: real JSON by default, not a Python dict repr.
        assert json.loads(capsys.readouterr().out) == {"id": "m1", "name": "Tiny"}

    def test_get_json_flag_prints_json(
        self, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        monkeypatch.setenv("DAGNAM_API_KEY", "k")
        parser = _build_parser()
        args = parser.parse_args(["models", "get", "m1", "--json"])
        with mock.patch(
            "dagnam._core.client.DagnamClient.get_model_entry",
            return_value={"id": "m1"},
        ):
            args.func(args)
        assert json.loads(capsys.readouterr().out) == {"id": "m1"}

    # -------------------------------------------------------------- list

    def test_list_defaults_search_page_limit(
        self, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        monkeypatch.setenv("DAGNAM_API_KEY", "k")
        parser = _build_parser()
        args = parser.parse_args(["models", "list"])
        with mock.patch(
            "dagnam._core.client.DagnamClient.list_model_entries",
            return_value=[{"id": "m1"}],
        ) as mock_list:
            args.func(args)
        mock_list.assert_called_once_with(search=None, page=1, limit=20)
        assert "m1" in capsys.readouterr().out

    def test_list_passes_search_page_limit_and_json(
        self, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        monkeypatch.setenv("DAGNAM_API_KEY", "k")
        parser = _build_parser()
        args = parser.parse_args(
            ["models", "list", "--search", "chat", "--page", "2", "--limit", "5", "--json"]
        )
        with mock.patch(
            "dagnam._core.client.DagnamClient.list_model_entries",
            return_value=[{"id": "m2"}],
        ) as mock_list:
            args.func(args)
        mock_list.assert_called_once_with(search="chat", page=2, limit=5)
        assert json.loads(capsys.readouterr().out) == [{"id": "m2"}]

    def test_list_output_flag_saves_full_json(
        self, monkeypatch: PytestMonkeyPatch, capsys: StrCapture, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("DAGNAM_API_KEY", "k")
        output = tmp_path / "models.json"
        payload = [{"id": "m1"}]
        parser = _build_parser()
        args = parser.parse_args(["models", "list", "--output", str(output)])
        with mock.patch(
            "dagnam._core.client.DagnamClient.list_model_entries",
            return_value=payload,
        ):
            args.func(args)
        assert json.loads(output.read_text(encoding="utf-8")) == payload
        assert "m1" in capsys.readouterr().out

    # ---------------------------------------------------------- download

    @patch("dagnam.cli.models.models.download")
    def test_download_invokes_resource_download(self, mock_download: MagicMock) -> None:
        mock_download.return_value = "/tmp/artifact.bin"
        parser = _build_parser()
        args = parser.parse_args(["models", "download", "v1", "a1"])
        args.func(args)
        mock_download.assert_called_once_with("v1", "a1", cache_dir=None)

    def test_download_passes_output_dir_and_prints_path(self, capsys: StrCapture) -> None:
        parser = _build_parser()
        args = parser.parse_args(["models", "download", "v1", "a1", "--output-dir", "/tmp/models"])
        with mock.patch(
            "dagnam.cli.models.models.download",
            return_value=Path("/tmp/models/a1.bin"),
        ) as mock_download:
            args.func(args)
        mock_download.assert_called_once_with("v1", "a1", cache_dir="/tmp/models")
        # I6: the default output is the same JSON shape as --json, not a bare
        # printed path.
        assert json.loads(capsys.readouterr().out) == {"path": "/tmp/models/a1.bin"}

    def test_download_json_flag_prints_json(self, capsys: StrCapture) -> None:
        parser = _build_parser()
        args = parser.parse_args(["models", "download", "v1", "a1", "--json"])
        with mock.patch(
            "dagnam.cli.models.models.download",
            return_value=Path("/tmp/models/a1.bin"),
        ):
            args.func(args)
        assert json.loads(capsys.readouterr().out) == {"path": "/tmp/models/a1.bin"}

    # -------------------------------------------------------------- lineage

    def test_lineage_invokes_resource_and_prints(self, capsys: StrCapture) -> None:
        parser = _build_parser()
        args = parser.parse_args(["models", "lineage", "v1"])
        with mock.patch(
            "dagnam.cli.models.models.get_lineage",
            return_value={"parents": []},
        ) as mock_lineage:
            args.func(args)
        mock_lineage.assert_called_once_with("v1")
        # I6: real JSON by default, not a Python dict repr.
        assert json.loads(capsys.readouterr().out) == {"parents": []}

    def test_lineage_json_flag_prints_json(self, capsys: StrCapture) -> None:
        parser = _build_parser()
        args = parser.parse_args(["models", "lineage", "v1", "--json"])
        with mock.patch(
            "dagnam.cli.models.models.get_lineage",
            return_value={"parents": []},
        ):
            args.func(args)
        assert json.loads(capsys.readouterr().out) == {"parents": []}

    # --------------------------------------------------------- task-contract

    def test_task_contract_invokes_resource_and_prints(self, capsys: StrCapture) -> None:
        parser = _build_parser()
        args = parser.parse_args(["models", "task-contract", "chat-completion", "1"])
        with mock.patch(
            "dagnam.cli.models.models.get_task_contract",
            return_value={"key": "chat-completion", "version": "1"},
        ) as mock_contract:
            args.func(args)
        mock_contract.assert_called_once_with("chat-completion", "1")
        # I6: real JSON by default, not a Python dict repr.
        assert json.loads(capsys.readouterr().out) == {"key": "chat-completion", "version": "1"}

    def test_task_contract_json_flag_prints_json(self, capsys: StrCapture) -> None:
        parser = _build_parser()
        args = parser.parse_args(["models", "task-contract", "chat-completion", "1", "--json"])
        with mock.patch(
            "dagnam.cli.models.models.get_task_contract",
            return_value={"key": "chat-completion"},
        ):
            args.func(args)
        assert json.loads(capsys.readouterr().out) == {"key": "chat-completion"}
