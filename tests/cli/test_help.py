"""Smoke tests for CLI help text."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dagnam.cli import main
from dagnam.cli.common import format_ascii_art

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, StrCapture


class TestHelpText:
    def test_top_level_help_includes_description_examples_and_new_commands(
        self, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        monkeypatch.setattr("sys.argv", ["dagnam", "--help"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "Official CLI for Dagnam.AI" in out
        assert format_ascii_art() not in out  # compact help omits the full banner
        assert "Examples:" in out
        assert "version" in out
        assert "whoami" in out
        assert "config" in out

    def test_dataset_list_help_includes_argument_descriptions(
        self, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        monkeypatch.setattr("sys.argv", ["dagnam", "dataset", "list", "--help"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "List available datasets" in out
        assert "Filter by dataset type" in out

    def test_command_group_help_uses_compact_sections(
        self, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        monkeypatch.setattr("sys.argv", ["dagnam", "projects", "--help"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "Usage: dagnam projects [-h] <command> ..." in out
        assert "Create, list, inspect, and delete projects." in out
        assert "Commands:" in out
        assert "  list          List projects." in out
        assert "  create        Create a project." in out
        assert "  architecture  Save a project's architecture." in out
        assert "Options:" in out
        assert "  -h, --help    Show this help and exit." in out
        assert "Docs: https://dagnam.ai/docs" in out
        assert "positional arguments:" not in out

    def test_config_get_help_describes_key(
        self, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        monkeypatch.setattr("sys.argv", ["dagnam", "config", "get", "--help"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        assert "Config key to read" in capsys.readouterr().out

    def test_config_set_help_mentions_training_metrics_path(
        self, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        monkeypatch.setattr("sys.argv", ["dagnam", "config", "set", "--help"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "training_metrics_path" in out

    def test_login_help_mentions_training_metrics_path(
        self, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        monkeypatch.setattr("sys.argv", ["dagnam", "login", "--help"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "--training-metrics-path" in out

    def test_version_help_includes_environment_info(
        self, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        monkeypatch.setattr("sys.argv", ["dagnam", "version", "--help"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        assert "Python version" in capsys.readouterr().out
