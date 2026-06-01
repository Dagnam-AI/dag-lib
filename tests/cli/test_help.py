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
        assert format_ascii_art() in out
        assert "Official CLI for Dagnam.AI" in out
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
