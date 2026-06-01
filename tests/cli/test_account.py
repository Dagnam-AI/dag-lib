"""Tests for version/whoami/logout/config CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from dagnam.cli import main

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, StrCapture


class TestVersionSubcommand:
    def test_prints_version_line(self, monkeypatch: PytestMonkeyPatch, capsys: StrCapture) -> None:
        monkeypatch.setattr("sys.argv", ["dagnam", "version"])
        main()
        out = capsys.readouterr().out
        assert out.startswith("dagnam ")
        assert "Python " in out

    def test_json_output_is_machine_readable(
        self, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        monkeypatch.setattr("sys.argv", ["dagnam", "version", "--json"])
        main()
        payload = json.loads(capsys.readouterr().out)
        assert set(payload) == {"dagnam", "platform", "python"}


class TestWhoami:
    def test_reads_key_from_config(
        self, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text('{"api_key": "sk_abcdefghijklmnop"}', encoding="utf-8")
        monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
        monkeypatch.delenv("DAGNAM_API_KEY", raising=False)
        monkeypatch.delenv("DAGNAM_API_URL", raising=False)
        monkeypatch.setattr("dagnam._core.auth._api_key", None)
        monkeypatch.setattr("dagnam._core.auth._api_url", None)
        monkeypatch.setattr("sys.argv", ["dagnam", "whoami"])

        main()

        out = capsys.readouterr().out
        assert "sk_abc...mnop" in out
        assert "config file" in out
        assert "Source: config file" in out
        assert "sk_abcdefghijklmnop" not in out

    def test_env_var_source(
        self, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", tmp_path / "missing.json")
        monkeypatch.setenv("DAGNAM_API_KEY", "sk_envkey1234567890")
        monkeypatch.setattr("dagnam._core.auth._api_key", None)
        monkeypatch.setattr("dagnam._core.auth._api_url", None)
        monkeypatch.setattr("sys.argv", ["dagnam", "whoami"])

        main()

        assert "environment" in capsys.readouterr().out.lower()

    def test_not_logged_in_exits_1(
        self, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", tmp_path / "missing.json")
        monkeypatch.delenv("DAGNAM_API_KEY", raising=False)
        monkeypatch.setattr("dagnam._core.auth._api_key", None)
        monkeypatch.setattr("sys.argv", ["dagnam", "whoami"])

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        assert "not logged in" in capsys.readouterr().err.lower()


class TestLogout:
    def test_removes_key_keeps_url(
        self, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(
            '{"api_key": "sk_x", "api_url": "https://example.test"}', encoding="utf-8"
        )
        monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
        monkeypatch.delenv("DAGNAM_API_KEY", raising=False)
        monkeypatch.setattr("sys.argv", ["dagnam", "logout"])

        main()

        data = json.loads(config_file.read_text(encoding="utf-8"))
        assert "api_key" not in data
        assert data["api_url"] == "https://example.test"
        assert "logged out" in capsys.readouterr().out.lower()

    def test_no_config_file(
        self, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", tmp_path / "missing.json")
        monkeypatch.setattr("sys.argv", ["dagnam", "logout"])

        main()

        assert "not logged in" in capsys.readouterr().out.lower()

    def test_warns_when_env_var_set(
        self, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text('{"api_key": "sk_x"}', encoding="utf-8")
        monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
        monkeypatch.setenv("DAGNAM_API_KEY", "sk_envvalue")
        monkeypatch.setattr("sys.argv", ["dagnam", "logout"])

        main()

        assert "DAGNAM_API_KEY" in capsys.readouterr().err


class TestConfig:
    def test_list_masks_key(
        self, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(
            '{"api_key": "sk_abcdefghijklmnop", "api_url": "https://e.test"}',
            encoding="utf-8",
        )
        monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
        monkeypatch.setattr("sys.argv", ["dagnam", "config", "list"])

        main()

        out = capsys.readouterr().out
        assert "sk_abc...mnop" in out
        assert "sk_abcdefghijklmnop" not in out
        assert "https://e.test" in out

    def test_list_empty_config_prints_empty_object(
        self, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", tmp_path / "missing.json")
        monkeypatch.setattr("sys.argv", ["dagnam", "config", "list"])

        main()

        assert capsys.readouterr().out.strip() == "{}"

    def test_get_single_value(
        self, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text('{"api_url": "https://e.test"}', encoding="utf-8")
        monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
        monkeypatch.setattr("sys.argv", ["dagnam", "config", "get", "api_url"])

        main()

        assert capsys.readouterr().out.strip() == "https://e.test"

    def test_get_masks_api_key(
        self, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text('{"api_key": "sk_abcdefghijklmnop"}', encoding="utf-8")
        monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
        monkeypatch.setattr("sys.argv", ["dagnam", "config", "get", "api_key"])

        main()

        out = capsys.readouterr().out
        assert "sk_abc...mnop" in out
        assert "sk_abcdefghijklmnop" not in out

    def test_get_missing_key_exits_1(
        self, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text('{"api_url": "https://e.test"}', encoding="utf-8")
        monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
        monkeypatch.setattr("sys.argv", ["dagnam", "config", "get", "api_key"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        assert "api_key" in capsys.readouterr().err

    def test_get_supported_unset_key_is_informational(
        self, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", tmp_path / "missing.json")
        monkeypatch.setattr("sys.argv", ["dagnam", "config", "get", "training_metrics_path"])

        main()

        assert capsys.readouterr().out == "training_metrics_path is not configured\n"

    def test_set_training_metrics_path_preserves_existing_config(
        self, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(
            '{"api_key": "sk_abcdefghijklmnop", "api_url": "https://e.test"}',
            encoding="utf-8",
        )
        monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
        monkeypatch.setattr(
            "sys.argv",
            ["dagnam", "config", "set", "training_metrics_path", "./dagnam_metrics.jsonl"],
        )

        main()

        data = json.loads(config_file.read_text(encoding="utf-8"))
        assert data["api_key"] == "sk_abcdefghijklmnop"
        assert data["api_url"] == "https://e.test"
        assert data["training_metrics_path"] == "./dagnam_metrics.jsonl"
        assert "training_metrics_path" in capsys.readouterr().out

    def test_unset_training_metrics_path_preserves_credentials(
        self, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "api_key": "sk_abcdefghijklmnop",
                    "training_metrics_path": "./dagnam_metrics.jsonl",
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
        monkeypatch.setattr("sys.argv", ["dagnam", "config", "unset", "training_metrics_path"])

        main()

        data = json.loads(config_file.read_text(encoding="utf-8"))
        assert data == {"api_key": "sk_abcdefghijklmnop"}
        assert "training_metrics_path" in capsys.readouterr().out

    def test_set_rejects_unsupported_config_key(
        self, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", tmp_path / "config.json")
        monkeypatch.setattr("sys.argv", ["dagnam", "config", "set", "api_key", "sk_nope"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        assert "Unsupported config key" in capsys.readouterr().err
