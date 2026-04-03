"""Unit tests for dagnam.cli module."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from dagnam.cli import _build_parser, _human_size, main


class TestMainNoArgs:
    """Running with no arguments should print help and exit."""

    def test_exits_with_code_2(self):
        with mock.patch("sys.argv", ["dagnam"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2


class TestHumanSize:
    def test_bytes(self):
        assert _human_size(500) == "500.0 B"

    def test_kilobytes(self):
        assert _human_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert _human_size(5 * 1024 * 1024) == "5.0 MB"

    def test_zero(self):
        assert _human_size(0) == "0.0 B"


class TestLogin:
    def test_creates_config_file(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".dagnam"
        config_file = config_dir / "config.json"

        monkeypatch.setattr("dagnam.cli.getpass.getpass", lambda _: "test-key-123")
        monkeypatch.setattr("dagnam.config.CONFIG_DIR", config_dir)
        monkeypatch.setattr("dagnam.config.CONFIG_FILE", config_file)
        monkeypatch.setattr("sys.argv", ["dagnam", "login"])

        # Mock the client so it doesn't actually hit the network
        with mock.patch("dagnam.client.DagnamClient.list_datasets", return_value=[]):
            main()

        assert config_file.exists()
        data = json.loads(config_file.read_text())
        assert data["api_key"] == "test-key-123"

    def test_empty_key_exits(self, monkeypatch):
        monkeypatch.setattr("dagnam.cli.getpass.getpass", lambda _: "  ")
        monkeypatch.setattr("sys.argv", ["dagnam", "login"])

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


class TestCacheList:
    def test_empty_cache(self, tmp_path, monkeypatch, capsys):
        empty_dir = tmp_path / "datasets"
        empty_dir.mkdir()
        monkeypatch.setattr("dagnam.cache.DEFAULT_CACHE_DIR", empty_dir)
        monkeypatch.setattr("sys.argv", ["dagnam", "cache", "list"])
        main()
        assert "empty" in capsys.readouterr().out.lower()

    def test_nonexistent_cache_dir(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("dagnam.cache.DEFAULT_CACHE_DIR", tmp_path / "nope")
        monkeypatch.setattr("sys.argv", ["dagnam", "cache", "list"])
        main()
        assert "empty" in capsys.readouterr().out.lower()

    def test_lists_cached_dataset(self, tmp_path, monkeypatch, capsys):
        cache = tmp_path / "datasets"
        ds_dir = cache / "ds-abc"
        ds_dir.mkdir(parents=True)
        (ds_dir / "meta.json").write_text(json.dumps({"name": "Test DS"}))
        (ds_dir / "data.csv").write_text("a,b\n1,2\n")

        monkeypatch.setattr("dagnam.cache.DEFAULT_CACHE_DIR", cache)
        monkeypatch.setattr("sys.argv", ["dagnam", "cache", "list"])
        main()
        out = capsys.readouterr().out
        assert "ds-abc" in out
        assert "Test DS" in out


class TestCacheClear:
    def test_empty_cache(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("dagnam.cache.DEFAULT_CACHE_DIR", tmp_path / "nope")
        monkeypatch.setattr("sys.argv", ["dagnam", "cache", "clear"])
        main()
        assert "already empty" in capsys.readouterr().out.lower()

    def test_clears_cache(self, tmp_path, monkeypatch, capsys):
        cache = tmp_path / "datasets"
        ds_dir = cache / "ds-abc"
        ds_dir.mkdir(parents=True)
        (ds_dir / "data.csv").write_text("a,b\n1,2\n")

        monkeypatch.setattr("dagnam.cache.DEFAULT_CACHE_DIR", cache)
        monkeypatch.setattr("sys.argv", ["dagnam", "cache", "clear"])
        main()
        assert not cache.exists()
        assert "freed" in capsys.readouterr().out.lower()


class TestBuildParser:
    def test_parser_has_subcommands(self):
        parser = _build_parser()
        # Smoke test: parsing known commands shouldn't raise
        args = parser.parse_args(["cache", "clear"])
        assert args.command == "cache"
        assert args.cache_command == "clear"

    def test_dataset_download_requires_id(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["dataset", "download"])
