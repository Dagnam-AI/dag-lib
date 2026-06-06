"""CLI login/logout/whoami/config subcommands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest

from dagnam.cli import login as login_mod

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture


def _bad_key_prompt(_prompt: str) -> str:
    return "bad-key"


def _good_key_prompt(_prompt: str) -> str:
    return "good-key"


def _key_prompt(_prompt: str) -> str:
    return "k"


# ---------------------------------------------------------------- login


def test_web_url_from_api_url_prod() -> None:
    assert login_mod._web_url_from_api_url("https://api.dagnam.ai") == "https://dagnam.ai"


def test_web_url_from_api_url_local() -> None:
    assert login_mod._web_url_from_api_url("http://localhost:8000") == "http://localhost:5173"


def test_web_url_from_api_url_unknown() -> None:
    assert login_mod._web_url_from_api_url("https://corp.internal") == ""


def test_login_prints_help_block(capsys: StrCapture, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr(login_mod, "error", lambda msg: (_ for _ in ()).throw(SystemExit(msg)))

    ns = argparse.Namespace(api_url="http://localhost:8000")
    with pytest.raises(SystemExit):
        login_mod.cmd_login(ns, getpass_func=lambda _p: "sk_will_fail")
    out = capsys.readouterr().out
    assert login_mod.format_ascii_art() in out
    assert "Don't have an API key yet?" in out
    assert "http://localhost:5173" in out


def test_login_apierror_exits(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    config_dir = tmp_path / ".dagnam"
    config_file = config_dir / "config.json"
    monkeypatch.setattr("dagnam._core.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("getpass.getpass", _bad_key_prompt)
    from dagnam._core.exceptions import AuthError

    with mock.patch(
        "dagnam._core.client.DagnamClient.list_datasets",
        side_effect=AuthError("invalid"),
    ):
        with pytest.raises(SystemExit):
            run_cli(["login"])


def test_login_preserves_existing_config(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    config_dir = tmp_path / ".dagnam"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    config_file.write_text(json.dumps({"other": "kept"}))
    monkeypatch.setattr("dagnam._core.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("getpass.getpass", _good_key_prompt)
    with mock.patch("dagnam._core.client.DagnamClient.list_datasets", return_value=[]):
        run_cli(["login"])
    data = json.loads(config_file.read_text())
    assert data["api_key"] == "good-key"
    assert data["other"] == "kept"


def test_login_persists_training_metrics_path(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    config_dir = tmp_path / ".dagnam"
    config_file = config_dir / "config.json"
    metrics_path = tmp_path / "metrics" / "events.jsonl"
    monkeypatch.setattr("dagnam._core.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("getpass.getpass", _good_key_prompt)
    with mock.patch("dagnam._core.client.DagnamClient.list_datasets", return_value=[]):
        run_cli(["login", "--training-metrics-path", str(metrics_path)])
    data = json.loads(config_file.read_text())
    assert data["api_key"] == "good-key"
    assert data["training_metrics_path"] == str(metrics_path)


def test_login_uses_default_training_metrics_path_when_non_interactive(
    run_cli: CliRunner,
    tmp_path: Path,
    monkeypatch: PytestMonkeyPatch,
    capsys: StrCapture,
) -> None:
    config_dir = tmp_path / ".dagnam"
    config_file = config_dir / "config.json"
    monkeypatch.setattr("dagnam._core.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("getpass.getpass", _good_key_prompt)
    with mock.patch("dagnam._core.client.DagnamClient.list_datasets", return_value=[]):
        run_cli(["login"])
    data = json.loads(config_file.read_text(encoding="utf-8"))
    metrics_path = Path(data["training_metrics_path"])
    assert metrics_path.name == "dagnam_metrics.jsonl"
    assert metrics_path.parent.name == "training-metrics"
    assert "Training metrics path:" in capsys.readouterr().out


def test_login_with_custom_api_url(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    config_dir = tmp_path / ".dagnam"
    config_file = config_dir / "config.json"
    monkeypatch.setattr("dagnam._core.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("getpass.getpass", _key_prompt)
    with mock.patch("dagnam._core.client.DagnamClient.list_datasets", return_value=[]):
        run_cli(["login", "--api-url", "https://custom"])
    data = json.loads(config_file.read_text())
    assert data["api_url"] == "https://custom"


def test_login_corrupt_existing_config_starts_fresh(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    config_dir = tmp_path / ".dagnam"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    config_file.write_text("not json {{{")
    monkeypatch.setattr("dagnam._core.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("getpass.getpass", _key_prompt)
    with mock.patch("dagnam._core.client.DagnamClient.list_datasets", return_value=[]):
        run_cli(["login"])
    data = json.loads(config_file.read_text())
    assert data["api_key"] == "k"
    assert Path(data["training_metrics_path"]).name == "dagnam_metrics.jsonl"


def test_login_prompts_for_metrics_path_when_interactive(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    config_dir = tmp_path / ".dagnam"
    config_file = config_dir / "config.json"
    monkeypatch.setattr("dagnam._core.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)

    class _Tty:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stdin", _Tty())
    entered_path = str(tmp_path / "chosen" / "metrics.jsonl")
    ns = argparse.Namespace(api_url="https://custom", training_metrics_path=None)
    with mock.patch("dagnam._core.client.DagnamClient.list_datasets", return_value=[]):
        login_mod.cmd_login(
            ns,
            getpass_func=lambda _p: "good-key",
            input_func=lambda _p: entered_path,
        )
    data = json.loads(config_file.read_text(encoding="utf-8"))
    assert data["training_metrics_path"] == entered_path


def test_login_interactive_empty_input_uses_default(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    config_dir = tmp_path / ".dagnam"
    config_file = config_dir / "config.json"
    monkeypatch.setattr("dagnam._core.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)

    class _Tty:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stdin", _Tty())
    ns = argparse.Namespace(api_url="https://custom", training_metrics_path=None)
    with mock.patch("dagnam._core.client.DagnamClient.list_datasets", return_value=[]):
        login_mod.cmd_login(
            ns,
            getpass_func=lambda _p: "good-key",
            input_func=lambda _p: "   ",
        )
    data = json.loads(config_file.read_text(encoding="utf-8"))
    assert Path(data["training_metrics_path"]).name == "dagnam_metrics.jsonl"


def test_login_chmods_config_on_posix(tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    """On POSIX the saved config file is chmod'd to 0o600."""
    config_dir = tmp_path / ".dagnam"
    config_file = config_dir / "config.json"
    monkeypatch.setattr("dagnam._core.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("getpass.getpass", _key_prompt)
    monkeypatch.setattr(login_mod.sys, "platform", "linux")
    monkeypatch.setattr(login_mod, "_lock_down_config_path", lambda *_a: None)

    chmod_calls: list[tuple[object, int]] = []
    real_chmod = login_mod.os.chmod

    def _record_chmod(path: str | os.PathLike[str], mode: int) -> None:
        chmod_calls.append((path, mode))
        try:
            real_chmod(path, mode)
        except (OSError, NotImplementedError):
            pass

    monkeypatch.setattr(login_mod.os, "chmod", _record_chmod)
    with mock.patch("dagnam._core.client.DagnamClient.list_datasets", return_value=[]):
        login_mod.cmd_login(argparse.Namespace(api_url="https://custom"))

    assert any(mode == 0o600 for _path, mode in chmod_calls)


def test_login_ignores_chmod_oserror_on_posix(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    """A failing chmod on POSIX must not crash login."""
    config_dir = tmp_path / ".dagnam"
    config_file = config_dir / "config.json"
    monkeypatch.setattr("dagnam._core.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("getpass.getpass", _key_prompt)
    monkeypatch.setattr(login_mod.sys, "platform", "linux")
    monkeypatch.setattr(login_mod, "_lock_down_config_path", lambda *_a: None)

    def _boom(_path: object, _mode: int) -> None:
        raise OSError("chmod denied")

    monkeypatch.setattr(login_mod.os, "chmod", _boom)
    with mock.patch("dagnam._core.client.DagnamClient.list_datasets", return_value=[]):
        login_mod.cmd_login(argparse.Namespace(api_url="https://custom"))

    assert json.loads(config_file.read_text(encoding="utf-8"))["api_key"] == "k"
