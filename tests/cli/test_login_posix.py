"""Coverage for the POSIX lock-down paths in cli/login.py.

The lock-down helper short-circuits on Windows, so on this build platform
those branches are unreachable unless we swap in a fake ``os`` namespace
plus ``sys.platform = 'linux'`` for the duration of the test.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import pytest

import dagnam.cli.login as login_mod
from dagnam.cli.login import _lock_down_config_path

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch


def _noop_chmod(_path: Path | str, _mode: int) -> None:
    return None


def _fake_os(
    *,
    stat_fn: Callable[[Path | str], object],
    chmod_fn: Callable[[Path | str, int], None] = _noop_chmod,
    uid: int = 1000,
) -> SimpleNamespace:
    """Build a SimpleNamespace standing in for the ``os`` module inside login.py."""
    return SimpleNamespace(
        getuid=lambda: uid,
        stat=stat_fn,
        chmod=chmod_fn,
        # The post-write atomic-create path needs these constants. Match the real
        # values from posix so getattr() lookups inside login.py still work.
        O_WRONLY=1,
        O_CREAT=64,
        O_TRUNC=512,
        O_NOFOLLOW=131072,
        open=lambda *_a, **_kw: 0,
        fdopen=lambda *_a, **_kw: __import__("io").StringIO(),
    )


@pytest.fixture
def force_linux(monkeypatch: PytestMonkeyPatch) -> SimpleNamespace:
    """Swap login_mod.sys for a stub so pretending to be Linux doesn't leak globally."""
    fake_sys = SimpleNamespace(platform="linux")
    monkeypatch.setattr(login_mod, "sys", fake_sys)
    return fake_sys


def test_lock_down_config_path_returns_early_on_windows(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(login_mod, "sys", SimpleNamespace(platform="win32"))
    # Should be a no-op even with a missing dir.
    _lock_down_config_path(tmp_path / "missing", tmp_path / "missing.json")


def test_lock_down_config_path_chmods_loose_dir(
    force_linux: SimpleNamespace, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    config_dir = tmp_path / ".dagnam"
    config_dir.mkdir()
    chmod_calls: list[tuple[Path | str, int]] = []

    def chmod(path: Path | str, mode: int) -> None:
        chmod_calls.append((path, mode))

    fake = _fake_os(
        stat_fn=lambda _path: SimpleNamespace(st_uid=1000, st_mode=0o777),
        chmod_fn=chmod,
    )
    monkeypatch.setattr(login_mod, "os", fake)
    _lock_down_config_path(config_dir, config_dir / "config.json")
    assert (config_dir, 0o700) in chmod_calls


def test_lock_down_config_path_rejects_foreign_uid(
    force_linux: SimpleNamespace, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    fake = _fake_os(
        stat_fn=lambda _path: SimpleNamespace(st_uid=42, st_mode=0o700),
    )
    monkeypatch.setattr(login_mod, "os", fake)
    with pytest.raises(SystemExit):
        _lock_down_config_path(tmp_path, tmp_path / "config.json")


def test_lock_down_config_path_handles_stat_failure(
    force_linux: SimpleNamespace, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    def _raise(_path: Path | str) -> None:
        raise OSError("perm denied")

    fake = _fake_os(stat_fn=_raise)
    monkeypatch.setattr(login_mod, "os", fake)
    with pytest.raises(SystemExit):
        _lock_down_config_path(tmp_path, tmp_path / "config.json")


def test_lock_down_config_path_chmod_failure_exits(
    force_linux: SimpleNamespace, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    def _bad_chmod(_p: Path | str, _m: int) -> None:
        raise OSError("locked")

    fake = _fake_os(
        stat_fn=lambda _path: SimpleNamespace(st_uid=1000, st_mode=0o777),
        chmod_fn=_bad_chmod,
    )
    monkeypatch.setattr(login_mod, "os", fake)
    with pytest.raises(SystemExit):
        _lock_down_config_path(tmp_path, tmp_path / "config.json")


def test_lock_down_config_path_file_uid_mismatch(
    force_linux: SimpleNamespace, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}")

    def stat(path: Path | str) -> SimpleNamespace:
        # Dir is owned by us; file is foreign.
        if str(path) == str(cfg_file):
            return SimpleNamespace(st_uid=99, st_mode=0o600)
        return SimpleNamespace(st_uid=1000, st_mode=0o700)

    fake = _fake_os(stat_fn=stat)
    monkeypatch.setattr(login_mod, "os", fake)
    with pytest.raises(SystemExit):
        _lock_down_config_path(tmp_path, cfg_file)


def test_lock_down_config_path_file_stat_failure(
    force_linux: SimpleNamespace, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}")

    def stat(path: Path | str) -> SimpleNamespace:
        if str(path) == str(cfg_file):
            raise OSError("file gone")
        return SimpleNamespace(st_uid=1000, st_mode=0o700)

    fake = _fake_os(stat_fn=stat)
    monkeypatch.setattr(login_mod, "os", fake)
    with pytest.raises(SystemExit):
        _lock_down_config_path(tmp_path, cfg_file)


def test_login_chmod_failure_swallowed(
    force_linux: SimpleNamespace, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """The post-write chmod() inside cmd_login is best-effort; OSError mustn't abort."""
    config_dir = tmp_path / ".dagnam"
    config_file = config_dir / "config.json"
    monkeypatch.setattr("dagnam._core.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "k")

    # Track chmod calls and have only the post-write file chmod fail.
    chmod_calls: list[tuple[Path | str, int]] = []

    def chmod(path: Path | str, mode: int) -> None:
        chmod_calls.append((path, mode))
        if str(path) == str(config_file):
            raise OSError("locked")

    # Forward through to real os for the calls login uses besides stat/chmod/uid.
    import os as real_os

    fake = SimpleNamespace(
        getuid=lambda: 1000,
        stat=lambda _path: SimpleNamespace(st_uid=1000, st_mode=0o700),
        chmod=chmod,
        O_WRONLY=real_os.O_WRONLY,
        O_CREAT=real_os.O_CREAT,
        O_TRUNC=real_os.O_TRUNC,
        O_NOFOLLOW=getattr(real_os, "O_NOFOLLOW", 0),
        open=real_os.open,
        fdopen=real_os.fdopen,
    )
    monkeypatch.setattr(login_mod, "os", fake)

    with mock.patch("dagnam._core.client.DagnamClient.list_datasets", return_value=[]):
        monkeypatch.setattr("sys.argv", ["dagnam", "login"])
        from dagnam.cli import main as cli_main

        cli_main()

    assert config_file.exists()
    assert json.loads(config_file.read_text())["api_key"] == "k"


def test_lock_down_config_path_existing_file_matching_uid_passes(
    force_linux: SimpleNamespace, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """Existing file owned by the current user passes the ownership check."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}")

    fake = _fake_os(
        stat_fn=lambda _path: SimpleNamespace(st_uid=1000, st_mode=0o700),
    )
    monkeypatch.setattr(login_mod, "os", fake)
    # Should return without raising (the 56->exit fall-through branch).
    _lock_down_config_path(tmp_path, cfg_file)


def test_login_reraises_when_fdopen_fails(
    force_linux: SimpleNamespace, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """The ``except BaseException: raise`` guard around os.fdopen propagates errors."""
    config_dir = tmp_path / ".dagnam"
    config_file = config_dir / "config.json"
    monkeypatch.setattr("dagnam._core.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "k")
    monkeypatch.setattr(login_mod, "_lock_down_config_path", lambda *_a: None)

    import os as real_os

    def _boom_fdopen(*_a: object, **_kw: object) -> object:
        raise OSError("disk full")

    fake = SimpleNamespace(
        getuid=lambda: 1000,
        stat=lambda _path: SimpleNamespace(st_uid=1000, st_mode=0o700),
        chmod=_noop_chmod,
        O_WRONLY=real_os.O_WRONLY,
        O_CREAT=real_os.O_CREAT,
        O_TRUNC=real_os.O_TRUNC,
        O_NOFOLLOW=getattr(real_os, "O_NOFOLLOW", 0),
        open=real_os.open,
        fdopen=_boom_fdopen,
    )
    monkeypatch.setattr(login_mod, "os", fake)

    with mock.patch("dagnam._core.client.DagnamClient.list_datasets", return_value=[]):
        with pytest.raises(OSError, match="disk full"):
            login_mod.cmd_login(argparse.Namespace(api_url="https://custom"))
