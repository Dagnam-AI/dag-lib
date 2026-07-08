"""Coverage for ``dagnam._core.config.save_config``.

``load_config``/``get_config_value`` are already covered by
``tests/core/test_auth.py``; this file focuses on the secure writer added for
the ``dagnam register`` bootstrap (and shared with ``login``/``logout``/
``config set``/``config unset``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from tests.typing_helpers import PytestMonkeyPatch

from dagnam._core import config as config_mod


@pytest.fixture
def isolated_config(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> Path:
    """Redirect CONFIG_DIR/CONFIG_FILE to a temp dir, mirroring tests/cli/test_login.py."""
    config_dir = tmp_path / ".dagnam"
    config_file = config_dir / "config.json"
    monkeypatch.setattr(config_mod, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_mod, "CONFIG_FILE", config_file)
    return config_file


def test_save_config_writes_json_and_creates_parent_dir(isolated_config: Path) -> None:
    assert not isolated_config.parent.exists()
    config_mod.save_config({"api_key": "sk_abc", "api_url": "https://e.test"})
    assert isolated_config.parent.exists()
    data = json.loads(isolated_config.read_text(encoding="utf-8"))
    assert data == {"api_key": "sk_abc", "api_url": "https://e.test"}
    assert isolated_config.read_text(encoding="utf-8").endswith("\n")


def test_save_config_overwrites_existing_content(isolated_config: Path) -> None:
    isolated_config.parent.mkdir(parents=True)
    isolated_config.write_text(json.dumps({"stale": "value"}), encoding="utf-8")
    config_mod.save_config({"fresh": "value"})
    assert json.loads(isolated_config.read_text(encoding="utf-8")) == {"fresh": "value"}


def test_save_config_sets_0600_on_posix(isolated_config: Path) -> None:
    if os.name != "posix":
        pytest.skip("POSIX-only permission check")
    config_mod.save_config({"api_key": "sk_abc"})
    assert (isolated_config.stat().st_mode & 0o777) == 0o600


def test_save_config_chmods_on_posix(isolated_config: Path, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr(config_mod.sys, "platform", "linux")
    chmod_calls: list[tuple[object, int]] = []
    real_chmod = config_mod.os.chmod

    def _record_chmod(path: str | os.PathLike[str], mode: int) -> None:
        chmod_calls.append((path, mode))
        try:
            real_chmod(path, mode)
        except (OSError, NotImplementedError):
            pass

    monkeypatch.setattr(config_mod.os, "chmod", _record_chmod)
    config_mod.save_config({"api_key": "sk_abc"})
    assert any(mode == 0o600 for _path, mode in chmod_calls)


def test_save_config_ignores_chmod_oserror_on_posix(
    isolated_config: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setattr(config_mod.sys, "platform", "linux")

    def _boom(_path: object, _mode: int) -> None:
        raise OSError("chmod denied")

    monkeypatch.setattr(config_mod.os, "chmod", _boom)
    config_mod.save_config({"api_key": "sk_abc"})
    assert json.loads(isolated_config.read_text(encoding="utf-8")) == {"api_key": "sk_abc"}


def test_save_config_skips_chmod_on_windows(
    isolated_config: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setattr(config_mod.sys, "platform", "win32")

    def _fail_chmod(_path: object, _mode: int) -> None:
        raise AssertionError("chmod must not be called on Windows")

    monkeypatch.setattr(config_mod.os, "chmod", _fail_chmod)
    config_mod.save_config({"api_key": "sk_abc"})
    assert json.loads(isolated_config.read_text(encoding="utf-8")) == {"api_key": "sk_abc"}
