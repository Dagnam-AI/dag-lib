"""Coverage for ``dagnam._core.config.save_config``.

``load_config``/``get_config_value`` are already covered by
``tests/core/test_auth.py``; this file focuses on the secure writer added for
the ``dagnam register`` bootstrap (and shared with ``login``/``logout``/
``config set``/``config unset``).
"""

from __future__ import annotations

import json
import logging
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


@pytest.fixture
def insecure_check_armed(isolated_config: Path, monkeypatch: PytestMonkeyPatch) -> Path:
    """A config file in place with the warn-once guard reset and POSIX forced."""
    monkeypatch.setattr(config_mod, "_config_perms_warned", False)
    monkeypatch.setattr(config_mod.sys, "platform", "linux")
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(json.dumps({"api_key": "sk_abc"}), encoding="utf-8")
    return isolated_config


def test_load_config_warns_once_on_world_readable(
    insecure_check_armed: Path, caplog: pytest.LogCaptureFixture
) -> None:
    if os.name != "posix":
        pytest.skip("POSIX-only permission check")
    os.chmod(insecure_check_armed, 0o644)
    with caplog.at_level(logging.WARNING):
        assert config_mod.load_config() == {"api_key": "sk_abc"}
        config_mod.load_config()  # second call must not warn again
    warnings = [r for r in caplog.records if "insecure permissions" in r.getMessage()]
    assert len(warnings) == 1


def test_load_config_silent_on_owner_only_file(
    insecure_check_armed: Path, caplog: pytest.LogCaptureFixture
) -> None:
    if os.name != "posix":
        pytest.skip("POSIX-only permission check")
    os.chmod(insecure_check_armed, 0o600)
    with caplog.at_level(logging.WARNING):
        assert config_mod.load_config() == {"api_key": "sk_abc"}
    assert not [r for r in caplog.records if "insecure permissions" in r.getMessage()]


def test_load_config_warns_on_foreign_owned_file(
    insecure_check_armed: Path,
    monkeypatch: PytestMonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A stat result owned by a different uid, with a tight 0o600 mode: the
    # foreign-owner leg must fire on its own, independent of the mode bits.
    foreign_stat = os.stat_result((0o100600, 0, 0, 1, os.getuid() + 1, 0, 0, 0, 0, 0))
    monkeypatch.setattr(config_mod.os, "stat", lambda _p: foreign_stat)
    with caplog.at_level(logging.WARNING):
        config_mod.load_config()
    warnings = [r for r in caplog.records if "insecure permissions" in r.getMessage()]
    assert len(warnings) == 1
    assert "foreign-owner=True" in warnings[0].getMessage()


def test_load_config_perms_check_tolerates_stat_oserror(
    insecure_check_armed: Path,
    monkeypatch: PytestMonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def _boom(_p: object) -> os.stat_result:
        raise OSError("stat denied")

    monkeypatch.setattr(config_mod.os, "stat", _boom)
    with caplog.at_level(logging.WARNING):
        assert config_mod.load_config() == {"api_key": "sk_abc"}
    assert not [r for r in caplog.records if "insecure permissions" in r.getMessage()]


def test_load_config_perms_check_skipped_on_windows(
    insecure_check_armed: Path,
    monkeypatch: PytestMonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(config_mod.sys, "platform", "win32")

    def _fail_stat(_p: object) -> os.stat_result:
        raise AssertionError("os.stat must not be called on Windows")

    monkeypatch.setattr(config_mod.os, "stat", _fail_stat)
    with caplog.at_level(logging.WARNING):
        assert config_mod.load_config() == {"api_key": "sk_abc"}
    assert not [r for r in caplog.records if "insecure permissions" in r.getMessage()]


def test_save_config_refuses_foreign_owned_dir(
    isolated_config: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    if os.name != "posix":
        pytest.skip("POSIX-only ownership check")
    from dagnam._core.exceptions import DagnamError

    isolated_config.parent.mkdir(parents=True)
    monkeypatch.setattr(config_mod.sys, "platform", "linux")
    real_stat = os.stat

    def _foreign_dir_stat(path: str | os.PathLike[str], **_kw: object) -> os.stat_result:
        st = real_stat(path)
        if Path(path) == config_mod.CONFIG_DIR:
            fields = list(st)
            fields[4] = os.getuid() + 1  # st_uid
            return os.stat_result(fields)
        return st

    monkeypatch.setattr(config_mod.os, "stat", _foreign_dir_stat)
    with pytest.raises(DagnamError, match="not owned by current user"):
        config_mod.save_config({"api_key": "sk"})


def test_save_config_refuses_foreign_owned_existing_file(
    isolated_config: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    if os.name != "posix":
        pytest.skip("POSIX-only ownership check")
    from dagnam._core.exceptions import DagnamError

    isolated_config.parent.mkdir(parents=True)
    isolated_config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(config_mod.sys, "platform", "linux")
    real_stat = os.stat

    def _foreign_file_stat(path: str | os.PathLike[str], **_kw: object) -> os.stat_result:
        st = real_stat(path)
        if Path(path) == config_mod.CONFIG_FILE:
            fields = list(st)
            fields[4] = os.getuid() + 1
            return os.stat_result(fields)
        return st

    monkeypatch.setattr(config_mod.os, "stat", _foreign_file_stat)
    with pytest.raises(DagnamError, match="not owned by current user"):
        config_mod.save_config({"api_key": "sk"})


def test_save_config_tightens_group_world_dir_mode(
    isolated_config: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    if os.name != "posix":
        pytest.skip("POSIX-only permission check")
    isolated_config.parent.mkdir(parents=True)
    # Deliberately insecure starting mode so the guard's tightening is exercised.
    os.chmod(isolated_config.parent, 0o755)  # noqa: S103 (simulating an exposed config dir)
    monkeypatch.setattr(config_mod.sys, "platform", "linux")
    config_mod.save_config({"api_key": "sk"})
    assert (isolated_config.parent.stat().st_mode & 0o077) == 0


def test_save_config_location_guard_skipped_on_windows(
    isolated_config: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setattr(config_mod.sys, "platform", "win32")

    def _fail_stat(*_a: object, **_k: object) -> os.stat_result:
        raise AssertionError("os.stat must not be called on Windows")

    # The guard must short-circuit before any stat on Windows.
    monkeypatch.setattr(config_mod.os, "stat", _fail_stat)
    monkeypatch.setattr(config_mod.os, "chmod", lambda *_a, **_k: None)
    config_mod.save_config({"api_key": "sk"})
    assert json.loads(isolated_config.read_text(encoding="utf-8")) == {"api_key": "sk"}


def test_save_config_skips_chmod_when_dir_already_tight(
    isolated_config: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    if os.name != "posix":
        pytest.skip("POSIX-only permission check")
    isolated_config.parent.mkdir(parents=True)
    os.chmod(isolated_config.parent, 0o700)  # already tight -> chmod branch skipped
    monkeypatch.setattr(config_mod.sys, "platform", "linux")

    real_chmod = os.chmod

    def _guard_chmod(path: str | os.PathLike[str], mode: int) -> None:
        if Path(path) == config_mod.CONFIG_DIR:
            raise AssertionError("dir chmod must not run when it is already 0o700")
        real_chmod(path, mode)  # the file's 0o600 chmod is still allowed

    monkeypatch.setattr(config_mod.os, "chmod", _guard_chmod)
    config_mod.save_config({"api_key": "sk"})
    assert json.loads(isolated_config.read_text(encoding="utf-8")) == {"api_key": "sk"}
