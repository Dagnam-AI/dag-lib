"""Coverage for ``dagnam._core.auth`` and ``dagnam._core.config``."""

from __future__ import annotations
from pathlib import Path
from tests.typing_helpers import PytestMonkeyPatch


import json

import pytest

from dagnam._core import auth as auth_mod, config as config_mod
from dagnam._core.exceptions import AuthError


@pytest.fixture(autouse=True)
def reset_inline_state():
    """Auth module keeps mutable module-level state; reset it around each test."""
    auth_mod.configure(api_key=None, api_url=None)
    yield
    auth_mod.configure(api_key=None, api_url=None)


@pytest.fixture
def isolated_config(monkeypatch: PytestMonkeyPatch, tmp_path: Path):
    """Redirect config to a temp dir and clear env vars."""
    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.delenv("DAGNAM_API_KEY", raising=False)
    monkeypatch.delenv("DAGNAM_API_URL", raising=False)
    return tmp_path


def test_get_api_key_uses_override(isolated_config: Path) -> None:
    assert auth_mod.get_api_key("explicit-key") == "explicit-key"


def test_get_api_key_uses_inline_state(isolated_config: Path) -> None:
    auth_mod.configure(api_key="inline-key")
    assert auth_mod.get_api_key() == "inline-key"


def test_get_api_key_uses_env_var(isolated_config: Path, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "env-key")
    assert auth_mod.get_api_key() == "env-key"


def test_get_api_key_uses_config_file(isolated_config: Path) -> None:
    (isolated_config / "config.json").write_text(json.dumps({"api_key": "file-key"}))
    assert auth_mod.get_api_key() == "file-key"


def test_get_api_key_raises_when_nothing_configured(isolated_config: Path) -> None:
    with pytest.raises(AuthError, match="No API key found"):
        auth_mod.get_api_key()


def test_get_api_url_uses_override(isolated_config: Path) -> None:
    assert auth_mod.get_api_url("https://override") == "https://override"


def test_get_api_url_uses_inline_state(isolated_config: Path) -> None:
    auth_mod.configure(api_url="https://inline")
    assert auth_mod.get_api_url() == "https://inline"


def test_get_api_url_uses_env_var(isolated_config: Path, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setenv("DAGNAM_API_URL", "https://env")
    assert auth_mod.get_api_url() == "https://env"


def test_get_api_url_uses_config_file(isolated_config: Path) -> None:
    (isolated_config / "config.json").write_text(json.dumps({"api_url": "https://file"}))
    assert auth_mod.get_api_url() == "https://file"


def test_get_api_url_falls_back_to_default(isolated_config: Path) -> None:
    assert auth_mod.get_api_url() == "https://api.dagnam.ai"


def test_load_config_missing_file(isolated_config: Path) -> None:
    assert config_mod.load_config() == {}


def test_load_config_returns_empty_on_malformed_json(isolated_config: Path) -> None:
    (isolated_config / "config.json").write_text("not json {")
    assert config_mod.load_config() == {}


def test_load_config_returns_empty_on_oserror(isolated_config: Path, monkeypatch: PytestMonkeyPatch) -> None:
    def _raise(*_args: object, **_kwargs: object) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(config_mod.Path, "read_text", _raise)
    assert config_mod.load_config() == {}


def test_get_config_value_default(isolated_config: Path) -> None:
    assert config_mod.get_config_value("missing", default="fallback") == "fallback"
