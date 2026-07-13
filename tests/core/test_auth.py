"""Coverage for ``dagnam._core.auth`` and ``dagnam._core.config``."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from tests.typing_helpers import PytestMonkeyPatch

from dagnam._core import auth as auth_mod, config as config_mod
from dagnam._core.exceptions import AuthError


@pytest.fixture(autouse=True)
def reset_inline_state(monkeypatch: PytestMonkeyPatch):
    """Auth module keeps mutable module-level state; reset it around each test."""
    auth_mod.configure(api_key=None, api_url=None)
    monkeypatch.setattr(auth_mod, "_api_url_warned", False)
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


def test_load_config_returns_empty_on_oserror(
    isolated_config: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    def _raise(*_args: object, **_kwargs: object) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(config_mod.Path, "read_text", _raise)
    assert config_mod.load_config() == {}


def test_get_config_value_default(isolated_config: Path) -> None:
    assert config_mod.get_config_value("missing", default="fallback") == "fallback"


def _credential_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "credentials will be sent" in r.getMessage()]


def test_get_api_url_warns_once_on_cleartext_env_url(
    isolated_config: Path,
    monkeypatch: PytestMonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("DAGNAM_API_URL", "http://evil.example")
    with caplog.at_level(logging.WARNING):
        assert auth_mod.get_api_url() == "http://evil.example"
        auth_mod.get_api_url()  # second call: no duplicate warning
    warnings = _credential_warnings(caplog)
    assert len(warnings) == 1
    assert "cleartext http" in warnings[0]


def test_get_api_url_warns_on_non_default_https_config_host(
    isolated_config: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (isolated_config / "config.json").write_text(json.dumps({"api_url": "https://other.example"}))
    with caplog.at_level(logging.WARNING):
        assert auth_mod.get_api_url() == "https://other.example"
    warnings = _credential_warnings(caplog)
    assert len(warnings) == 1
    assert "cleartext" not in warnings[0]


def test_get_api_url_default_host_via_env_stays_silent(
    isolated_config: Path,
    monkeypatch: PytestMonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Checked (env origin) but benign: default host over https — no warning.
    monkeypatch.setenv("DAGNAM_API_URL", "https://api.dagnam.ai")
    with caplog.at_level(logging.WARNING):
        assert auth_mod.get_api_url() == "https://api.dagnam.ai"
    assert not _credential_warnings(caplog)


def test_get_api_url_override_and_inline_stay_silent(
    isolated_config: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Explicit override and configure(api_url=…) are deliberate programmatic
    # choices — never warned, even for a non-default cleartext host.
    with caplog.at_level(logging.WARNING):
        assert auth_mod.get_api_url("http://staging.internal") == "http://staging.internal"
        auth_mod.configure(api_url="http://staging.internal")
        assert auth_mod.get_api_url() == "http://staging.internal"
    assert not _credential_warnings(caplog)


def test_get_api_url_default_fallback_stays_silent(
    isolated_config: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        assert auth_mod.get_api_url() == "https://api.dagnam.ai"
    assert not _credential_warnings(caplog)


def test_warn_helper_allows_cleartext_localhost(
    isolated_config: Path,
    monkeypatch: PytestMonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # http://localhost is non-default (warned as such) but NOT flagged cleartext.
    monkeypatch.setenv("DAGNAM_API_URL", "http://localhost:8000")
    with caplog.at_level(logging.WARNING):
        assert auth_mod.get_api_url() == "http://localhost:8000"
    warnings = _credential_warnings(caplog)
    assert len(warnings) == 1
    assert "cleartext" not in warnings[0]
