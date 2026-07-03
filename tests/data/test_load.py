"""Unit tests for dagnam.data.load metadata validators."""

from __future__ import annotations

import pytest
from tests.typing_helpers import JsonObject, PytestMonkeyPatch

from dagnam.data.load import (
    _optional_meta_str,
    _required_meta_str,
    _resolve_cache_budget,
)


class TestRequiredMetaStr:
    def test_returns_string_value(self) -> None:
        meta: JsonObject = {"checksum": "sha256:abc"}
        assert _required_meta_str(meta, "checksum") == "sha256:abc"

    def test_raises_when_missing(self) -> None:
        meta: JsonObject = {}
        with pytest.raises(ValueError, match="must be a string"):
            _required_meta_str(meta, "checksum")

    def test_raises_when_wrong_type(self) -> None:
        meta: JsonObject = {"checksum": 123}
        with pytest.raises(ValueError, match="must be a string"):
            _required_meta_str(meta, "checksum")


class TestOptionalMetaStr:
    def test_returns_string_value(self) -> None:
        meta: JsonObject = {"download_url": "https://x/y"}
        assert _optional_meta_str(meta, "download_url") == "https://x/y"

    def test_returns_none_when_absent(self) -> None:
        meta: JsonObject = {}
        assert _optional_meta_str(meta, "download_url") is None

    def test_returns_none_when_explicitly_none(self) -> None:
        meta: JsonObject = {"download_url": None}
        assert _optional_meta_str(meta, "download_url") is None

    def test_raises_when_wrong_type(self) -> None:
        meta: JsonObject = {"filename": 42}
        with pytest.raises(ValueError, match="must be a string when provided"):
            _optional_meta_str(meta, "filename")


class TestResolveCacheBudget:
    """The eviction budget comes from a user-editable config value, so a corrupt
    value must be tolerated (fall back to None), never crash a fresh download."""

    def test_int_value_passes_through(self, monkeypatch: PytestMonkeyPatch) -> None:
        monkeypatch.setattr("dagnam.data.load.get_config_value", lambda _k, _d: 2048)
        assert _resolve_cache_budget() == 2048

    def test_string_value_returns_none_without_raising(
        self, monkeypatch: PytestMonkeyPatch
    ) -> None:
        monkeypatch.setattr("dagnam.data.load.get_config_value", lambda _k, _d: "10GB")
        assert _resolve_cache_budget() is None

    def test_none_value_returns_none(self, monkeypatch: PytestMonkeyPatch) -> None:
        monkeypatch.setattr("dagnam.data.load.get_config_value", lambda _k, _d: None)
        assert _resolve_cache_budget() is None

    def test_bool_value_returns_none(self, monkeypatch: PytestMonkeyPatch) -> None:
        monkeypatch.setattr("dagnam.data.load.get_config_value", lambda _k, _d: True)
        assert _resolve_cache_budget() is None
