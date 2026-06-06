"""Unit tests for dagnam.data.load metadata validators."""

from __future__ import annotations

import pytest
from tests.typing_helpers import JsonObject

from dagnam.data.load import (
    _optional_meta_int,
    _optional_meta_str,
    _required_meta_str,
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


class TestOptionalMetaInt:
    def test_returns_int_value(self) -> None:
        meta: JsonObject = {"max_cache_size": 1024}
        assert _optional_meta_int(meta, "max_cache_size") == 1024

    def test_returns_none_when_absent(self) -> None:
        meta: JsonObject = {}
        assert _optional_meta_int(meta, "max_cache_size") is None

    def test_returns_none_when_explicitly_none(self) -> None:
        meta: JsonObject = {"max_cache_size": None}
        assert _optional_meta_int(meta, "max_cache_size") is None

    def test_raises_on_bool(self) -> None:
        meta: JsonObject = {"max_cache_size": True}
        with pytest.raises(ValueError, match="must be an integer when provided"):
            _optional_meta_int(meta, "max_cache_size")

    def test_raises_on_wrong_type(self) -> None:
        meta: JsonObject = {"max_cache_size": "big"}
        with pytest.raises(ValueError, match="must be an integer when provided"):
            _optional_meta_int(meta, "max_cache_size")
