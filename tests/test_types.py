"""Tests for the canonical JSON validators in ``dagnam._types``.

These pure functions are the single source of truth for the
``Expected JSON ...`` contract checks reused across the client mixins, so
covering them here exercises the shared narrowing logic directly.
"""

from __future__ import annotations

import pytest

from dagnam._types import (
    ensure_json_array,
    ensure_json_object,
    ensure_json_value,
    is_json_value,
)


class TestIsJsonValue:
    @pytest.mark.parametrize(
        "value",
        [None, "s", 1, 1.5, True, [1, "a", None], {"k": [1, 2]}],
    )
    def test_accepts_json_compatible(self, value: object) -> None:
        assert is_json_value(value) is True

    @pytest.mark.parametrize(
        "value",
        [object(), {1: "non-str-key"}, [object()], {"k": object()}],
    )
    def test_rejects_non_json(self, value: object) -> None:
        assert is_json_value(value) is False


class TestEnsureJsonValue:
    def test_returns_value_when_compatible(self) -> None:
        assert ensure_json_value({"a": 1}) == {"a": 1}

    def test_raises_on_non_json(self) -> None:
        with pytest.raises(TypeError, match="Expected JSON-compatible value, got object"):
            ensure_json_value(object())


class TestEnsureJsonObject:
    def test_returns_object(self) -> None:
        assert ensure_json_object({"a": 1}) == {"a": 1}

    def test_raises_on_list(self) -> None:
        with pytest.raises(TypeError, match="Expected JSON object, got list"):
            ensure_json_object([1, 2])

    def test_raises_on_non_string_keys(self) -> None:
        with pytest.raises(TypeError, match="Expected JSON object"):
            ensure_json_object({1: "v"})


class TestEnsureJsonArray:
    def test_returns_array(self) -> None:
        assert ensure_json_array([1, "a"]) == [1, "a"]

    def test_raises_on_dict(self) -> None:
        with pytest.raises(TypeError, match="Expected JSON array, got dict"):
            ensure_json_array({"a": 1})

    def test_raises_on_non_json_item(self) -> None:
        with pytest.raises(TypeError, match="Expected JSON array"):
            ensure_json_array([object()])
