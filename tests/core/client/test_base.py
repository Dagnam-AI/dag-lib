"""Coverage for ``dagnam._core.client.base`` shared helpers."""

from __future__ import annotations

import pytest

from dagnam._core.client.base import BaseDagnamClient
from dagnam._core.exceptions import ResponseError


def test_expect_object_raises_response_error_on_non_dict() -> None:
    with pytest.raises(ResponseError):
        BaseDagnamClient._expect_object([1, 2, 3])


def test_expect_object_passes_dict_through() -> None:
    assert BaseDagnamClient._expect_object({"k": "v"}) == {"k": "v"}
