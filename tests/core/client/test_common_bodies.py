"""A response body is data from the network: odd shapes must map to a typed error, never a crash.

Bodies here are built in the test: thousands of nested brackets (``json`` raises
``RecursionError`` for them, which is not a ``ValueError``) and a multi-megabyte message.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import requests

from dagnam._core.client import common
from dagnam._core.client.audit import raise_for_audit
from dagnam._core.exceptions import (
    APIError,
    ModelError,
    QuotaExceededError,
    ResponseError,
    TeardownInProgressError,
)

DEEP = "[" * 200_000 + "]" * 200_000
HUGE = "x" * 5_000_000


def _response(status: int, body: str) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status
    resp._content = body.encode()
    resp.encoding = "utf-8"
    return resp


@pytest.mark.parametrize(
    "decode", [common.response_json_value, common.response_json_object, common.response_json_array]
)
def test_a_deeply_nested_success_body_is_a_malformed_response(
    decode: Callable[[requests.Response], object],
) -> None:
    with pytest.raises(ResponseError, match="malformed response body"):
        decode(_response(200, DEEP))


@pytest.mark.parametrize(
    ("status", "raiser", "error"),
    [
        (402, common.raise_for_generic, QuotaExceededError),
        (403, common.raise_for_generic, APIError),
        (409, common.raise_for_generic, APIError),
        (409, common.raise_for_purge, ModelError),
        (409, raise_for_audit, APIError),
        (503, raise_for_audit, APIError),
    ],
)
def test_a_deeply_nested_error_body_is_an_unparseable_body(
    status: int, raiser: Callable[[requests.Response], None], error: type[Exception]
) -> None:
    with pytest.raises(error) as exc:
        raiser(_response(status, DEEP))

    assert not isinstance(exc.value, TeardownInProgressError)


def test_the_words_of_a_teardown_wait_are_capped() -> None:
    body = f'{{"detail": "{HUGE}", "error": "teardown_in_progress"}}'

    with pytest.raises(TeardownInProgressError) as exc:
        raise_for_audit(_response(409, body))

    assert len(exc.value.message) < 3000
    assert "truncated" in exc.value.message
