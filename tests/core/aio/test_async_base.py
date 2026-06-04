"""Async client base helpers."""

from __future__ import annotations

import httpx
import pytest
import respx

from dagnam._core.aio.base import (
    BaseAsyncDagnamClient,
    _sanitize_filename,
    parse_content_disposition_filename,
    raise_for_job_response,
)
from dagnam._core.exceptions import (
    APIError,
    AuthError,
    TrainingJobNotFoundError,
)

API = "https://api.test"

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------- base helpers


def testparse_content_disposition_filename_quoted() -> None:
    assert parse_content_disposition_filename('attachment; filename="x.csv"') == "x.csv"


def testparse_content_disposition_filename_unquoted() -> None:
    assert parse_content_disposition_filename("attachment; filename=x.csv") == "x.csv"


def testparse_content_disposition_filename_default() -> None:
    assert parse_content_disposition_filename(None) == "data"
    assert parse_content_disposition_filename("inline") == "data"


def test_sanitize_filename_rejects_unsafe() -> None:
    for bad in ("../x", "C:foo", "", ".", "..", "CON.txt"):
        with pytest.raises(ValueError):
            _sanitize_filename(bad)


async def test_async_base_has_no_inference_headers_method() -> None:
    c = BaseAsyncDagnamClient("https://x", "sk_key")
    try:
        assert not hasattr(c, "_inference_headers")
        assert c._headers() == {"Authorization": "Bearer sk_key"}
    finally:
        await c._client.aclose()


def testraise_for_job_response_2xx_returns() -> None:
    r = httpx.Response(200)
    raise_for_job_response(r, "job1")


def testraise_for_job_response_401() -> None:
    with pytest.raises(AuthError):
        raise_for_job_response(httpx.Response(401), "job1")


def testraise_for_job_response_404() -> None:
    with pytest.raises(TrainingJobNotFoundError):
        raise_for_job_response(httpx.Response(404), "job1")


def testraise_for_job_response_500() -> None:
    with pytest.raises(APIError):
        raise_for_job_response(httpx.Response(500, text="boom"), "job1")


async def test_base_request_connectionerror_wraps() -> None:
    base = BaseAsyncDagnamClient(API, "k")
    try:
        with respx.mock(base_url=API) as r:
            r.get("/x").mock(side_effect=httpx.ConnectError("nope"))
            with pytest.raises(APIError, match="Connection failed"):
                await base._request("GET", "/x")
    finally:
        await base._client.aclose()


async def test_base_request_timeout_wraps() -> None:
    base = BaseAsyncDagnamClient(API, "k")
    try:
        with respx.mock(base_url=API) as r:
            r.get("/x").mock(side_effect=httpx.ReadTimeout("slow"))
            with pytest.raises(APIError, match="Request timed out"):
                await base._request("GET", "/x")
    finally:
        await base._client.aclose()
