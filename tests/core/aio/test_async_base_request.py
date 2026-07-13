"""Retry wiring for the async ``BaseAsyncDagnamClient._request`` (opt-in via raise_for)."""

from __future__ import annotations

import httpx
import pytest
import respx

from dagnam._core.aio.base import BaseAsyncDagnamClient
from dagnam._core.exceptions import APIError

API = "https://api.test"

pytestmark = pytest.mark.anyio


def _raise_for(resp: httpx.Response) -> None:
    if resp.status_code >= 400:
        raise APIError(resp.status_code, "err")


async def _no_sleep(_d: float) -> None: ...


async def test_async_request_retries_transient() -> None:
    base = BaseAsyncDagnamClient(API, "k")
    base._async_sleep = _no_sleep
    base._rng = lambda: 1.0
    try:
        with respx.mock(base_url=API) as r:
            r.get("/api/v1/ping").mock(
                side_effect=[
                    httpx.Response(503, json={}),
                    httpx.Response(503, json={}),
                    httpx.Response(200, json={"ok": True}),
                ]
            )
            resp = await base._request("GET", "/api/v1/ping", raise_for=_raise_for)
        assert resp.status_code == 200
    finally:
        await base._client.aclose()


async def test_async_request_no_raise_for_is_backward_compatible() -> None:
    base = BaseAsyncDagnamClient(API, "k")
    base._async_sleep = _no_sleep
    base._rng = lambda: 1.0
    try:
        with respx.mock(base_url=API) as r:
            route = r.get("/api/v1/ping").mock(return_value=httpx.Response(503, json={}))
            resp = await base._request("GET", "/api/v1/ping")  # no raise_for -> no retry
        assert resp.status_code == 503
        assert route.call_count == 1
    finally:
        await base._client.aclose()


async def test_async_request_404_not_retried() -> None:
    base = BaseAsyncDagnamClient(API, "k")
    base._async_sleep = _no_sleep
    base._rng = lambda: 1.0
    try:
        with respx.mock(base_url=API) as r:
            route = r.get("/x").mock(return_value=httpx.Response(404, json={}))
            with pytest.raises(APIError):
                await base._request("GET", "/x", raise_for=_raise_for)
        assert route.call_count == 1
    finally:
        await base._client.aclose()


async def test_async_request_post_not_retried_without_idempotency() -> None:
    """POST is non-idempotent: a transient status raises immediately, no retry."""
    base = BaseAsyncDagnamClient(API, "k")
    base._async_sleep = _no_sleep
    base._rng = lambda: 1.0
    try:
        with respx.mock(base_url=API) as r:
            route = r.post("/x").mock(return_value=httpx.Response(503, json={}))
            with pytest.raises(APIError):
                await base._request("POST", "/x", raise_for=_raise_for)
        assert route.call_count == 1
    finally:
        await base._client.aclose()


async def test_async_request_post_retries_with_idempotency_key() -> None:
    """A POST carrying an idempotency key becomes retryable and stamps the header."""
    base = BaseAsyncDagnamClient(API, "k")
    base._async_sleep = _no_sleep
    base._rng = lambda: 1.0
    seen_keys: list[str | None] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen_keys.append(request.headers.get("Idempotency-Key"))
        if len(seen_keys) < 2:
            return httpx.Response(503, json={})
        return httpx.Response(200, json={"ok": True})

    try:
        with respx.mock(base_url=API) as r:
            r.post("/x").mock(side_effect=_handler)
            resp = await base._request(
                "POST", "/x", raise_for=_raise_for, idempotency_key="fixed-key"
            )
        assert resp.status_code == 200
        assert seen_keys == ["fixed-key", "fixed-key"]  # same key reused across the retry
    finally:
        await base._client.aclose()


async def test_async_request_idempotent_flag_mints_key() -> None:
    """idempotent=True with no explicit key mints a uuid4 and stamps it."""
    base = BaseAsyncDagnamClient(API, "k")
    base._async_sleep = _no_sleep
    base._rng = lambda: 1.0
    seen_keys: list[str | None] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen_keys.append(request.headers.get("Idempotency-Key"))
        return httpx.Response(200, json={"ok": True})

    try:
        with respx.mock(base_url=API) as r:
            r.post("/x").mock(side_effect=_handler)
            await base._request("POST", "/x", raise_for=_raise_for, idempotent=True)
        assert len(seen_keys) == 1
        assert seen_keys[0]  # a non-empty minted key was stamped
    finally:
        await base._client.aclose()


async def test_async_request_retries_on_transport_error() -> None:
    """A raw transport failure (httpx.ReadError, not just 5xx/timeout) also retries.

    httpx.ReadError is a TransportError distinct from ConnectError/TimeoutException;
    monkeypatch the client's own .request so the raise is unambiguous.
    """
    base = BaseAsyncDagnamClient(API, "k")
    base._async_sleep = _no_sleep
    base._rng = lambda: 1.0
    calls = {"n": 0}

    async def _fake_request(*_a: object, **_k: object) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ReadError("connection reset")
        return httpx.Response(200, json={"ok": True})

    base._client.request = _fake_request  # type: ignore[method-assign]
    try:
        resp = await base._request("GET", "/api/v1/ping", raise_for=_raise_for)
        assert resp.status_code == 200
        assert calls["n"] == 2
    finally:
        await base._client.aclose()


async def test_async_request_timeout_maps_to_api_error() -> None:
    """A timeout with no raise_for is non-retryable and maps to APIError."""
    base = BaseAsyncDagnamClient(API, "k")
    try:
        with respx.mock(base_url=API) as r:
            r.get("/x").mock(side_effect=httpx.ReadTimeout("slow"))
            with pytest.raises(APIError, match="Request timed out"):
                await base._request("GET", "/x")
    finally:
        await base._client.aclose()


async def test_async_post_409_with_idempotency_key_retries_into_replay() -> None:
    """Async mirror: a 409 conflict on an idempotent POST retries into the replay."""
    base = BaseAsyncDagnamClient(API, "k")
    base._async_sleep = _no_sleep
    base._rng = lambda: 1.0
    seen_keys: list[str | None] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen_keys.append(request.headers.get("Idempotency-Key"))
        if len(seen_keys) < 2:
            return httpx.Response(409, json={})
        return httpx.Response(201, json={"id": "j1"})

    try:
        with respx.mock(base_url=API) as r:
            r.post("/api/v1/jobs").mock(side_effect=_handler)
            resp = await base._request(
                "POST", "/api/v1/jobs", raise_for=_raise_for, idempotent=True
            )
        assert resp.status_code == 201
        assert len(seen_keys) == 2
        assert seen_keys[0] == seen_keys[1]
        assert seen_keys[0]  # a real (non-empty) key was minted and reused
    finally:
        await base._client.aclose()
