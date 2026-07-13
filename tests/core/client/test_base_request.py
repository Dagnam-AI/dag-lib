"""Transport-level tests for the centralized retrying ``BaseDagnamClient._request``."""

from __future__ import annotations

import pytest
import requests

from dagnam._core.client.base import BaseDagnamClient
from dagnam._core.client.common import raise_for_generic
from dagnam._core.exceptions import APIError, DatasetNotFoundError


def _client() -> BaseDagnamClient:
    c = BaseDagnamClient("https://api.test", "sk_key")
    c._sleep = lambda _s: None  # no real sleeping in tests
    c._rng = lambda: 1.0
    return c


def test_request_success_sends_auth_header(requests_mock):
    requests_mock.get("https://api.test/api/v1/ping", json={"ok": True})
    c = _client()
    resp = c._request("GET", "/api/v1/ping", raise_for=lambda r: raise_for_generic(r))
    assert resp.json() == {"ok": True}
    assert requests_mock.last_request.headers["Authorization"] == "Bearer sk_key"


def test_request_retries_transient_then_succeeds(requests_mock):
    requests_mock.get(
        "https://api.test/api/v1/ping",
        [
            {"status_code": 503},
            {"status_code": 503},
            {"status_code": 200, "json": {"ok": True}},
        ],
    )
    c = _client()
    resp = c._request("GET", "/api/v1/ping", raise_for=lambda r: raise_for_generic(r))
    assert resp.json() == {"ok": True}
    assert requests_mock.call_count == 3


def test_request_does_not_retry_404_domain_error(requests_mock):
    requests_mock.get("https://api.test/api/v1/datasets/ds1", status_code=404)
    c = _client()

    def raise_for(r: requests.Response) -> None:
        from dagnam._core.client.common import raise_for_dataset

        raise_for_dataset(r, "ds1")

    with pytest.raises(DatasetNotFoundError):
        c._request("GET", "/api/v1/datasets/ds1", raise_for=raise_for)
    assert requests_mock.call_count == 1  # no retry on a domain 404


def test_request_maps_transport_error_to_api_error(requests_mock):
    requests_mock.get("https://api.test/api/v1/ping", exc=requests.ConnectionError("down"))
    c = _client()
    with pytest.raises(APIError) as ei:
        c._request("GET", "/api/v1/ping", raise_for=lambda r: raise_for_generic(r))
    # status 0 sentinel; retried up to the cap since GET is retryable then surfaced
    assert ei.value.status_code == 0


def test_post_without_idempotency_is_not_retried(requests_mock):
    requests_mock.post("https://api.test/api/v1/jobs", status_code=503)
    c = _client()
    with pytest.raises(APIError):
        c._request("POST", "/api/v1/jobs", raise_for=lambda r: raise_for_generic(r), json={})
    assert requests_mock.call_count == 1


def test_post_with_idempotent_sends_key_and_retries(requests_mock):
    requests_mock.post(
        "https://api.test/api/v1/jobs",
        [{"status_code": 503}, {"status_code": 201, "json": {"id": "j1"}}],
    )
    c = _client()
    resp = c._request(
        "POST",
        "/api/v1/jobs",
        raise_for=lambda r: raise_for_generic(r),
        json={},
        idempotent=True,
    )
    assert resp.json() == {"id": "j1"}
    assert requests_mock.call_count == 2
    keys = {req.headers.get("Idempotency-Key") for req in requests_mock.request_history}
    assert len(keys) == 1  # same key on both attempts
    assert next(iter(keys))  # and it is non-empty


def test_absolute_url_used_verbatim(requests_mock):
    requests_mock.get("https://cdn.test/blob", json={"ok": True})
    c = _client()
    resp = c._request("GET", "https://cdn.test/blob", raise_for=lambda r: raise_for_generic(r))
    assert resp.json() == {"ok": True}


def test_extra_headers_are_merged_over_auth(requests_mock):
    requests_mock.get("https://api.test/api/v1/ping", json={"ok": True})
    c = _client()
    c._request(
        "GET",
        "/api/v1/ping",
        raise_for=lambda r: raise_for_generic(r),
        headers={"X-Trace": "abc"},
    )
    sent = requests_mock.last_request.headers
    assert sent["X-Trace"] == "abc"
    assert sent["Authorization"] == "Bearer sk_key"  # base auth header preserved


def test_post_409_with_idempotency_key_retries_into_replay(requests_mock):
    """A 409 from the server-side idempotency middleware means a copy is in
    flight — the scoped conflict-retry resolves it into the replayed 201."""
    requests_mock.post(
        "https://api.test/api/v1/jobs",
        [{"status_code": 409}, {"status_code": 201, "json": {"id": "j1"}}],
    )
    c = _client()
    resp = c._request(
        "POST",
        "/api/v1/jobs",
        raise_for=lambda r: raise_for_generic(r),
        json={},
        idempotent=True,
    )
    assert resp.json() == {"id": "j1"}
    assert requests_mock.call_count == 2
    # Both attempts carried the SAME minted key so the server dedupes.
    keys = {req.headers.get("Idempotency-Key") for req in requests_mock.request_history}
    assert len(keys) == 1
    assert next(iter(keys))
