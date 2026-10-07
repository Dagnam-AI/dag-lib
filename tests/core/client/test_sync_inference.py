"""Wire-level coverage for the sync inference streaming client methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import requests as requests_lib

from dagnam._core.client import DagnamClient
from dagnam._core.client.base import DEFAULT_PREDICT_TIMEOUT, DEFAULT_TIMEOUT
from dagnam._core.exceptions import (
    AccountSuspendedError,
    APIError,
    AuthError,
    DeploymentNotFoundError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

API = "https://api.test"


def test_mint_inference_stream_token_posts_with_bearer_header(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/inference/dep1/stream-access-token", json={"token": "stream-t"})
    assert client.mint_inference_stream_token("dep1") == "stream-t"
    assert rmock.last_request.headers["Authorization"] == "Bearer k"


def test_mint_inference_stream_token_401_maps_auth_error(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/inference/dep1/stream-access-token", status_code=401)
    with pytest.raises(AuthError):
        client.mint_inference_stream_token("dep1")


def test_mint_inference_stream_token_connection_error_wraps_apierror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(
        f"{API}/api/v1/inference/dep1/stream-access-token",
        exc=requests_lib.ConnectionError("down"),
    )
    with pytest.raises(APIError) as exc_info:
        client.mint_inference_stream_token("dep1")
    assert exc_info.value.status_code == 0


def test_mint_inference_stream_token_timeout_wraps_apierror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(
        f"{API}/api/v1/inference/dep1/stream-access-token",
        exc=requests_lib.Timeout("slow"),
    )
    with pytest.raises(APIError) as exc_info:
        client.mint_inference_stream_token("dep1")
    assert exc_info.value.status_code == 0


SESSION_URL = f"{API}/api/v1/inference/dep1/predict/stream/session"
STREAM_URL = f"{API}/api/v1/inference/dep1/predict/stream/sess-1"
SESSION = {"session_id": "sess-1", "token": "stream-t", "expires_in": 300}


def test_open_inference_stream_posts_the_input_then_streams_the_session(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(SESSION_URL, json=SESSION)
    rmock.get(
        STREAM_URL,
        text="event: complete\ndata: {}\n\n",
        headers={"Content-Type": "text/event-stream"},
    )
    resp = client.open_inference_stream("dep1", {"text": "hi"})
    assert resp.status_code == 200
    session_req, stream_req = rmock.request_history
    # The input goes in a header-authenticated POST body, never in a URL.
    assert session_req.json() == {"input": {"text": "hi"}}
    assert session_req.headers["Authorization"] == "Bearer k"
    # The stream URL carries only the session's short-lived token: no input, no API key.
    assert stream_req.qs == {"token": ["stream-t"]}
    assert "Authorization" not in stream_req.headers
    assert stream_req.headers["Accept"] == "text/event-stream"


def test_open_inference_stream_session_404_maps_not_found(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/inference/missing/predict/stream/session", status_code=404)
    with pytest.raises(DeploymentNotFoundError):
        client.open_inference_stream("missing", {"x": 1})


def test_open_inference_stream_expired_session_404_maps_not_found(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(SESSION_URL, json=SESSION)
    rmock.get(STREAM_URL, status_code=404)
    with pytest.raises(DeploymentNotFoundError):
        client.open_inference_stream("dep1", {"x": 1})


def test_open_inference_stream_connection_error_wraps_apierror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(SESSION_URL, json=SESSION)
    rmock.get(STREAM_URL, exc=requests_lib.ConnectionError("down"))
    with pytest.raises(APIError) as exc_info:
        client.open_inference_stream("dep1", {"x": 1})
    assert exc_info.value.status_code == 0


def test_open_inference_stream_timeout_wraps_apierror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(SESSION_URL, json=SESSION)
    rmock.get(STREAM_URL, exc=requests_lib.Timeout("slow"))
    with pytest.raises(APIError) as exc_info:
        client.open_inference_stream("dep1", {"x": 1})
    assert exc_info.value.status_code == 0


# ------------------------------------------- shared account-status 403 mapping
# These rejections come from server-side middleware, so they can land on any
# route. The sync client must map them exactly like the async mirror does —
# see the twin tests in tests/core/aio/test_async_inference.py.


def test_predict_suspended_403_raises_account_suspended(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(
        f"{API}/api/v1/inference/dep1/predict",
        status_code=403,
        json={"detail": {"error": "account_suspended", "message": "Account suspended."}},
    )
    with pytest.raises(AccountSuspendedError, match=r"Account suspended\."):
        client.predict("dep1", {"x": 1})


def test_predict_blocked_ip_403_raises_auth_error(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(
        f"{API}/api/v1/inference/dep1/predict",
        status_code=403,
        json={"detail": {"error": "blocked_ip", "message": "IP not permitted."}},
    )
    with pytest.raises(AuthError, match=r"IP not permitted\."):
        client.predict("dep1", {"x": 1})


# ------------------------------------------------ cold-start timeout defaults


def test_predict_waits_the_cold_start_budget_by_default(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/inference/dep1/predict", json={"y": 1})
    client.predict("dep1", {"x": 1})
    assert DEFAULT_PREDICT_TIMEOUT == 600
    assert rmock.last_request.timeout == (DEFAULT_TIMEOUT, DEFAULT_PREDICT_TIMEOUT)


def test_predict_batch_waits_the_cold_start_budget_by_default(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/inference/dep1/predict/batch", json=[])
    client.predict_batch("dep1", [{"x": 1}])
    assert rmock.last_request.timeout == (DEFAULT_TIMEOUT, DEFAULT_PREDICT_TIMEOUT)


def test_predict_timeout_can_be_lowered(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/inference/dep1/predict", json={"y": 1})
    client.predict("dep1", {"x": 1}, timeout=5)
    assert rmock.last_request.timeout == (5, 5)


def test_predict_connect_stays_short_when_the_read_timeout_is_raised(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/inference/dep1/predict/batch", json=[])
    client.predict_batch("dep1", [], timeout=900)
    assert rmock.last_request.timeout == (DEFAULT_TIMEOUT, 900)


def test_schema_keeps_the_short_default_timeout(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.get(f"{API}/api/v1/inference/dep1/schema", json={})
    client.schema("dep1")
    assert rmock.last_request.timeout == DEFAULT_TIMEOUT
