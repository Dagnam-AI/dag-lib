"""Wire-level coverage for the sync inference streaming client methods."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import requests as requests_lib

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError, AuthError, DeploymentNotFoundError

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


def test_open_inference_stream_carries_token_and_input_in_query(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/inference/dep1/stream-access-token", json={"token": "stream-t"})
    rmock.get(
        f"{API}/api/v1/inference/dep1/predict/stream",
        text="event: complete\ndata: {}\n\n",
        headers={"Content-Type": "text/event-stream"},
    )
    resp = client.open_inference_stream("dep1", {"text": "hi"})
    assert resp.status_code == 200
    qs = rmock.last_request.qs
    assert qs["token"] == ["stream-t"]
    assert json.loads(qs["input"][0]) == {"text": "hi"}
    # Auth rides only in the query token — the API key never appears in the URL
    # params, and no Authorization header is sent on the stream request.
    assert "api_key" not in qs
    assert "Authorization" not in rmock.last_request.headers
    assert rmock.last_request.headers["Accept"] == "text/event-stream"


def test_open_inference_stream_404_maps_not_found(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/inference/missing/stream-access-token", json={"token": "t"})
    rmock.get(f"{API}/api/v1/inference/missing/predict/stream", status_code=404)
    with pytest.raises(DeploymentNotFoundError):
        client.open_inference_stream("missing", {"x": 1})


def test_open_inference_stream_connection_error_wraps_apierror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/inference/dep1/stream-access-token", json={"token": "t"})
    rmock.get(
        f"{API}/api/v1/inference/dep1/predict/stream",
        exc=requests_lib.ConnectionError("down"),
    )
    with pytest.raises(APIError) as exc_info:
        client.open_inference_stream("dep1", {"x": 1})
    assert exc_info.value.status_code == 0


def test_open_inference_stream_timeout_wraps_apierror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/inference/dep1/stream-access-token", json={"token": "t"})
    rmock.get(
        f"{API}/api/v1/inference/dep1/predict/stream",
        exc=requests_lib.Timeout("slow"),
    )
    with pytest.raises(APIError) as exc_info:
        client.open_inference_stream("dep1", {"x": 1})
    assert exc_info.value.status_code == 0
