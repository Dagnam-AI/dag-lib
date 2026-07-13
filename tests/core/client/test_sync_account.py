"""Wire-level coverage for the sync account / usage client mixin.

Covers the shared ``_account_get`` transport helper (a thin delegator to
``_account_write("GET", path)``: connection + timeout wrapping, empty-body and
non-JSON fallbacks) and the three public read methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import requests

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError, ResponseError

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

API = "https://api.test"

ENTITLEMENTS = f"{API}/api/v1/users/me/entitlements"
QUOTA = f"{API}/api/v1/datasets/storage/quota"
USAGE = f"{API}/api/v1/users/me/api-keys/key1/usage"


def test_get_entitlements(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(ENTITLEMENTS, json={"plan": "pro", "limits": {}})
    assert client.get_entitlements()["plan"] == "pro"
    assert rmock.last_request.headers["Authorization"] == "Bearer k"


def test_get_storage_quota(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(QUOTA, json={"used_bytes": 10, "limit_bytes": 100})
    assert client.get_storage_quota()["limit_bytes"] == 100


def test_get_api_key_usage(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(USAGE, json={"calls": 5})
    assert client.get_api_key_usage("key1") == {"calls": 5}


def test_api_key_usage_quotes_path_segment(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/users/me/api-keys/a%2Fb/usage", json={"calls": 0})
    client.get_api_key_usage("a/b")
    assert rmock.last_request.path.lower().endswith("/api-keys/a%2fb/usage")


def test_account_empty_body_raises_response_error(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    # _account_get returns None on an empty body; _expect_object then rejects it.
    rmock.get(ENTITLEMENTS, status_code=204, text="")
    with pytest.raises(ResponseError, match="Expected JSON object"):
        client.get_entitlements()


def test_account_non_json_body_raises_response_error(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    # Non-JSON body falls back to resp.text (a str); _expect_object rejects it.
    rmock.get(ENTITLEMENTS, text="not-json", headers={"Content-Type": "text/plain"})
    with pytest.raises(ResponseError, match="Expected JSON object"):
        client.get_entitlements()


def test_account_500_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(ENTITLEMENTS, status_code=500, text="boom")
    client._sleep = lambda _s: None  # 500 is transient on a GET → retried; don't sleep
    with pytest.raises(APIError):
        client.get_entitlements()


def test_account_get_retries_transient(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(ENTITLEMENTS, [{"status_code": 503}, {"status_code": 200, "json": {"plan": "pro"}}])
    client._sleep = lambda _s: None
    client._rng = lambda: 1.0
    assert client.get_entitlements()["plan"] == "pro"
    assert rmock.call_count == 2


def test_account_get_404_not_retried(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(ENTITLEMENTS, status_code=404, text="nope")
    client._sleep = lambda _s: None
    with pytest.raises(APIError):
        client.get_entitlements()
    assert rmock.call_count == 1


def test_account_connectionerror_wrapped(client: DagnamClient, rmock: RequestsMocker) -> None:
    # Transport errors now map centrally in ``_request`` to
    # ``APIError(0, "Request failed: ...")``; the GET path exhausts its retries.
    client._sleep = lambda _s: None
    rmock.get(ENTITLEMENTS, exc=requests.ConnectionError("nope"))
    with pytest.raises(APIError, match="Request failed"):
        client.get_entitlements()


def test_account_timeout_wrapped(client: DagnamClient, rmock: RequestsMocker) -> None:
    client._sleep = lambda _s: None
    rmock.get(QUOTA, exc=requests.Timeout("slow"))
    with pytest.raises(APIError, match="Request failed"):
        client.get_storage_quota()
