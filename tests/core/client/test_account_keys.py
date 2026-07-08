"""Wire-level coverage for API-key create/list/revoke sync client methods.

Covers ``AccountClientMixin.create_api_key/list_api_keys/revoke_api_key``, plus
the ``raise_for_generic``/``_expect_object``/``_expect_array`` error mapping
these methods share with the rest of the account surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError, AuthError, QuotaExceededError

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

API = "https://api.test"
API_KEYS = f"{API}/api/v1/users/me/api-keys"

CREATED = {
    "id": "key-1",
    "name": "ci-key",
    "key_prefix": "dgk_abcd",
    "permissions": ["read"],
    "usage_count": 0,
    "last_used_at": None,
    "expires_at": None,
    "created_at": "2026-01-01T00:00:00",
    "key": "dgk_abcdEFGH12345678SECRET",
}


# ----------------------------------------------------------------- create_api_key


def test_create_api_key_sends_name_only_when_no_scopes(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(API_KEYS, status_code=201, json=CREATED)
    result = client.create_api_key("ci-key")
    assert result == CREATED
    assert rmock.last_request.json() == {"name": "ci-key"}
    assert rmock.last_request.headers["Authorization"] == "Bearer k"


def test_create_api_key_maps_scopes_to_permissions(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(API_KEYS, status_code=201, json=CREATED)
    client.create_api_key("ci-key", ["read", "write"])
    assert rmock.last_request.json() == {"name": "ci-key", "permissions": ["read", "write"]}


def test_create_api_key_sends_expires_in_days_when_set(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(API_KEYS, status_code=201, json=CREATED)
    client.create_api_key("ci-key", ["read"], expires_in_days=30)
    assert rmock.last_request.json() == {
        "name": "ci-key",
        "permissions": ["read"],
        "expires_in_days": 30,
    }


def test_create_api_key_omits_expires_in_days_when_none(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(API_KEYS, status_code=201, json=CREATED)
    client.create_api_key("ci-key")
    body = rmock.last_request.json()
    assert "expires_in_days" not in body


def test_create_api_key_400_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(API_KEYS, status_code=400, json={"detail": "invalid scope"})
    with pytest.raises(APIError):
        client.create_api_key("ci-key", ["not-a-real-scope"])


def test_create_api_key_401_raises_autherror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(API_KEYS, status_code=401, text="nope")
    with pytest.raises(AuthError):
        client.create_api_key("ci-key")


def test_create_api_key_402_raises_quotaexceedederror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(API_KEYS, status_code=402, json={"message": "Plan limit reached"})
    with pytest.raises(QuotaExceededError):
        client.create_api_key("ci-key")


# ------------------------------------------------------------------- list_api_keys


def test_list_api_keys_returns_array(client: DagnamClient, rmock: RequestsMocker) -> None:
    payload = [{k: v for k, v in CREATED.items() if k != "key"}]
    rmock.get(API_KEYS, json=payload)
    result = client.list_api_keys()
    assert result == payload
    assert rmock.last_request.headers["Authorization"] == "Bearer k"


def test_list_api_keys_401_raises_autherror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(API_KEYS, status_code=401, text="nope")
    with pytest.raises(AuthError):
        client.list_api_keys()


# ------------------------------------------------------------------ revoke_api_key


def test_revoke_api_key_sends_delete(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(f"{API_KEYS}/key-1", status_code=204)
    result = client.revoke_api_key("key-1")
    assert result is None
    assert rmock.last_request.headers["Authorization"] == "Bearer k"


def test_revoke_api_key_quotes_id(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(f"{API_KEYS}/a%2Fb", status_code=204)
    client.revoke_api_key("a/b")


def test_revoke_api_key_401_raises_autherror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(f"{API_KEYS}/key-1", status_code=401, text="nope")
    with pytest.raises(AuthError):
        client.revoke_api_key("key-1")


def test_revoke_api_key_404_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(f"{API_KEYS}/ghost", status_code=404, json={"detail": "API key not found"})
    with pytest.raises(APIError):
        client.revoke_api_key("ghost")
