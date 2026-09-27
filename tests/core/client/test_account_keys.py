"""Wire-level coverage for the sync ``create_api_key`` client method.

``dagnam.account.register`` mints its first key through it with a session
token; the tests also pin the ``raise_for_generic``/``_expect_object`` error
mapping it shares with the rest of the account surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dagnam._core.client import DagnamClient
from dagnam._core.client.account import ApiKeyPlanError
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
    with pytest.raises(QuotaExceededError) as exc_info:
        client.create_api_key("ci-key")
    # No refusal code in the body, so it is not reported as a plan without keys.
    assert not isinstance(exc_info.value, ApiKeyPlanError)


def test_create_api_key_plan_refusal_without_message_raises_apikeyplanerror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(API_KEYS, status_code=403, json={"error": "feature_gated"})
    with pytest.raises(ApiKeyPlanError, match="Plan limit reached"):
        client.create_api_key("ci-key")


def test_create_api_key_non_json_403_maps_generically(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(API_KEYS, status_code=403, text="forbidden")
    with pytest.raises(APIError) as exc_info:
        client.create_api_key("ci-key")
    assert not isinstance(exc_info.value, ApiKeyPlanError)
    assert exc_info.value.status_code == 403
