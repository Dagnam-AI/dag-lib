"""Wire-level coverage for user-settings and notification-preference sync client methods.

Covers ``AccountClientMixin.get_settings/update_settings/reset_settings`` and
``get_notification_prefs/update_notification_prefs``: the shared
``_account_write`` transport helper (connection/timeout wrapping, empty-body
and non-JSON fallbacks) plus the ``raise_for_generic`` error mapping these
methods share with the rest of the account surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import requests

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError, AuthError, QuotaExceededError, ResponseError

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

API = "https://api.test"
SETTINGS = f"{API}/api/v1/users/me/settings"
SETTINGS_RESET = f"{API}/api/v1/users/me/settings/reset"
NOTIFICATIONS = f"{API}/api/v1/users/me/notifications"


def test_get_settings(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(SETTINGS, json={"theme": "system", "auto_save_interval": 500})
    result = client.get_settings()
    assert result["theme"] == "system"
    assert rmock.last_request.headers["Authorization"] == "Bearer k"


def test_update_settings_sends_patch_body(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.put(SETTINGS, json={"theme": "dark"})
    result = client.update_settings({"theme": "dark"})
    assert result["theme"] == "dark"
    assert rmock.last_request.json() == {"theme": "dark"}


def test_reset_settings(client: DagnamClient, rmock: RequestsMocker) -> None:
    # rmock.post only matches a POST to this URL, so a successful call already
    # proves the client issued a POST (a GET/PUT would raise NoMockAddress).
    rmock.post(SETTINGS_RESET, json={"theme": "system"})
    result = client.reset_settings()
    assert result["theme"] == "system"


def test_get_notification_prefs(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(NOTIFICATIONS, json={"email_enabled": True})
    assert client.get_notification_prefs()["email_enabled"] is True


def test_update_notification_prefs_sends_patch_body(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.put(NOTIFICATIONS, json={"training_alerts": False})
    result = client.update_notification_prefs({"training_alerts": False})
    assert result["training_alerts"] is False
    assert rmock.last_request.json() == {"training_alerts": False}


def test_update_settings_401_raises_autherror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.put(SETTINGS, status_code=401, text="nope")
    with pytest.raises(AuthError):
        client.update_settings({"theme": "dark"})


def test_update_settings_402_raises_quotaexceedederror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.put(SETTINGS, status_code=402, json={"message": "Plan limit reached"})
    with pytest.raises(QuotaExceededError):
        client.update_settings({"theme": "dark"})


def test_update_notification_prefs_422_raises_apierror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.put(NOTIFICATIONS, status_code=422, json={"detail": "bad field"})
    with pytest.raises(APIError):
        client.update_notification_prefs({"training_alerts": "not-a-bool"})


def test_update_settings_404_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    # No not_found_exc is wired for settings, so a 404 falls through to the
    # generic APIError branch like every other unmapped status here.
    rmock.put(SETTINGS, status_code=404, json={"detail": "not found"})
    with pytest.raises(APIError):
        client.update_settings({"theme": "dark"})


def test_reset_settings_409_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(SETTINGS_RESET, status_code=409, json={"detail": "conflict"})
    with pytest.raises(APIError):
        client.reset_settings()


def test_reset_settings_empty_body_raises_response_error(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    # _account_write returns None on an empty body; _expect_object then rejects it.
    rmock.post(SETTINGS_RESET, status_code=204, text="")
    with pytest.raises(ResponseError, match="Expected JSON object"):
        client.reset_settings()


def test_reset_settings_non_json_body_raises_response_error(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    # Non-JSON body falls back to resp.text (a str); _expect_object rejects it.
    rmock.post(SETTINGS_RESET, text="not-json", headers={"Content-Type": "text/plain"})
    with pytest.raises(ResponseError, match="Expected JSON object"):
        client.reset_settings()


def test_account_write_connectionerror_wrapped(client: DagnamClient, rmock: RequestsMocker) -> None:
    # reset_settings is a POST → issued once, never retried; transport error
    # maps centrally in ``_request`` to ``APIError(0, "Request failed: ...")``.
    rmock.post(SETTINGS_RESET, exc=requests.ConnectionError("nope"))
    with pytest.raises(APIError, match="Request failed"):
        client.reset_settings()


def test_account_write_timeout_wrapped(client: DagnamClient, rmock: RequestsMocker) -> None:
    # update_notification_prefs is a PUT (retryable) → exhausts retries; stub sleep.
    client._sleep = lambda _s: None
    rmock.put(NOTIFICATIONS, exc=requests.Timeout("slow"))
    with pytest.raises(APIError, match="Request failed"):
        client.update_notification_prefs({"weekly_digest": True})
