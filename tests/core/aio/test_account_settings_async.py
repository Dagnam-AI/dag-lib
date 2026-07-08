"""Wire-level coverage for async user-settings and notification-preference methods.

Async mirror of ``tests/core/client/test_account_settings.py``: exercises the
shared ``_account_write`` transport helper and the ``raise_for_generic`` error
mapping for ``AsyncAccountMixin.get_settings/update_settings/reset_settings``
and ``get_notification_prefs/update_notification_prefs``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import APIError, AuthError, QuotaExceededError

if TYPE_CHECKING:
    from tests.typing_helpers import RespxMockRouter

SETTINGS = "/api/v1/users/me/settings"
SETTINGS_RESET = "/api/v1/users/me/settings/reset"
NOTIFICATIONS = "/api/v1/users/me/notifications"

pytestmark = pytest.mark.anyio


async def test_get_settings(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get(SETTINGS).mock(
        return_value=httpx.Response(200, json={"theme": "system", "auto_save_interval": 500})
    )
    result = await client.get_settings()
    assert result["theme"] == "system"


async def test_update_settings_sends_patch_body(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.put(SETTINGS).mock(return_value=httpx.Response(200, json={"theme": "dark"}))
    result = await client.update_settings({"theme": "dark"})
    assert result["theme"] == "dark"
    assert route.calls[0].request.read() == b'{"theme":"dark"}'


async def test_reset_settings(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    route = mock.post(SETTINGS_RESET).mock(
        return_value=httpx.Response(200, json={"theme": "system"})
    )
    result = await client.reset_settings()
    assert result["theme"] == "system"
    assert route.calls[0].request.method == "POST"


async def test_get_notification_prefs(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get(NOTIFICATIONS).mock(return_value=httpx.Response(200, json={"email_enabled": True}))
    assert (await client.get_notification_prefs())["email_enabled"] is True


async def test_update_notification_prefs_sends_patch_body(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.put(NOTIFICATIONS).mock(
        return_value=httpx.Response(200, json={"training_alerts": False})
    )
    result = await client.update_notification_prefs({"training_alerts": False})
    assert result["training_alerts"] is False
    assert route.calls[0].request.read() == b'{"training_alerts":false}'


async def test_update_settings_401_raises_autherror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.put(SETTINGS).mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await client.update_settings({"theme": "dark"})


async def test_update_settings_402_raises_quotaexceedederror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.put(SETTINGS).mock(
        return_value=httpx.Response(402, json={"message": "Plan limit reached"})
    )
    with pytest.raises(QuotaExceededError):
        await client.update_settings({"theme": "dark"})


async def test_update_notification_prefs_422_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.put(NOTIFICATIONS).mock(return_value=httpx.Response(422, json={"detail": "bad field"}))
    with pytest.raises(APIError):
        await client.update_notification_prefs({"training_alerts": "not-a-bool"})


async def test_update_settings_404_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    # No not_found_exc is wired for settings, so a 404 falls through to the
    # generic APIError branch like every other unmapped status here.
    mock.put(SETTINGS).mock(return_value=httpx.Response(404, json={"detail": "not found"}))
    with pytest.raises(APIError):
        await client.update_settings({"theme": "dark"})


async def test_reset_settings_409_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(SETTINGS_RESET).mock(return_value=httpx.Response(409, json={"detail": "conflict"}))
    with pytest.raises(APIError):
        await client.reset_settings()


async def test_reset_settings_empty_body_raises_type_error(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(SETTINGS_RESET).mock(return_value=httpx.Response(204))
    with pytest.raises(TypeError, match="Expected JSON object"):
        await client.reset_settings()


async def test_reset_settings_non_json_body_raises_type_error(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(SETTINGS_RESET).mock(
        return_value=httpx.Response(200, text="not-json", headers={"Content-Type": "text/plain"})
    )
    with pytest.raises(TypeError, match="Expected JSON object"):
        await client.reset_settings()
