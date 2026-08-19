"""Wire-level coverage for the async two-factor methods.

Async mirror of ``tests/core/client/test_account_two_factor.py``. The mirror is
the point: a sync/async pair that drifts is how one transport keeps a fixed bug
and the other does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import APIError, AuthError

if TYPE_CHECKING:
    from tests.typing_helpers import RespxMockRouter

PROFILE = "/api/v1/users/me/profile"
ENABLE = "/api/v1/users/me/2fa/enable"
VERIFY = "/api/v1/users/me/2fa/verify"
DISABLE = "/api/v1/users/me/2fa/disable"

ENROLLMENT = {
    "secret": "JBSWY3DPEHPK3PXP",
    "qr_code_uri": "otpauth://totp/Dagnam:me?secret=JBSWY3DPEHPK3PXP",
    "backup_codes": ["11111111", "22222222"],
}

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("declared", [True, False])
async def test_status_reads_the_field_off_the_profile(
    client: AsyncDagnamClient, mock: RespxMockRouter, declared: bool
) -> None:
    mock.get(PROFILE).mock(return_value=httpx.Response(200, json={"two_factor_enabled": declared}))
    assert await client.two_factor_enabled() is declared


async def test_a_profile_missing_the_field_reports_disabled(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get(PROFILE).mock(return_value=httpx.Response(200, json={"email": "me@example.com"}))
    assert await client.two_factor_enabled() is False


async def test_enable_sends_only_the_password_and_returns_the_material(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post(ENABLE).mock(return_value=httpx.Response(200, json=ENROLLMENT))

    result = await client.enable_two_factor("Passw0rd!")

    assert result == ENROLLMENT
    assert route.calls[0].request.read() == b'{"password":"Passw0rd!"}'


async def test_the_password_never_reaches_the_url(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post(ENABLE).mock(return_value=httpx.Response(200, json=ENROLLMENT))
    await client.enable_two_factor("Passw0rd!")
    assert "Passw0rd!" not in str(route.calls[0].request.url)


async def test_enable_with_a_wrong_password_raises_autherror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(ENABLE).mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await client.enable_two_factor("wrong")


async def test_verify_sends_the_code(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    route = mock.post(VERIFY).mock(return_value=httpx.Response(200, json={"message": "enabled"}))

    result = await client.verify_two_factor("123456")

    assert result["message"] == "enabled"
    assert route.calls[0].request.read() == b'{"code":"123456"}'


async def test_a_rejected_code_raises_rather_than_reporting_success(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(VERIFY).mock(return_value=httpx.Response(400, json={"detail": "Invalid code"}))
    with pytest.raises(APIError):
        await client.verify_two_factor("000000")


async def test_disable_sends_only_the_password(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post(DISABLE).mock(return_value=httpx.Response(200, json={"message": "off"}))

    result = await client.disable_two_factor("Passw0rd!")

    assert result["message"] == "off"
    assert route.calls[0].request.read() == b'{"password":"Passw0rd!"}'


async def test_disable_with_a_wrong_password_raises_autherror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(DISABLE).mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await client.disable_two_factor("wrong")
