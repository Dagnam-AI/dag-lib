"""Wire-level coverage for the async account-bootstrap methods.

Async mirror of ``tests/core/client/test_register_bootstrap.py``: exercises
``AsyncAccountMixin.register``/``login_for_bootstrap``. Connection/timeout
wrapping is the shared ``_request`` transport's job (already covered by
``tests/core/aio/test_async_base.py``), so this file focuses on the JSON vs.
form-encoded body, the no-auth-header trap, and the response mapping these
two methods add.
"""

from __future__ import annotations

import json as json_mod
from typing import TYPE_CHECKING
from urllib.parse import parse_qs

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import APIError, AuthError

if TYPE_CHECKING:
    from tests.typing_helpers import RespxMockRouter

pytestmark = pytest.mark.anyio

REGISTER = "/api/v1/auth/register"
LOGIN = "/api/v1/auth/login"

USER = {
    "id": "u1",
    "email": "a@b.c",
    "first_name": None,
    "last_name": None,
    "profile_photo_url": None,
    "created_at": "2026-01-01T00:00:00",
    "role": "user",
}

LOGIN_RESPONSE = {
    "access_token": "tok-123",
    "token_type": "bearer",
    "expires_in": 900,
    "user": USER,
}


def _no_auth_header(headers: httpx.Headers) -> bool:
    return "authorization" not in {key.lower() for key in headers}


# ------------------------------------------------------------------ register


async def test_register_sends_json_body_with_no_auth_header(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post(REGISTER).mock(return_value=httpx.Response(201, json=USER))
    result = await client.register("a@b.c", "Secret123!")
    assert result == USER
    request = route.calls[0].request
    assert json_mod.loads(request.read()) == {"email": "a@b.c", "password": "Secret123!"}
    assert _no_auth_header(request.headers)


async def test_register_400_email_exists_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(REGISTER).mock(
        return_value=httpx.Response(400, json={"detail": "Email already registered"})
    )
    with pytest.raises(APIError) as exc_info:
        await client.register("dup@b.c", "Secret123!")
    assert exc_info.value.status_code == 400


# ------------------------------------------------------- login_for_bootstrap


async def test_login_for_bootstrap_sends_form_body_with_no_auth_header(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post(LOGIN).mock(return_value=httpx.Response(200, json=LOGIN_RESPONSE))
    token = await client.login_for_bootstrap("a@b.c", "Secret123!")
    assert token == "tok-123"
    request = route.calls[0].request
    assert request.headers["content-type"].startswith("application/x-www-form-urlencoded")
    assert parse_qs(request.read().decode()) == {
        "username": ["a@b.c"],
        "password": ["Secret123!"],
    }
    assert _no_auth_header(request.headers)


async def test_login_for_bootstrap_401_raises_autherror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(LOGIN).mock(
        return_value=httpx.Response(401, json={"detail": "Incorrect email or password"})
    )
    with pytest.raises(AuthError):
        await client.login_for_bootstrap("a@b.c", "wrong")


async def test_login_for_bootstrap_403_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    """Forward-compat: a future email-verification gate returns 403 here."""
    mock.post(LOGIN).mock(return_value=httpx.Response(403, json={"detail": "Account not verified"}))
    with pytest.raises(APIError) as exc_info:
        await client.login_for_bootstrap("a@b.c", "Secret123!")
    assert exc_info.value.status_code == 403


async def test_login_for_bootstrap_missing_access_token_raises_typeerror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(LOGIN).mock(return_value=httpx.Response(200, json={"token_type": "bearer"}))
    with pytest.raises(TypeError, match="access_token"):
        await client.login_for_bootstrap("a@b.c", "Secret123!")
