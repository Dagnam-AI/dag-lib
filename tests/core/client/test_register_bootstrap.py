"""Wire-level coverage for the sync account-bootstrap methods.

Covers ``AccountClientMixin.register``/``login_for_bootstrap``: the two
UNAUTHENTICATED calls behind ``dagnam register``. ``register`` is JSON;
``login_for_bootstrap`` is form-encoded (OAuth2 password grant) - neither
sends an ``Authorization`` header.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import parse_qs

import pytest
import requests

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError, AuthError

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, RequestsMocker

API = "https://api.test"
REGISTER = f"{API}/api/v1/auth/register"
LOGIN = f"{API}/api/v1/auth/login"

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


# ------------------------------------------------------------------ register


def test_register_sends_json_body_with_no_auth_header(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(REGISTER, status_code=201, json=USER)
    result = client.register("a@b.c", "Secret123!")
    assert result == USER
    assert rmock.last_request.json() == {"email": "a@b.c", "password": "Secret123!"}
    assert "Authorization" not in rmock.last_request.headers


def test_register_400_email_exists_raises_apierror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(REGISTER, status_code=400, json={"detail": "Email already registered"})
    with pytest.raises(APIError) as exc_info:
        client.register("dup@b.c", "Secret123!")
    assert exc_info.value.status_code == 400


def test_register_connectionerror_wrapped(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "request", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.register("a@b.c", "Secret123!")


def test_register_timeout_wrapped(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "request", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.register("a@b.c", "Secret123!")


# ------------------------------------------------------- login_for_bootstrap


def test_login_for_bootstrap_sends_form_body_with_no_auth_header(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(LOGIN, status_code=200, json=LOGIN_RESPONSE)
    token = client.login_for_bootstrap("a@b.c", "Secret123!")
    assert token == "tok-123"
    req = rmock.last_request
    assert req.headers["Content-Type"].startswith("application/x-www-form-urlencoded")
    assert parse_qs(req.text or "") == {"username": ["a@b.c"], "password": ["Secret123!"]}
    assert "Authorization" not in req.headers


def test_login_for_bootstrap_401_raises_autherror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(LOGIN, status_code=401, json={"detail": "Incorrect email or password"})
    with pytest.raises(AuthError):
        client.login_for_bootstrap("a@b.c", "wrong")


def test_login_for_bootstrap_403_raises_apierror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    """Forward-compat: a future email-verification gate returns 403 here."""
    rmock.post(LOGIN, status_code=403, json={"detail": "Account not verified"})
    with pytest.raises(APIError) as exc_info:
        client.login_for_bootstrap("a@b.c", "Secret123!")
    assert exc_info.value.status_code == 403


def test_login_for_bootstrap_missing_access_token_raises_typeerror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(LOGIN, status_code=200, json={"token_type": "bearer"})
    with pytest.raises(TypeError, match="access_token"):
        client.login_for_bootstrap("a@b.c", "Secret123!")


def test_login_for_bootstrap_connectionerror_wrapped(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "request", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.login_for_bootstrap("a@b.c", "Secret123!")


def test_login_for_bootstrap_timeout_wrapped(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "request", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.login_for_bootstrap("a@b.c", "Secret123!")
