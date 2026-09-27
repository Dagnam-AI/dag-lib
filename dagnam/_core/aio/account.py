"""Async account / usage client methods.

Async mirror of ``dagnam._core.client.account.AccountClientMixin``: read-only
access to the caller's entitlement snapshot, dataset storage quota, and
per-API-key usage counters. Connection/timeout failures are wrapped into
``APIError`` by the shared ``_request`` transport, so this mixin only maps the
response body (mirroring the sync helper's empty-body and non-JSON fallbacks).
"""

from __future__ import annotations

from dagnam._core.aio.base import BaseAsyncDagnamClient
from dagnam._core.client.account import bootstrap_access_token
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_generic,
    response_json_value,
)
from dagnam._core.exceptions import ResponseError
from dagnam._types import JsonObject, JsonValue, ensure_json_object


class AsyncAccountMixin(BaseAsyncDagnamClient):
    """Async Account, entitlement, and usage methods for AsyncDagnamClient."""

    async def register(self, email: str, password: str) -> JsonObject:
        """Create a new account. ``POST /api/v1/auth/register`` (UNAUTHENTICATED).

        Sends ``email``/``password`` as a JSON body with no ``Authorization``
        header - this is the account-bootstrap step and no credential exists
        yet. The base ``_request`` transport does ``headers or self._headers()``,
        so an empty ``{}`` here would silently fall back to the auth header;
        passing a non-empty header dict without ``Authorization`` avoids that
        trap. Returns the created user's public profile.
        """
        resp = await self._request(
            "POST",
            "/api/v1/auth/register",
            json={"email": email, "password": password},
            headers={"Accept": "application/json"},
        )
        raise_for_generic(resp)
        return ensure_json_object(resp.json())

    async def login_for_bootstrap(self, email: str, password: str) -> str:
        """Log in once to obtain a session token, held in memory only.

        ``POST /api/v1/auth/login`` (UNAUTHENTICATED), sent as
        ``application/x-www-form-urlencoded`` (an OAuth2 password grant)
        rather than JSON, matching the backend's ``OAuth2PasswordRequestForm``.
        The password is sent in the request body only - it is never logged,
        persisted, or returned - and the returned access token is a plain
        in-memory ``str`` this method never writes to disk. It exists solely
        to authorize the one-time bootstrap API-key creation that follows it;
        callers must discard it immediately after use. See :meth:`register`
        for why a non-empty, auth-less ``headers`` dict is required here too.
        """
        resp = await self._request(
            "POST",
            "/api/v1/auth/login",
            data={"username": email, "password": password},
            headers={"Accept": "application/json"},
        )
        raise_for_generic(resp)
        return bootstrap_access_token(ensure_json_object(resp.json()))

    async def _account_get(self, path: str) -> JsonValue | str | None:
        return await self._account_write("GET", path)

    async def get_entitlements(self) -> JsonObject:
        """Return the entitlement snapshot. ``GET /api/v1/users/me/entitlements``."""
        return ensure_json_object(await self._account_get("/api/v1/users/me/entitlements"))

    async def get_credit_balance(self) -> int:
        """Return the caller's current credit balance. ``GET /api/v1/users/me/credits``."""
        body = ensure_json_object(await self._account_get("/api/v1/users/me/credits"))
        balance = body.get("balance")
        if not isinstance(balance, int) or isinstance(balance, bool):
            raise TypeError("Credit balance response did not include a 'balance' integer")
        return balance

    async def get_storage_quota(self) -> JsonObject:
        """Return dataset storage usage. ``GET /api/v1/datasets/storage/quota``."""
        return ensure_json_object(await self._account_get("/api/v1/datasets/storage/quota"))

    async def get_api_key_usage(self, key_id: str) -> JsonObject:
        """Return per-key usage. ``GET /api/v1/users/me/api-keys/{key_id}/usage``."""
        return ensure_json_object(
            await self._account_get(f"/api/v1/users/me/api-keys/{quote_path_segment(key_id)}/usage")
        )

    async def _account_write(
        self, method: str, path: str, json_body: JsonObject | None = None
    ) -> JsonValue | str | None:
        resp = await self._request(
            method, path, json=json_body, raise_for=lambda r: raise_for_generic(r)
        )
        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ResponseError:
            return resp.text

    async def get_public_profile(self, username: str) -> JsonObject:
        """Return a user's public profile. ``GET /api/v1/users/{username}/profile``.

        Requires no special permission on the backend, but the SDK still sends
        the caller's credentials like every other request.
        """
        return ensure_json_object(
            await self._account_get(f"/api/v1/users/{quote_path_segment(username)}/profile")
        )
