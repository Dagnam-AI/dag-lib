"""Async account / usage client methods.

Async mirror of ``dagnam._core.client.account.AccountClientMixin``: read-only
access to the caller's entitlement snapshot, dataset storage quota, and
per-API-key usage counters. Connection/timeout failures are wrapped into
``APIError`` by the shared ``_request`` transport, so this mixin only maps the
response body (mirroring the sync helper's empty-body and non-JSON fallbacks).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import httpx

from dagnam._core.aio.base import BaseAsyncDagnamClient, content_disposition_safe_name
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_generic,
    raise_for_upload,
    response_json_value,
)
from dagnam._core.exceptions import APIError
from dagnam._types import JsonArray, JsonObject, JsonValue, ensure_json_array, ensure_json_object

# Mirrors dagnam._core.client.account._ACCOUNT_DELETION_CONFIRMATION (same
# literal) - the backend's required machine confirmation token for account
# deletion, supplied by the SDK itself; the human caller never types it.
_ACCOUNT_DELETION_CONFIRMATION = "DELETE MY ACCOUNT"


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
        body = ensure_json_object(resp.json())
        token = body.get("access_token")
        if not isinstance(token, str):
            raise TypeError("Login response did not include an 'access_token' string")
        return token

    async def _account_get(self, path: str) -> JsonValue | str | None:
        return await self._account_write("GET", path)

    async def get_entitlements(self) -> JsonObject:
        """Return the entitlement snapshot. ``GET /api/v1/users/me/entitlements``."""
        return ensure_json_object(await self._account_get("/api/v1/users/me/entitlements"))

    async def get_storage_quota(self) -> JsonObject:
        """Return dataset storage usage. ``GET /api/v1/datasets/storage/quota``."""
        return ensure_json_object(await self._account_get("/api/v1/datasets/storage/quota"))

    async def get_api_key_usage(self, key_id: str) -> JsonObject:
        """Return per-key usage. ``GET /api/v1/users/me/api-keys/{key_id}/usage``."""
        return ensure_json_object(
            await self._account_get(f"/api/v1/users/me/api-keys/{quote_path_segment(key_id)}/usage")
        )

    async def create_api_key(
        self,
        name: str,
        scopes: Sequence[str] | None = None,
        expires_in_days: int | None = None,
    ) -> JsonObject:
        """Create an API key. ``POST /api/v1/users/me/api-keys``.

        The returned object contains the plaintext ``key`` exactly once; the
        backend never returns it again. ``scopes`` maps to the request's
        ``permissions`` field (omitted when ``None`` so the backend applies its
        default). ``expires_in_days`` sets an optional expiry.
        """
        body: JsonObject = {"name": name}
        if scopes is not None:
            body["permissions"] = list(scopes)
        if expires_in_days is not None:
            body["expires_in_days"] = expires_in_days
        return ensure_json_object(
            await self._account_write("POST", "/api/v1/users/me/api-keys", body)
        )

    async def list_api_keys(self) -> JsonArray:
        """List the caller's API keys (secrets never included).

        ``GET /api/v1/users/me/api-keys``.
        """
        return ensure_json_array(await self._account_get("/api/v1/users/me/api-keys"))

    async def revoke_api_key(self, key_id: str) -> None:
        """Revoke (soft-delete) one API key.

        ``DELETE /api/v1/users/me/api-keys/{key_id}``.
        """
        await self._account_write(
            "DELETE", f"/api/v1/users/me/api-keys/{quote_path_segment(key_id)}"
        )

    async def _account_write(
        self, method: str, path: str, json_body: JsonObject | None = None
    ) -> JsonValue | str | None:
        resp = await self._request(method, path, json=json_body)
        raise_for_generic(resp)
        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ValueError:
            return resp.text

    async def get_settings(self) -> JsonObject:
        """Return the caller's UI/editor settings. ``GET /api/v1/users/me/settings``."""
        return ensure_json_object(await self._account_get("/api/v1/users/me/settings"))

    async def update_settings(self, patch: JsonObject) -> JsonObject:
        """Patch one or more settings fields. ``PUT /api/v1/users/me/settings``."""
        return ensure_json_object(
            await self._account_write("PUT", "/api/v1/users/me/settings", patch)
        )

    async def reset_settings(self) -> JsonObject:
        """Reset settings to defaults. ``POST /api/v1/users/me/settings/reset``."""
        return ensure_json_object(
            await self._account_write("POST", "/api/v1/users/me/settings/reset")
        )

    async def get_notification_prefs(self) -> JsonObject:
        """Return notification preferences. ``GET /api/v1/users/me/notifications``."""
        return ensure_json_object(await self._account_get("/api/v1/users/me/notifications"))

    async def update_notification_prefs(self, patch: JsonObject) -> JsonObject:
        """Patch notification preferences. ``PUT /api/v1/users/me/notifications``."""
        return ensure_json_object(
            await self._account_write("PUT", "/api/v1/users/me/notifications", patch)
        )

    async def get_profile(self) -> JsonObject:
        """Return the caller's profile. ``GET /api/v1/users/me/profile``."""
        return ensure_json_object(await self._account_get("/api/v1/users/me/profile"))

    async def update_profile(self, patch: JsonObject) -> JsonObject:
        """Patch one or more profile fields. ``PUT /api/v1/users/me/profile``."""
        return ensure_json_object(
            await self._account_write("PUT", "/api/v1/users/me/profile", patch)
        )

    async def upload_profile_photo(self, file_path: str | Path) -> JsonObject:
        """Upload a profile photo. ``POST /api/v1/users/me/profile/photo`` (multipart).

        Streams the file as ``multipart/form-data`` under the ``file`` field via
        the shared ``_request`` transport, mirroring the async dataset-upload
        method rather than ``_account_write`` (which only sends JSON bodies).
        """
        path = Path(file_path)
        if not path.is_file():  # noqa: ASYNC240 - one-shot local stat before opening, not I/O-bound
            raise FileNotFoundError(f"No such file: {path}")

        with open(path, "rb") as fh:
            files = {"file": (path.name, fh, "application/octet-stream")}
            resp = await self._request(
                "POST",
                "/api/v1/users/me/profile/photo",
                files=files,
                timeout=None,
            )
        raise_for_upload(resp)
        return ensure_json_object(resp.json())

    async def get_public_profile(self, username: str) -> JsonObject:
        """Return a user's public profile. ``GET /api/v1/users/{username}/profile``.

        Requires no special permission on the backend, but the SDK still sends
        the caller's credentials like every other request.
        """
        return ensure_json_object(
            await self._account_get(f"/api/v1/users/{quote_path_segment(username)}/profile")
        )

    async def change_password(self, current_password: str, new_password: str) -> JsonObject:
        """Change the caller's password. ``POST /api/v1/users/me/change-password``.

        Sends both values in the JSON request body only; neither is logged or
        returned in this method's result beyond whatever the backend's own
        confirmation payload contains (a plain ``message`` field, never a
        password value).
        """
        return ensure_json_object(
            await self._account_write(
                "POST",
                "/api/v1/users/me/change-password",
                {"current_password": current_password, "new_password": new_password},
            )
        )

    async def list_sessions(self) -> JsonArray:
        """Return the caller's active sessions. ``GET /api/v1/users/me/sessions``."""
        return ensure_json_array(await self._account_get("/api/v1/users/me/sessions"))

    async def revoke_session(self, session_id: str) -> None:
        """Revoke one session. ``DELETE /api/v1/users/me/sessions/{session_id}``."""
        await self._account_write(
            "DELETE", f"/api/v1/users/me/sessions/{quote_path_segment(session_id)}"
        )

    async def revoke_all_sessions(self) -> JsonObject:
        """Revoke every session and invalidate every live token for the caller.

        ``POST /api/v1/users/me/revoke-all-sessions`` bumps the caller's
        ``token_version`` and purges cached refresh tokens, which is what
        actually invalidates outstanding access/refresh tokens immediately -
        the real "log out everywhere" primitive. This is distinct from
        ``DELETE /api/v1/users/me/sessions/{id}`` (``revoke_session``), which
        only removes ``UserSession`` bookkeeping rows and does not by itself
        invalidate a live token.
        """
        return ensure_json_object(
            await self._account_write("POST", "/api/v1/users/me/revoke-all-sessions")
        )

    async def export_data(self) -> JsonObject:
        """Request a data export of the caller's account. ``POST /api/v1/users/me/export``.

        Returns export metadata (``export_id``, ``status``, ``created_at``,
        ``expires_at``); pass ``export_id`` to :meth:`download_export` to
        fetch the archive once it is ready.
        """
        return ensure_json_object(await self._account_write("POST", "/api/v1/users/me/export"))

    async def download_export(self, export_id: str, dest_dir: str | Path) -> Path:
        """Stream-download a data export archive to a file inside ``dest_dir``.

        ``GET /api/v1/users/me/export/{export_id}``. The saved filename is
        taken from the response's ``Content-Disposition`` header and reduced
        to a bare basename (see ``content_disposition_safe_name``), so a
        hostile or malformed header can never write outside ``dest_dir``. The
        body is streamed chunk by chunk straight to disk - the whole archive
        is never buffered in memory.
        """
        url = f"{self.api_url}/api/v1/users/me/export/{quote_path_segment(export_id)}"
        try:
            async with self._client.stream("GET", url, headers=self._headers()) as resp:
                if not resp.is_success:
                    await resp.aread()  # populate the body for the error message
                    raise_for_generic(resp)
                name = content_disposition_safe_name(
                    resp.headers.get("content-disposition"), default="export.zip"
                )
                dest = Path(dest_dir) / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as fh:
                    async for chunk in resp.aiter_bytes():
                        fh.write(chunk)
                return dest
        except httpx.ConnectError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

    async def delete_account(self, password: str) -> JsonObject:
        """Permanently delete the caller's account. ``DELETE /api/v1/users/me``.

        Sends the password in the request body only; it is never logged,
        printed, or returned. The backend also requires a fixed confirmation
        token, which the SDK supplies automatically.
        """
        return ensure_json_object(
            await self._account_write(
                "DELETE",
                "/api/v1/users/me",
                {"password": password, "confirmation": _ACCOUNT_DELETION_CONFIRMATION},
            )
        )
