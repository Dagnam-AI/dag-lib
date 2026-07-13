"""Synchronous account / usage client methods.

Read-only access to the caller's plan, entitlement snapshot, storage quota, and
per-API-key usage counters. These are the building blocks behind
``dagnam.account.*`` and the ``dagnam usage`` CLI command.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from dagnam._core.client.base import (
    ALLOW_REDIRECTS,
    DEFAULT_TIMEOUT,
    APIError,
    BaseDagnamClient,
    content_disposition_safe_name,
    requests,
)
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_generic,
    raise_for_upload,
    response_json_object,
    response_json_value,
)
from dagnam._core.exceptions import ResponseError
from dagnam._types import JsonArray, JsonObject, JsonValue

# The backend's required machine confirmation token for account deletion - the
# SDK supplies this literal itself; the human caller never types it. Defined
# once here and mirrored (same literal) in the async client.
_ACCOUNT_DELETION_CONFIRMATION = "DELETE MY ACCOUNT"


class AccountClientMixin(BaseDagnamClient):
    """Account, entitlement, and usage methods for DagnamClient."""

    def register(self, email: str, password: str) -> JsonObject:
        """Create a new account. ``POST /api/v1/auth/register`` (UNAUTHENTICATED).

        Sends ``email``/``password`` as a JSON body; no ``Authorization``
        header is sent, since this is the account-bootstrap step and no
        credential exists yet. Returns the created user's public profile.
        """
        url = f"{self.api_url}/api/v1/auth/register"
        try:
            resp = requests.request(
                "POST",
                url,
                json={"email": email, "password": password},
                timeout=DEFAULT_TIMEOUT,
                allow_redirects=ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_generic(resp)
        return response_json_object(resp)

    def login_for_bootstrap(self, email: str, password: str) -> str:
        """Log in once to obtain a session token, held in memory only.

        ``POST /api/v1/auth/login`` (UNAUTHENTICATED), sent as
        ``application/x-www-form-urlencoded`` (an OAuth2 password grant)
        rather than JSON, matching the backend's ``OAuth2PasswordRequestForm``.
        The password is sent in the request body only - it is never logged,
        persisted, or returned - and the returned access token is a plain
        in-memory ``str`` this method never writes to disk. It exists solely
        to authorize the one-time bootstrap API-key creation that follows it;
        callers must discard it immediately after use.
        """
        url = f"{self.api_url}/api/v1/auth/login"
        try:
            resp = requests.request(
                "POST",
                url,
                data={"username": email, "password": password},
                timeout=DEFAULT_TIMEOUT,
                allow_redirects=ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_generic(resp)
        body = response_json_object(resp)
        token = body.get("access_token")
        if not isinstance(token, str):
            raise TypeError("Login response did not include an 'access_token' string")
        return token

    def _account_get(self, path: str) -> JsonValue | str | None:
        return self._account_write("GET", path)

    def get_entitlements(self) -> JsonObject:
        """Return the entitlement snapshot. ``GET /api/v1/users/me/entitlements``."""
        return self._expect_object(self._account_get("/api/v1/users/me/entitlements"))

    def get_storage_quota(self) -> JsonObject:
        """Return dataset storage usage. ``GET /api/v1/datasets/storage/quota``."""
        return self._expect_object(self._account_get("/api/v1/datasets/storage/quota"))

    def get_api_key_usage(self, key_id: str) -> JsonObject:
        """Return per-key usage. ``GET /api/v1/users/me/api-keys/{key_id}/usage``."""
        return self._expect_object(
            self._account_get(f"/api/v1/users/me/api-keys/{quote_path_segment(key_id)}/usage")
        )

    def create_api_key(
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
        return self._expect_object(self._account_write("POST", "/api/v1/users/me/api-keys", body))

    def list_api_keys(self) -> JsonArray:
        """List the caller's API keys (secrets never included).

        ``GET /api/v1/users/me/api-keys``.
        """
        return self._expect_array(self._account_get("/api/v1/users/me/api-keys"))

    def revoke_api_key(self, key_id: str) -> None:
        """Revoke (soft-delete) one API key.

        ``DELETE /api/v1/users/me/api-keys/{key_id}``.
        """
        self._account_write("DELETE", f"/api/v1/users/me/api-keys/{quote_path_segment(key_id)}")

    def _account_write(
        self, method: str, path: str, json_body: JsonObject | None = None
    ) -> JsonValue | str | None:
        url = f"{self.api_url}{path}"
        resp = self._request(
            method,
            url,
            raise_for=lambda r: raise_for_generic(r),
            json=json_body,
            allow_redirects=ALLOW_REDIRECTS,
        )
        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ResponseError:
            return resp.text

    def get_settings(self) -> JsonObject:
        """Return the caller's UI/editor settings. ``GET /api/v1/users/me/settings``."""
        return self._expect_object(self._account_get("/api/v1/users/me/settings"))

    def update_settings(self, patch: JsonObject) -> JsonObject:
        """Patch one or more settings fields. ``PUT /api/v1/users/me/settings``."""
        return self._expect_object(self._account_write("PUT", "/api/v1/users/me/settings", patch))

    def reset_settings(self) -> JsonObject:
        """Reset settings to defaults. ``POST /api/v1/users/me/settings/reset``."""
        return self._expect_object(self._account_write("POST", "/api/v1/users/me/settings/reset"))

    def get_notification_prefs(self) -> JsonObject:
        """Return notification preferences. ``GET /api/v1/users/me/notifications``."""
        return self._expect_object(self._account_get("/api/v1/users/me/notifications"))

    def update_notification_prefs(self, patch: JsonObject) -> JsonObject:
        """Patch notification preferences. ``PUT /api/v1/users/me/notifications``."""
        return self._expect_object(
            self._account_write("PUT", "/api/v1/users/me/notifications", patch)
        )

    def get_profile(self) -> JsonObject:
        """Return the caller's profile. ``GET /api/v1/users/me/profile``."""
        return self._expect_object(self._account_get("/api/v1/users/me/profile"))

    def update_profile(self, patch: JsonObject) -> JsonObject:
        """Patch one or more profile fields. ``PUT /api/v1/users/me/profile``."""
        return self._expect_object(self._account_write("PUT", "/api/v1/users/me/profile", patch))

    def upload_profile_photo(self, file_path: str | Path) -> JsonObject:
        """Upload a profile photo. ``POST /api/v1/users/me/profile/photo`` (multipart).

        Streams the file as ``multipart/form-data`` under the ``file`` field,
        mirroring the dataset-upload client method rather than routing through
        ``_account_write`` (which only sends JSON bodies).
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"No such file: {path}")

        url = f"{self.api_url}/api/v1/users/me/profile/photo"
        try:
            with open(path, "rb") as fh:
                files = {"file": (path.name, fh)}
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    files=files,
                    timeout=None,
                    allow_redirects=ALLOW_REDIRECTS,
                )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_upload(resp)
        return response_json_object(resp)

    def get_public_profile(self, username: str) -> JsonObject:
        """Return a user's public profile. ``GET /api/v1/users/{username}/profile``.

        Requires no special permission on the backend, but the SDK still sends
        the caller's credentials like every other request.
        """
        return self._expect_object(
            self._account_get(f"/api/v1/users/{quote_path_segment(username)}/profile")
        )

    def change_password(self, current_password: str, new_password: str) -> JsonObject:
        """Change the caller's password. ``POST /api/v1/users/me/change-password``.

        Sends both values in the JSON request body only; neither is logged or
        returned in this method's result beyond whatever the backend's own
        confirmation payload contains (a plain ``message`` field, never a
        password value).
        """
        return self._expect_object(
            self._account_write(
                "POST",
                "/api/v1/users/me/change-password",
                {"current_password": current_password, "new_password": new_password},
            )
        )

    def list_sessions(self) -> JsonArray:
        """Return the caller's active sessions. ``GET /api/v1/users/me/sessions``."""
        return self._expect_array(self._account_get("/api/v1/users/me/sessions"))

    def revoke_session(self, session_id: str) -> None:
        """Revoke one session. ``DELETE /api/v1/users/me/sessions/{session_id}``."""
        self._account_write("DELETE", f"/api/v1/users/me/sessions/{quote_path_segment(session_id)}")

    def revoke_all_sessions(self) -> JsonObject:
        """Revoke every session and invalidate every live token for the caller.

        ``POST /api/v1/users/me/revoke-all-sessions`` bumps the caller's
        ``token_version`` and purges cached refresh tokens, which is what
        actually invalidates outstanding access/refresh tokens immediately -
        the real "log out everywhere" primitive. This is distinct from
        ``DELETE /api/v1/users/me/sessions/{id}`` (``revoke_session``), which
        only removes ``UserSession`` bookkeeping rows and does not by itself
        invalidate a live token.
        """
        return self._expect_object(
            self._account_write("POST", "/api/v1/users/me/revoke-all-sessions")
        )

    def export_data(self) -> JsonObject:
        """Request a data export of the caller's account. ``POST /api/v1/users/me/export``.

        Returns export metadata (``export_id``, ``status``, ``created_at``,
        ``expires_at``); pass ``export_id`` to :meth:`download_export` to
        fetch the archive once it is ready.
        """
        return self._expect_object(self._account_write("POST", "/api/v1/users/me/export"))

    def download_export(self, export_id: str, dest_dir: str | Path) -> Path:
        """Stream-download a data export archive to a file inside ``dest_dir``.

        ``GET /api/v1/users/me/export/{export_id}``. The saved filename is
        taken from the response's ``Content-Disposition`` header and reduced
        to a bare basename (see ``content_disposition_safe_name``), so a
        hostile or malformed header can never write outside ``dest_dir``.
        The body is streamed straight to disk, never buffered in memory.
        """
        url = f"{self.api_url}/api/v1/users/me/export/{quote_path_segment(export_id)}"
        resp = self._get_stream(url)
        raise_for_generic(resp)
        name = content_disposition_safe_name(
            resp.headers.get("Content-Disposition"), default="export.zip"
        )
        return self._stream_response_to_file(resp, Path(dest_dir) / name)

    def delete_account(self, password: str) -> JsonObject:
        """Permanently delete the caller's account. ``DELETE /api/v1/users/me``.

        Sends the password in the request body only; it is never logged,
        printed, or returned. The backend also requires a fixed confirmation
        token, which the SDK supplies automatically.
        """
        return self._expect_object(
            self._account_write(
                "DELETE",
                "/api/v1/users/me",
                {"password": password, "confirmation": _ACCOUNT_DELETION_CONFIRMATION},
            )
        )
