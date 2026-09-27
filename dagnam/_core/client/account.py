"""Synchronous account / usage client methods.

Read-only access to the caller's plan, entitlement snapshot, storage quota, and
per-API-key usage counters. These are the building blocks behind
``dagnam.account.*`` and the ``dagnam usage`` CLI command.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from dagnam._core.client.base import (
    ALLOW_REDIRECTS,
    DEFAULT_TIMEOUT,
    APIError,
    BaseDagnamClient,
    requests,
)
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_generic,
    response_json_object,
    response_json_value,
)
from dagnam._core.exceptions import AuthError, QuotaExceededError, ResponseError
from dagnam._types import JsonObject, JsonValue

# The backend's plan refusal codes for creating a key: a plan without API keys
# answers 403 ``feature_gated``; a plan at its key allowance, 402 ``limit_exceeded``.
_PLAN_REFUSAL_CODES = frozenset({"feature_gated", "limit_exceeded"})


class ApiKeyPlanError(QuotaExceededError):
    """The caller's plan refused to create an API key (its refusal body says so)."""


def raise_for_api_key_create(resp: requests.Response) -> None:
    """Map a plan's refusal to create a key to :class:`ApiKeyPlanError`.

    Keyed on the refusal code in the body, so any other 402 or 403 (an
    unverified email, a suspended account) maps exactly as everywhere else.
    """
    if resp.status_code in (402, 403):
        try:
            body = response_json_value(resp)
        except ResponseError:
            body = None
        if isinstance(body, dict) and body.get("error") in _PLAN_REFUSAL_CODES:
            message = body.get("message")
            raise ApiKeyPlanError(message if isinstance(message, str) else "Plan limit reached")
    raise_for_generic(resp)


def bootstrap_access_token(body: JsonObject) -> str:
    """Return the session token from a password login, refusing a 2FA challenge.

    ``POST /api/v1/auth/login`` answers an account with two-factor
    authentication with a challenge (``two_factor_required: true``) and no
    token. The CLI cannot complete that challenge, so the caller is sent to the
    web app instead of failing on a missing field.
    """
    if body.get("two_factor_required") is True:
        raise AuthError(
            "This account uses two-factor authentication, so it cannot sign in with a "
            "password here. Sign in at https://dagnam.ai, create an API key in Settings, "
            "Security, then run `dagnam login`."
        )
    token = body.get("access_token")
    if not isinstance(token, str):
        raise TypeError("Login response did not include an 'access_token' string")
    return token


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
        return bootstrap_access_token(response_json_object(resp))

    def _account_get(self, path: str) -> JsonValue | str | None:
        return self._account_write("GET", path)

    def get_entitlements(self) -> JsonObject:
        """Return the entitlement snapshot. ``GET /api/v1/users/me/entitlements``."""
        return self._expect_object(self._account_get("/api/v1/users/me/entitlements"))

    def get_credit_balance(self) -> int:
        """Return the caller's current credit balance. ``GET /api/v1/users/me/credits``."""
        body = self._expect_object(self._account_get("/api/v1/users/me/credits"))
        balance = body.get("balance")
        if not isinstance(balance, int) or isinstance(balance, bool):
            raise TypeError("Credit balance response did not include a 'balance' integer")
        return balance

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

        The route accepts only a browser-session token, never an API key, so
        the one caller is :func:`dagnam.account.register`, which holds the
        session token from :meth:`login_for_bootstrap`. The returned object
        contains the plaintext ``key`` exactly once; the backend never returns
        it again. ``scopes`` maps to the request's ``permissions`` field
        (omitted when ``None`` so the backend applies its default).
        ``expires_in_days`` sets an optional expiry. A plan that refuses keys
        raises :class:`ApiKeyPlanError`.
        """
        body: JsonObject = {"name": name}
        if scopes is not None:
            body["permissions"] = list(scopes)
        if expires_in_days is not None:
            body["expires_in_days"] = expires_in_days
        return self._expect_object(
            self._account_write(
                "POST", "/api/v1/users/me/api-keys", body, raise_for=raise_for_api_key_create
            )
        )

    def _account_write(
        self,
        method: str,
        path: str,
        json_body: JsonObject | None = None,
        *,
        raise_for: Callable[[requests.Response], None] | None = None,
    ) -> JsonValue | str | None:
        url = f"{self.api_url}{path}"
        resp = self._request(
            method,
            url,
            raise_for=raise_for or (lambda r: raise_for_generic(r)),
            json=json_body,
            allow_redirects=ALLOW_REDIRECTS,
        )
        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ResponseError:
            return resp.text

    def get_public_profile(self, username: str) -> JsonObject:
        """Return a user's public profile. ``GET /api/v1/users/{username}/profile``.

        Requires no special permission on the backend, but the SDK still sends
        the caller's credentials like every other request.
        """
        return self._expect_object(
            self._account_get(f"/api/v1/users/{quote_path_segment(username)}/profile")
        )
