"""Account, entitlement, and usage - sync SDK surface.

Read-only views of the caller's plan and consumption:

* :func:`entitlements` - plan, period usage, limit statuses, feature flags
* :func:`storage_quota` - dataset storage usage vs. allowance
* :func:`api_key_usage` - per-API-key request counters

Exposed as ``dagnam.account.*`` to match the namespace style used by
``dagnam.projects`` / ``dagnam.deployments``.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from dagnam._core.auth import get_api_url
from dagnam._core.client import DagnamClient
from dagnam._core.client.account import ApiKeyPlanError
from dagnam._core.exceptions import QuotaExceededError
from dagnam._core.resolver import resolve_client
from dagnam._types import JsonObject

# The API-key scopes and name the ``dagnam register`` bootstrap flow mints for
# a freshly-created account. Defined here (not in the CLI) because they are
# properties of the SDK's bootstrap contract, not of the terminal UI around it.
DEFAULT_SDK_SCOPES: tuple[str, ...] = ("read", "write")
DEFAULT_KEY_NAME = "dagnam-cli"
_NO_KEYS_ON_PLAN = (
    "Account created; API keys need a paid plan. Choose one in the web app at "
    "https://dagnam.ai, create a key under Settings, Security, then run `dagnam login`."
)


def register(email: str, password: str, *, api_url: Optional[str] = None) -> JsonObject:
    """Register a new account, bootstrap a session, and mint a fresh API key.

    Orchestrates the full terminal-only onboarding flow:

    1. Create the account (``POST /api/v1/auth/register``).
    2. Log in once to obtain a short-lived session token
       (:meth:`~dagnam._core.client.account.AccountClientMixin.login_for_bootstrap`),
       held only in a local variable and never written to disk.
    3. Use that token as the ``Authorization: Bearer`` credential to mint a
       long-lived API key (``POST /api/v1/users/me/api-keys``) scoped to
       :data:`DEFAULT_SDK_SCOPES`.

    Returns the created API key object, which contains the plaintext ``key``
    exactly once. This function does not persist anything to disk - the
    caller (the ``dagnam register`` CLI command) is responsible for saving it.

    API keys need a paid plan and a new account starts on Free, so step 3 is
    usually refused: that raises :class:`~dagnam.QuotaExceededError` saying the
    account was created and where to get a key.

    >>> key_obj = dagnam.account.register("me@example.com", "correct horse battery staple")
    >>> key_obj["key"]
    """
    url = get_api_url(override=api_url)
    unauth = DagnamClient(url, "")
    unauth.register(email, password)
    token = unauth.login_for_bootstrap(email, password)
    authed = DagnamClient(url, token)
    try:
        return authed.create_api_key(name=DEFAULT_KEY_NAME, scopes=DEFAULT_SDK_SCOPES)
    except ApiKeyPlanError as exc:
        raise QuotaExceededError(_NO_KEYS_ON_PLAN) from exc


def entitlements(
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Return the current credential's entitlement snapshot.

    Includes ``plan``, ``period`` usage, per-limit ``limits``, ``features``,
    and the ``read_only_grace`` flag.

    >>> dagnam.account.entitlements()["plan"]["code"]
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_entitlements()


def storage_quota(
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Return dataset storage usage and remaining allowance."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_storage_quota()


def api_key_usage(
    key_id: str | UUID,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Return usage counters for a single API key."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_api_key_usage(str(key_id))


def get_public_profile(
    username: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Return a user's public profile by username."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_public_profile(username)


__all__ = [
    "api_key_usage",
    "entitlements",
    "get_public_profile",
    "register",
    "storage_quota",
]
