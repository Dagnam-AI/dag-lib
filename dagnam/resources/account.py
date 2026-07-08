"""Account, entitlement, and usage - sync SDK surface.

Read-only views of the caller's plan and consumption:

* :func:`entitlements` - plan, period usage, limit statuses, feature flags
* :func:`storage_quota` - dataset storage usage vs. allowance
* :func:`api_key_usage` - per-API-key request counters

Exposed as ``dagnam.account.*`` to match the namespace style used by
``dagnam.projects`` / ``dagnam.deployments``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Optional
from uuid import UUID

from dagnam._core.auth import get_api_url
from dagnam._core.client import DagnamClient
from dagnam._core.resolver import resolve_client
from dagnam._types import JsonArray, JsonObject, JsonValue

# The API-key scopes and name the ``dagnam register`` bootstrap flow mints for
# a freshly-created account. Defined here (not in the CLI) because they are
# properties of the SDK's bootstrap contract, not of the terminal UI around it.
DEFAULT_SDK_SCOPES: tuple[str, ...] = ("read", "write")
DEFAULT_KEY_NAME = "dagnam-cli"


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

    >>> key_obj = dagnam.account.register("me@example.com", "correct horse battery staple")
    >>> key_obj["key"]
    """
    url = get_api_url(override=api_url)
    unauth = DagnamClient(url, "")
    unauth.register(email, password)
    token = unauth.login_for_bootstrap(email, password)
    authed = DagnamClient(url, token)
    return authed.create_api_key(name=DEFAULT_KEY_NAME, scopes=DEFAULT_SDK_SCOPES)


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


def create_api_key(
    name: str,
    scopes: Sequence[str] | None = None,
    expires_in_days: int | None = None,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Create an API key; the plaintext secret is returned once under ``key``."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.create_api_key(name, scopes, expires_in_days)


def list_api_keys(
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonArray:
    """List the caller's API keys (secrets never included)."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.list_api_keys()


def revoke_api_key(
    key_id: str | UUID,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> None:
    """Revoke a single API key by id."""
    resolved = resolve_client(client, api_key, api_url)
    resolved.revoke_api_key(str(key_id))


def get_settings(
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Return the caller's UI/editor settings."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_settings()


def update_settings(
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    **fields: JsonValue,
) -> JsonObject:
    """Patch one or more settings fields.

    >>> dagnam.account.update_settings(theme="dark")
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.update_settings(dict(fields))


def reset_settings(
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Reset settings to their defaults."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.reset_settings()


def notification_preferences(
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Return the caller's notification preferences."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_notification_prefs()


def update_notification_preferences(
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    **fields: JsonValue,
) -> JsonObject:
    """Patch one or more notification-preference fields.

    >>> dagnam.account.update_notification_preferences(training_alerts=False)
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.update_notification_prefs(dict(fields))


def get_profile(
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Return the caller's profile."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_profile()


def update_profile(
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    **fields: JsonValue,
) -> JsonObject:
    """Patch one or more profile fields.

    >>> dagnam.account.update_profile(bio="Building things.")
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.update_profile(dict(fields))


def upload_profile_photo(
    path: str | Path,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Upload a profile photo from a local file path."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.upload_profile_photo(path)


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


def change_password(
    current_password: str,
    new_password: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Change the caller's password.

    >>> dagnam.account.change_password(current, new)
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.change_password(current_password, new_password)


def list_sessions(
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonArray:
    """Return the caller's active sessions."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.list_sessions()


def revoke_session(
    session_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> None:
    """Revoke a single session by id."""
    resolved = resolve_client(client, api_key, api_url)
    resolved.revoke_session(session_id)


def revoke_all_sessions(
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Revoke every active session for the caller."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.revoke_all_sessions()


def export_data(
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Request a data export of the caller's account.

    Returns export metadata (``export_id``, ``status``, ``created_at``,
    ``expires_at``); pass ``export_id`` to :func:`download_export` once the
    export is ready.
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.export_data()


def download_export(
    export_id: str,
    out: str | Path | None = None,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Path:
    """Download a data export archive to ``out`` (a directory; default is cwd)."""
    resolved = resolve_client(client, api_key, api_url)
    dest_dir = Path(out) if out is not None else Path.cwd()
    return resolved.download_export(export_id, dest_dir)


def delete_account(
    password: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Permanently delete the caller's account. Requires the current password.

    >>> dagnam.account.delete_account(current_password)
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.delete_account(password)


__all__ = [
    "api_key_usage",
    "change_password",
    "create_api_key",
    "delete_account",
    "download_export",
    "entitlements",
    "export_data",
    "get_profile",
    "get_public_profile",
    "get_settings",
    "list_api_keys",
    "list_sessions",
    "notification_preferences",
    "register",
    "reset_settings",
    "revoke_all_sessions",
    "revoke_api_key",
    "revoke_session",
    "storage_quota",
    "update_notification_preferences",
    "update_profile",
    "update_settings",
    "upload_profile_photo",
]
