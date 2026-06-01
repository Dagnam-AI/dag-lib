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

from dagnam._core.client import DagnamClient
from dagnam._core.resolver import resolve_client
from dagnam._types import JsonObject


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


__all__ = [
    "api_key_usage",
    "entitlements",
    "storage_quota",
]
