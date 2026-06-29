"""Async account / usage client methods.

Async mirror of ``dagnam._core.client.account.AccountClientMixin``: read-only
access to the caller's entitlement snapshot, dataset storage quota, and
per-API-key usage counters. Connection/timeout failures are wrapped into
``APIError`` by the shared ``_request`` transport, so this mixin only maps the
response body (mirroring the sync helper's empty-body and non-JSON fallbacks).
"""

from __future__ import annotations

from dagnam._core.aio.base import BaseAsyncDagnamClient
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_generic,
    response_json_value,
)
from dagnam._types import JsonObject, JsonValue, ensure_json_object


class AsyncAccountMixin(BaseAsyncDagnamClient):
    """Async Account, entitlement, and usage methods for AsyncDagnamClient."""

    async def _account_get(self, path: str) -> JsonValue | str | None:
        resp = await self._request("GET", path)
        raise_for_generic(resp)
        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ValueError:
            return resp.text

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
