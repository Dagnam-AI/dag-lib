"""Synchronous account / usage client methods.

Read-only access to the caller's plan, entitlement snapshot, storage quota, and
per-API-key usage counters. These are the building blocks behind
``dagnam.account.*`` and the ``dagnam usage`` CLI command.
"""

from __future__ import annotations

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
    response_json_value,
)
from dagnam._types import JsonObject, JsonValue


class AccountClientMixin(BaseDagnamClient):
    """Account, entitlement, and usage methods for DagnamClient."""

    def _account_get(self, path: str) -> JsonValue | str | None:
        url = f"{self.api_url}{path}"
        try:
            resp = requests.get(
                url,
                headers=self._headers(),
                timeout=DEFAULT_TIMEOUT,
                allow_redirects=ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_generic(resp)

        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ValueError:
            return resp.text

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
