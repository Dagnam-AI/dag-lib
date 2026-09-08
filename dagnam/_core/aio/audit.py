"""Async workload-audit publish client methods.

Async mirror of ``dagnam._core.client.audit.AuditClientMixin``: the same six
routes, the same bodies, the same uniform 404 for a key without the ``write``
scope. The shared ``_request`` transport wraps connect/timeout failures into
``APIError``, so this mixin only maps the response body.
"""

from __future__ import annotations

from dagnam._core.aio.base import BaseAsyncDagnamClient
from dagnam._core.client.audit import AUDITS_PATH
from dagnam._core.client.common import quote_path_segment, raise_for_generic
from dagnam._types import JsonObject, ensure_json_object


class AsyncAuditMixin(BaseAsyncDagnamClient):
    """Async workload-audit publish methods for AsyncDagnamClient."""

    async def _audit_req(
        self, method: str, path: str, json_body: JsonObject | None = None
    ) -> JsonObject:
        resp = await self._request(method, path, json=json_body, raise_for=raise_for_generic)
        return ensure_json_object(resp.json())

    async def create_audit(self, payload: JsonObject) -> JsonObject:
        """``POST /api/v1/audits``: the scan header and every workload it found."""
        return await self._audit_req("POST", AUDITS_PATH, payload)

    async def create_audit_candidate(self, audit_id: str, payload: JsonObject) -> JsonObject:
        """``POST /api/v1/audits/{id}/candidates`` (idempotent per workload x kind)."""
        return await self._audit_req(
            "POST", f"{AUDITS_PATH}/{quote_path_segment(audit_id)}/candidates", payload
        )

    async def patch_audit_candidate(
        self, audit_id: str, candidate_id: str, payload: JsonObject
    ) -> JsonObject:
        """``PATCH /api/v1/audits/{id}/candidates/{cid}``: record one step of a candidate."""
        return await self._audit_req(
            "PATCH",
            (
                f"{AUDITS_PATH}/{quote_path_segment(audit_id)}"
                f"/candidates/{quote_path_segment(candidate_id)}"
            ),
            payload,
        )

    async def halt_audit(self, audit_id: str, reason: str) -> JsonObject:
        """``POST /api/v1/audits/{id}/halt``: the run stopped short, and why."""
        return await self._audit_req(
            "POST", f"{AUDITS_PATH}/{quote_path_segment(audit_id)}/halt", {"reason": reason}
        )

    async def cancel_audit(self, audit_id: str) -> JsonObject:
        """``POST /api/v1/audits/{id}/cancel``: pause deployments, cancel runs, delete nothing."""
        return await self._audit_req("POST", f"{AUDITS_PATH}/{quote_path_segment(audit_id)}/cancel")

    async def delete_audit(self, audit_id: str) -> JsonObject:
        """``DELETE /api/v1/audits/{id}``: delete every artifact it published, with a receipt."""
        return await self._audit_req("DELETE", f"{AUDITS_PATH}/{quote_path_segment(audit_id)}")


__all__ = ["AsyncAuditMixin"]
