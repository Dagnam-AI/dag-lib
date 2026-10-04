"""Async workload-audit publish client methods.

Async mirror of ``dagnam._core.client.audit.AuditClientMixin``: the same
routes, the same bodies, the same uniform 404 for a key without the ``write``
scope. The shared ``_request`` transport wraps connect/timeout failures into
``APIError``, so this mixin only maps the response body.

It publishes; it does NOT tear an audit down. ``cancel_audit`` and ``delete_audit`` exist on the
sync client only, because a teardown is the platform's one walk plus what the caller does with
its receipt (the walk timeout, the single send, the local files, the exit status), all of which
live in the CLI and in :mod:`dagnam.audit.cleanup`. An async caller that wants one calls the sync
client; there is no second, subtly different copy of the rules here. Likewise no preflight and
no resumable creates: there is no async ``audit run``.
"""

from __future__ import annotations

from dagnam._core.aio.base import BaseAsyncDagnamClient
from dagnam._core.client.audit import AUDITS_PATH
from dagnam._core.client.common import quote_path_segment, raise_for_generic
from dagnam._types import JsonArray, JsonObject, ensure_json_object


class AsyncAuditMixin(BaseAsyncDagnamClient):
    """Async workload-audit publish methods for AsyncDagnamClient."""

    async def _audit_req(
        self,
        method: str,
        path: str,
        json_body: JsonObject | None = None,
        *,
        idempotent: bool = False,
    ) -> JsonObject:
        resp = await self._request(
            method, path, json=json_body, raise_for=raise_for_generic, idempotent=idempotent
        )
        return ensure_json_object(resp.json())

    async def create_audit(self, payload: JsonObject) -> JsonObject:
        """``POST /api/v1/audits`` with an ``Idempotency-Key`` (see the sync twin)."""
        return await self._audit_req("POST", AUDITS_PATH, payload, idempotent=True)

    async def claim_audit_resources(self, audit_id: str, entries: JsonArray) -> JsonObject:
        """``POST /api/v1/audits/{id}/claims`` (see the sync twin)."""
        return await self._audit_req(
            "POST", f"{AUDITS_PATH}/{quote_path_segment(audit_id)}/claims", {"entries": entries}
        )

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

    async def get_audit(self, audit_id: str) -> JsonObject:
        """``GET /api/v1/audits/{id}``: the audit with its status."""
        return await self._audit_req("GET", f"{AUDITS_PATH}/{quote_path_segment(audit_id)}")

    async def resume_audit(self, audit_id: str) -> JsonObject:
        """``POST /api/v1/audits/{id}/resume``: un-halt the audit."""
        return await self._audit_req("POST", f"{AUDITS_PATH}/{quote_path_segment(audit_id)}/resume")

    async def halt_audit(self, audit_id: str, reason: str) -> JsonObject:
        """``POST /api/v1/audits/{id}/halt``: the run stopped short, and why."""
        return await self._audit_req(
            "POST", f"{AUDITS_PATH}/{quote_path_segment(audit_id)}/halt", {"reason": reason}
        )


__all__ = ["AsyncAuditMixin"]
