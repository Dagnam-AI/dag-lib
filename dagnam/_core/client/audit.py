"""Synchronous workload-audit publish client methods.

The six routes ``dagnam audit run`` uses to mirror a local audit into the
account: the scan header with its workloads, a candidate per workload x kind,
one PATCH per step, and the three that end a run -- halt, cancel, delete. Every
one needs an API key with the ``write`` scope; a key without it, or an audit
belonging to somebody else, is the same uniform 404.
"""

from __future__ import annotations

from dagnam._core.client.base import ALLOW_REDIRECTS, DEFAULT_TIMEOUT, BaseDagnamClient
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_generic,
    response_json_object,
)
from dagnam._types import JsonObject

AUDITS_PATH = "/api/v1/audits"


class AuditClientMixin(BaseDagnamClient):
    """Workload-audit publish methods for DagnamClient."""

    def _audit_request(
        self, method: str, path: str, json_body: JsonObject | None = None
    ) -> JsonObject:
        resp = self._request(
            method,
            f"{self.api_url}{path}",
            raise_for=raise_for_generic,
            json=json_body,
            timeout=DEFAULT_TIMEOUT,
            allow_redirects=ALLOW_REDIRECTS,
        )
        return response_json_object(resp)

    def create_audit(self, payload: JsonObject) -> JsonObject:
        """``POST /api/v1/audits``: the scan header and every workload it found."""
        return self._audit_request("POST", AUDITS_PATH, payload)

    def create_audit_candidate(self, audit_id: str, payload: JsonObject) -> JsonObject:
        """``POST /api/v1/audits/{id}/candidates`` (idempotent per workload x kind)."""
        return self._audit_request(
            "POST", f"{AUDITS_PATH}/{quote_path_segment(audit_id)}/candidates", payload
        )

    def patch_audit_candidate(
        self, audit_id: str, candidate_id: str, payload: JsonObject
    ) -> JsonObject:
        """``PATCH /api/v1/audits/{id}/candidates/{cid}``: record one step of a candidate."""
        return self._audit_request(
            "PATCH",
            (
                f"{AUDITS_PATH}/{quote_path_segment(audit_id)}"
                f"/candidates/{quote_path_segment(candidate_id)}"
            ),
            payload,
        )

    def halt_audit(self, audit_id: str, reason: str) -> JsonObject:
        """``POST /api/v1/audits/{id}/halt``: the run stopped short, and why."""
        return self._audit_request(
            "POST", f"{AUDITS_PATH}/{quote_path_segment(audit_id)}/halt", {"reason": reason}
        )

    def cancel_audit(self, audit_id: str) -> JsonObject:
        """``POST /api/v1/audits/{id}/cancel``: pause deployments, cancel runs, delete nothing."""
        return self._audit_request("POST", f"{AUDITS_PATH}/{quote_path_segment(audit_id)}/cancel")

    def delete_audit(self, audit_id: str) -> JsonObject:
        """``DELETE /api/v1/audits/{id}``: delete every artifact it published, with a receipt."""
        return self._audit_request("DELETE", f"{AUDITS_PATH}/{quote_path_segment(audit_id)}")


__all__ = ["AUDITS_PATH", "AuditClientMixin"]
