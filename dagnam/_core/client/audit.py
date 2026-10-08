"""Synchronous workload-audit publish client methods.

The routes ``dagnam audit run`` uses to mirror a local audit into the
account: the scan header with its workloads, a candidate per workload x kind,
one PATCH per step, the read and the resume a run starts with, and the three
that end a run -- halt, cancel, delete. Every
one needs an API key with the ``write`` scope; a key without it is refused with
a 403, and an audit belonging to somebody else is a 404.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dagnam._core._resume import confirmed_by, created_body, gone
from dagnam._core.client.base import ALLOW_REDIRECTS, DEFAULT_TIMEOUT, BaseDagnamClient
from dagnam._core.client.common import (
    UNPARSEABLE_BODY,
    quote_path_segment,
    raise_for_generic,
    response_json_object,
    short_error_text,
)
from dagnam._core.exceptions import EndpointsServingError, TeardownInProgressError
from dagnam._types import JsonArray, JsonObject

if TYPE_CHECKING:
    import requests

AUDITS_PATH = "/api/v1/audits"
WALK_TIMEOUT = (10.0, 300.0)
"""(connect, read) for the account's own cancel and delete: they walk every artifact in one request."""
BUILD_PATH = "/health/build"
"""Which build the platform runs; unversioned and unauthenticated, on the API base."""


TEARDOWN_BUSY = "teardown_in_progress"
"""The ``error`` a 409 carries while another walk of the same audit holds its lock."""
TEARDOWN_UNAVAILABLE = "teardown_unavailable"
"""The ``error`` a 503 carries when the platform cannot take the lock just now: ask again shortly."""
ENDPOINTS_SERVING = "endpoints_serving"
"""The ``error`` a 409 carries when the delete stopped because an endpoint of the audit is serving."""
SERVING_TEXT_MAX = 8192
"""The longest refusal sentence kept: it names up to five endpoints, whose names can be long."""
WAIT_MARKERS = {409: TEARDOWN_BUSY, 503: TEARDOWN_UNAVAILABLE}
"""Which ``error`` marks a wait-and-ask-again answer, by status."""


def raise_for_audit(resp: requests.Response) -> None:
    """``raise_for_generic``, except that the platform's refusals to go on are typed.

    The 409 "a teardown is running" and the 503 "the lock is unavailable" are "ask again
    shortly" answers; the 409 "endpoints are serving" is a refusal a person must act on. The
    backend's body is ``{"detail": "<words>", "error": "<marker>"}``; the marker is also read
    under a ``detail`` object, whichever a platform sends; any other 409 or 503 stays an
    ``APIError``.
    """
    if resp.status_code in (409, 503):
        try:
            data = resp.json()
        except UNPARSEABLE_BODY:
            data = None
        detail = data.get("detail") if isinstance(data, dict) else None
        source = detail if isinstance(detail, dict) else data
        marker = source.get("error") if isinstance(source, dict) else None
        if isinstance(source, dict) and marker == WAIT_MARKERS[resp.status_code]:
            words = source.get("message") or (detail if isinstance(detail, str) else None)
            raise TeardownInProgressError(
                resp.status_code,
                short_error_text(str(words or marker)),
                retry_after_header=resp.headers.get("Retry-After"),
            )
        if isinstance(source, dict) and resp.status_code == 409 and marker == ENDPOINTS_SERVING:
            words = source.get("message") or (detail if isinstance(detail, str) else None)
            listed = source.get("endpoints")
            raise EndpointsServingError(
                str(words or marker)[:SERVING_TEXT_MAX],
                [e for e in listed if isinstance(e, dict)] if isinstance(listed, list) else [],
            )
    raise_for_generic(resp)


class AuditClientMixin(BaseDagnamClient):
    """Workload-audit publish methods for DagnamClient."""

    def _audit_request(
        self,
        method: str,
        path: str,
        json_body: JsonObject | None = None,
        *,
        idempotent: bool = False,
        resumable: bool = False,
        resume_nonce: str | None = None,
        walk: bool = False,
        params: dict[str, str] | None = None,
    ) -> JsonObject:
        resp = self._request(
            method,
            f"{self.api_url}{path}",
            raise_for=raise_for_audit,
            json=json_body,
            params=params,
            timeout=WALK_TIMEOUT if walk else DEFAULT_TIMEOUT,
            allow_redirects=ALLOW_REDIRECTS,
            idempotent=idempotent,
            resumable=resumable,
            resume_salt=resume_nonce,
            confirm=confirmed_by(self.get_audit, gone()) if resumable else None,
            retry=not walk,
        )
        body = response_json_object(resp)
        # A replay that dropped its body points at the audit it made: read it.
        return created_body(resp, body, self.get_audit) if resumable else body

    def create_audit(self, payload: JsonObject, *, resume_nonce: str | None = None) -> JsonObject:
        """``POST /api/v1/audits``: the scan header and every workload it found.

        Sends an ``Idempotency-Key`` the platform deduplicates on, so a transient
        failure retries into the first answer instead of opening a second audit.
        Resumable (``resume_creates``): the body names the audit's own project,
        and ``resume_nonce`` -- a value the caller saved before asking -- keeps
        one directory's audit from replaying another's.
        """
        return self._audit_request(
            "POST", AUDITS_PATH, payload, idempotent=True, resumable=True, resume_nonce=resume_nonce
        )

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

    def get_audit(self, audit_id: str) -> JsonObject:
        """``GET /api/v1/audits/{id}``: the audit with its status; a uniform 404 once it is gone."""
        return self._audit_request("GET", f"{AUDITS_PATH}/{quote_path_segment(audit_id)}")

    def resume_audit(self, audit_id: str) -> JsonObject:
        """``POST /api/v1/audits/{id}/resume``: un-halt the audit for a run starting again."""
        return self._audit_request("POST", f"{AUDITS_PATH}/{quote_path_segment(audit_id)}/resume")

    def halt_audit(self, audit_id: str, reason: str) -> JsonObject:
        """``POST /api/v1/audits/{id}/halt``: the run stopped short, and why."""
        return self._audit_request(
            "POST", f"{AUDITS_PATH}/{quote_path_segment(audit_id)}/halt", {"reason": reason}
        )

    def cancel_audit(self, audit_id: str) -> JsonObject:
        """``POST /api/v1/audits/{id}/cancel``: pause deployments, cancel runs, delete nothing.

        The platform walks every artifact in this one request, so it gets a long
        read timeout and is sent once: a failure is the caller's to handle, and
        the walk is safe to ask for again.
        """
        return self._audit_request(
            "POST", f"{AUDITS_PATH}/{quote_path_segment(audit_id)}/cancel", walk=True
        )

    def delete_audit(self, audit_id: str, *, include_endpoints: bool = False) -> JsonObject:
        """``DELETE /api/v1/audits/{id}``: delete every artifact it published, with a receipt.

        Sent once, with a long read timeout (see :meth:`cancel_audit`); repeating it
        after a failure is what finishes a partial walk. While an endpoint of the audit is
        serving the platform deletes nothing and answers ``409 endpoints_serving``, raised as
        :class:`~dagnam.audit.EndpointsServingError`; ``include_endpoints=True`` deletes those
        endpoints too, and apps calling them start getting errors.
        """
        return self._audit_request(
            "DELETE",
            f"{AUDITS_PATH}/{quote_path_segment(audit_id)}",
            walk=True,
            params={"include_endpoints": "true"} if include_endpoints else None,
        )

    def claim_audit_resources(self, audit_id: str, entries: JsonArray) -> JsonObject:
        """``POST /api/v1/audits/{id}/claims``: ask the platform to take ids this caller created.

        ``entries`` are ``{"kind", "id"}`` (kinds ``dataset``, ``training_job``, ``deployment``,
        ``project``; at most 200) of resources made before the audit existed. The answer is
        ``{"results": [{"kind", "id", "result", "code"}]}`` in request order, ``result`` being
        ``claimed`` or ``refused``. A platform without the route answers 404.
        """
        return self._audit_request(
            "POST",
            f"{AUDITS_PATH}/{quote_path_segment(audit_id)}/claims",
            {"entries": entries},
        )

    def get_platform_build(self) -> JsonObject:
        """``GET /health/build``: the running build, with the contract version it installed.

        ``contracts`` is the platform's ``dagnam-contracts`` version; a platform
        that predates the key answers without it, and one that predates the
        route with a 404. ``dagnam audit run`` reads it before its first upload.
        """
        return self._audit_request("GET", BUILD_PATH)


__all__ = ["AUDITS_PATH", "BUILD_PATH", "AuditClientMixin"]
