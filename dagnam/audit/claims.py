"""Hand an earlier, unpublished run's ids to the audit that is publishing them.

A ``--local-only`` run creates its resources with no audit to tag them. When a later run on the
same directory publishes, the audit is created first and asked to take every id the state already
records (``POST /audits/{id}/claims``, kinds ``dataset``, ``training_job``, ``deployment`` and
``project``, at most :data:`BATCH` per request; versions inherit from their run and are never
sent). The platform answers one ``claimed`` or ``refused`` per entry, in order.

A refusal is not a verdict that the resource is not ours: this directory created it. It is
recorded as unclaimed (:attr:`~dagnam.audit.state.AuditState.unclaimed_ids`, with the version its
run pushed) and ``audit delete`` and ``audit cancel`` handle it themselves, as they would for an
unpublished audit. A request that fails as a whole (no answer, a 5xx, a 4xx, an answer with no list
of results) records NOTHING, raises :class:`ClaimError`, and the run halts: the claim is asked again
the next time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from dagnam._core.exceptions import DagnamError
from dagnam._types import JsonArray, JsonObject
from dagnam.audit.cleanup_local import recorded_ids
from dagnam.audit.state import AuditState

CLAIMABLE = ("dataset", "training_job", "deployment", "project")
"""The kinds the claim route takes; a registry version inherits its run's tag."""
BATCH = 200
"""Most entries one claim request may carry."""
CLAIMED = "claimed"


class ClaimClient(Protocol):
    """The one client method a claim needs."""

    def claim_audit_resources(self, audit_id: str, entries: JsonArray) -> JsonObject:
        """``POST /api/v1/audits/{id}/claims``."""
        ...


class ClaimError(DagnamError):
    """A claim request failed as a whole: nothing was recorded, and it is asked again."""


def _results(client: ClaimClient, audit_id: str, entries: JsonArray) -> dict[str, str]:
    """``id -> result`` for every entry, or :class:`ClaimError` if any request fails."""
    answered: dict[str, str] = {}
    for start in range(0, len(entries), BATCH):
        try:
            answer = client.claim_audit_resources(audit_id, entries[start : start + BATCH])
        except DagnamError as exc:
            raise ClaimError(str(exc)) from exc
        results = answer.get("results")
        if not isinstance(results, list):
            raise ClaimError("the answer had no list of results")
        for row in results:
            if isinstance(row, dict):
                answered[str(row.get("id"))] = str(row.get("result"))
    return answered


def claim_recorded(
    client: ClaimClient, state: AuditState, say: Callable[[str], None] = lambda _line: None
) -> None:
    """Ask the audit to take every id the state records; record what it will not as unclaimed.

    Raises:
        ClaimError: a request failed; the state is unchanged and ``claim_pending`` stays set.
    """
    ids = recorded_ids(state)
    entries: JsonArray = [{"kind": kind, "id": i} for kind in CLAIMABLE for i in ids[kind]]
    if state.audit_id is None or all(
        e["kind"] == "project" for e in entries if isinstance(e, dict)
    ):
        state.claim_pending = False  # a lone project is tagged by the audit's own create
        return
    answered = _results(client, state.audit_id, entries)
    refused = {
        str(e["id"])
        for e in entries
        if isinstance(e, dict) and answered.get(str(e["id"])) != CLAIMED and e["kind"] != "project"
    }
    for step in state.all_steps():
        if step.training_job_id in refused and step.model_version_id is not None:
            refused.add(step.model_version_id)  # its weights follow its run
    state.unclaimed_ids += sorted(refused - set(state.unclaimed_ids))
    state.claim_pending = False
    if refused:
        count = len(refused)
        say(
            f"{count} resource{'s' if count != 1 else ''} the earlier local-only run made could"
            " not be handed to this audit; they stay yours, and `dagnam audit delete` and"
            " `dagnam audit cancel` handle them directly"
        )


__all__ = ["BATCH", "CLAIMABLE", "CLAIMED", "ClaimClient", "ClaimError", "claim_recorded"]
