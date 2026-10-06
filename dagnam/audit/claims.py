"""Hand an earlier, unpublished run's ids to the audit that is publishing them.

A ``--local-only`` run creates its resources with no audit to tag them. When a later run on the
same directory publishes, the audit is created first and asked to take every id the state already
records (``POST /audits/{id}/claims``, kinds ``dataset``, ``training_job``, ``deployment`` and
``project``, at most :data:`BATCH` per request; versions inherit from their run and are never
sent). The platform answers one ``claimed`` or ``refused`` per entry, in order.

A refusal is not a verdict that the resource is not ours: this directory created it. It is
recorded as unclaimed (:attr:`~dagnam.audit.state.AuditState.unclaimed_ids`, with the version its
run pushed) and ``audit delete`` and ``audit cancel`` handle it themselves, as they would for an
unpublished audit. One refused with the code ``in_use_elsewhere`` (another of the owner's runs
uses it) is the owner's, not this directory's: it is recorded as kept
(:attr:`~dagnam.audit.state.AuditState.kept_ids`), shown, and never stopped or deleted from here.
A request that fails as a whole (no answer, a 5xx, a 4xx, an answer with no list
of results) records NOTHING, raises :class:`ClaimError`, and the run halts: the claim is asked again
the next time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from dagnam._core.exceptions import DagnamError
from dagnam._types import JsonArray, JsonObject
from dagnam.audit.cleanup_kinds import IN_USE_ELSEWHERE
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


def claim_halt_detail(exc: ClaimError) -> str:
    """The halt's text for a claim that failed: what is, and is not, still in the account."""
    return (
        f"the claim for the earlier local-only run's resources failed: {exc}. Nothing new was"
        " uploaded, but what that run made (datasets, runs, endpoints) still exists and an"
        " endpoint may be serving: run `dagnam audit run` again to hand it over, or"
        " `dagnam audit cancel` to stop it"
    )


def _results(client: ClaimClient, audit_id: str, entries: JsonArray) -> dict[str, tuple[str, str]]:
    """``id -> (result, code)`` for every entry, or :class:`ClaimError` if any request fails."""
    answered: dict[str, tuple[str, str]] = {}
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
                answered[str(row.get("id"))] = (str(row.get("result")), str(row.get("code")))
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
    refused: set[str] = set()
    kept: set[str] = set()
    for e in entries:
        if not isinstance(e, dict) or e["kind"] == "project":
            continue
        result, code = answered.get(str(e["id"]), ("", ""))
        if result != CLAIMED:
            (kept if code == IN_USE_ELSEWHERE else refused).add(str(e["id"]))
    for step in state.all_steps():
        for ids in (refused, kept):
            if step.training_job_id in ids and step.model_version_id is not None:
                ids.add(step.model_version_id)  # its weights follow its run
    state.unclaimed_ids += sorted(refused - set(state.unclaimed_ids))
    state.kept_ids += sorted(kept - set(state.kept_ids))
    state.claim_pending = False
    if kept:
        count = len(kept)
        say(
            f"{count} resource{'s' if count != 1 else ''} the earlier local-only run made"
            " is in use by something else of yours; it is left alone and"
            " `dagnam audit delete` and `dagnam audit cancel` never touch it"
        )
    if refused:
        count = len(refused)
        say(
            f"{count} resource{'s' if count != 1 else ''} the earlier local-only run made could"
            " not be handed to this audit; they stay yours, and `dagnam audit delete` and"
            " `dagnam audit cancel` handle them directly"
        )


__all__ = [
    "BATCH",
    "CLAIMABLE",
    "CLAIMED",
    "IN_USE_ELSEWHERE",
    "ClaimClient",
    "ClaimError",
    "claim_recorded",
]
