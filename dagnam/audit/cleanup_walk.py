"""The walks this client makes itself: an unpublished audit's ids, and what a claim refused.

A published audit's resources belong to the platform's own cancel and delete; the only ids this
client walks are those of an audit with no platform record, and those the platform refused to
claim, because this directory made them before the audit existed. Each walk stops or deletes an
id, reads it back, and records a row. Three rules keep it from destroying what is not this
directory's: an id the walk or the platform marked kept (``kept_ids``) is never touched; a dataset
that a project this directory did not create uses is kept (``in_use_elsewhere``); and a walk in
which this key is not PROVEN to see the account proves nothing: not-found is what a key from
another account is told, and a walk that also lost an id to a 5xx cannot tell the two apart. A
key is proven by its own earlier confirmation of a deletion (same host and key), by a deletion or
a stop it just got a positive answer to, or by a read of a recorded id that came back with a row;
the caller's word that it is gone (``--already-deleted``) stands in for the proof. An unproven walk
changes nothing local and marks nothing.
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

from dagnam_contracts.audit import CANCELLED_SCHEMA

from dagnam._core.exceptions import DagnamError, DatasetNotFoundError
from dagnam.audit.cleanup_kinds import (
    IN_USE_ELSEWHERE,
    KINDS,
    STOPS,
    WEIGHTS_SERVED,
    CleanupClient,
    can_read,
    datasets_in_other_projects,
    delete_one,
)
from dagnam.audit.cleanup_local import (
    SCHEMA,
    forget_locally,
    live,
    now_iso,
    recorded_ids,
    settle_cancel,
    unanswered,
    write_receipt,
)
from dagnam.audit.receipt_rows import (
    ALREADY_ABSENT,
    DELETED,
    KEPT,
    NOT_ANSWERED,
    NOT_CREATED_HERE,
    STOPPED,
    Verdict,
    blocked,
    decide,
)
from dagnam.audit.state import AuditState

PROJECT_HELD = "project_held"
"""The ``code`` of the project row this client writes while something in it was not removed."""


def proven(
    client: CleanupClient,
    state: AuditState,
    rows: list[dict[str, Any]],
    ids: dict[str, list[str]],
) -> bool:
    """Whether this walk's answers come from a key that sees the account the ids live in.

    Proof is a positive answer, never an absence: this very key (host and key) confirmed a
    deletion before; or a row says a delete or a stop was answered with success; or a recorded id
    that is not reported gone reads back with a row. Everything else -- every id not found,
    some not found and the rest a 5xx or a timeout -- is what another account's key is told too.
    """
    if state.confirmed_gone and state.confirmed_by == client.identity:
        return True
    if any(_answered_positively(r) for r in rows):
        return True
    unresolved = {(str(r.get("kind")), str(r.get("id"))) for r in rows} - {
        (str(r.get("kind")), str(r.get("id"))) for r in rows if r.get("status") == ALREADY_ABSENT
    }
    return any(
        can_read(client, kind, item_id)
        for kind, found in ids.items()
        for item_id in found
        if (kind, item_id) in unresolved
    )


def _answered_positively(row: dict[str, Any]) -> bool:
    """A row only an account that holds the id can have: a delete or stop that succeeded, or a keep.

    A keep is the platform's decision about a resource it knows (a version a live endpoint
    serves) or one read in another of the owner's projects. ``project_held`` is this client's own.
    """
    status = row.get("status")
    return status in (DELETED, STOPPED) or (status == KEPT and row.get("code") != PROJECT_HELD)


def none_visible(count: int, verb: str) -> dict[str, Any]:
    """The receipt of a walk that could not show its key sees the account: nothing was touched."""
    why = (
        f"this key could not be shown to see any of the {count} ids this directory recorded"
        " (they are all not found, or the rest could not be read): another account or host?"
        " If you know they are gone, `dagnam audit delete <dir> --already-deleted` clears this"
        " directory"
    )
    return unanswered(None, verb, why)


def cancel_unpublished(
    state: AuditState, client: CleanupClient, *, only: Collection[str] | None = None
) -> dict[str, Any]:
    """Stop every live run and endpoint the state records (those in ``only``, when given); the receipt.

    Every stop is tried and every outcome is a row, so one that fails never keeps the next from
    being tried. Only what stopped or was found gone is marked; a run that had already finished
    answers ``already_stopped`` and stays resumable. An id marked kept is not stopped, and when
    every id answers not-found nothing is marked (see the module docstring).
    """
    targets = [
        (kind, item_id)
        for step in state.all_steps()
        for kind, item_id in live(step)
        if item_id not in state.kept_ids and (only is None or item_id in only)
    ]
    entries = [STOPS[kind](client, item_id) for kind, item_id in targets]
    ids: dict[str, list[str]] = {kind: [] for kind in STOPS}
    for kind, item_id in targets:
        ids[kind].append(item_id)
    if only is None and entries and not proven(client, state, entries, ids):
        return none_visible(len(entries), "cancel")
    settle_cancel(state, {"entries": entries})
    return {"schema": CANCELLED_SCHEMA, "deleted_at": now_iso(), "entries": entries}


class _Links:
    """Where the owner's datasets are linked, read once and only when a walk reaches a dataset."""

    def __init__(self, client: CleanupClient, own_project: str | None) -> None:
        self._client, self._own = client, own_project
        self._found: set[str] | None = None

    def elsewhere(self, dataset_id: str) -> bool:
        """Whether a project this directory did not create uses ``dataset_id`` (raises if unread)."""
        if self._found is None:
            self._found = datasets_in_other_projects(self._client, self._own)
        return dataset_id in self._found


def _kept(kind: str, item_id: str, code: str, reason: str) -> dict[str, Any]:
    return {"kind": kind, "id": item_id, "status": KEPT, "code": code, "reason": reason}


def walk(
    client: CleanupClient, ids: dict[str, list[str]], own_project: str | None
) -> list[dict[str, Any]]:
    """Each id deleted and re-read, children first.

    A dataset another project uses is kept, not deleted; one whose links cannot be read is a
    ``blocked`` row. The project stays while anything in it was not removed, or weights an
    endpoint of the owner's still serves were kept.
    """
    rows: list[dict[str, Any]] = []
    links = _Links(client, own_project)
    for kind, delete, get, absent in KINDS:
        for item_id in ids[kind]:
            if kind == "project" and any(_holds_project(r) for r in rows):
                reason = "left in place: something in it was not removed or is still in use"
                rows.append(_kept(kind, item_id, PROJECT_HELD, reason))
            elif kind == "dataset" and (row := _dataset_guard(client, links, item_id)) is not None:
                rows.append(row)
            else:
                rows.append(delete_one(client, kind, delete, get, absent, item_id))
    return rows


def _holds_project(row: dict[str, Any]) -> bool:
    return decide(row).verdict is Verdict.LEFT or row.get("code") == WEIGHTS_SERVED


def _dataset_guard(client: CleanupClient, links: _Links, item_id: str) -> dict[str, Any] | None:
    """The row that keeps a dataset another project uses, or says its links could not be read.

    A dataset that is not there needs no link check: when the links cannot be read the dataset
    itself is read, and one that is gone is ``already_absent`` (what another account's key, or a
    host that answers not-found to everything, is told too), never a reason to guess.
    """
    try:
        if not links.elsewhere(item_id):
            return None
    except (DagnamError, KeyError, TypeError) as exc:
        try:
            client.get_dataset_meta(item_id)
        except DatasetNotFoundError:
            return {"kind": "dataset", "id": item_id, "status": ALREADY_ABSENT}
        except (DagnamError, KeyError, TypeError):
            pass  # unreadable too: the links error below is the reason
        reason = f"could not check whether another project uses it: {exc}"
        return blocked("dataset", item_id, reason)
    reason = "linked to a project this directory did not create; left alone"
    return _kept("dataset", item_id, IN_USE_ELSEWHERE, reason)


def remember(state: AuditState, rows: list[dict[str, Any]], identity: str) -> None:
    """Record what a walk saw: ids it confirmed deleted, and datasets it left to their other users.

    What was confirmed is scoped to the identity that saw it. Only the keep is final (never
    touched again). A project kept for what is still in it is
    asked again, so it is not recorded.
    """
    gone = {str(r.get("id")) for r in rows if r.get("status") == DELETED}
    kept = {str(r.get("id")) for r in rows if r.get("code") == IN_USE_ELSEWHERE}
    if gone:
        if state.confirmed_by != identity:  # another key's record is no proof for this one
            state.confirmed_gone = []
        state.confirmed_by = identity
        state.confirmed_gone += sorted(gone - set(state.confirmed_gone))
    state.kept_ids += sorted(kept - set(state.kept_ids))


def delete_unpublished(
    audit_dir: Path,
    client: CleanupClient,
    state: AuditState,
    *,
    assume_gone: bool = False,
) -> dict[str, Any]:
    """Delete every recorded id of an unpublished audit, write ``deleted.json``, then local files.

    Each id is deleted and re-read expecting not-found; a refusal, a failure, or a delete that
    still reads back is that id's ``blocked`` row, never the end of the walk. A registry version
    goes through its own purge route (``blocked [platform_only]`` when the platform has none; a
    version the owner serves from another endpoint is kept), never through the registry entry
    around it. The local rows and keys go unless an endpoint is still up; the state is marked
    deleted only when nothing is left. When EVERY id answers not-found the walk changes nothing
    local (see the module docstring) unless ``assume_gone``.
    """
    ids = recorded_ids(state)
    rows = walk(client, ids, state.project_id)
    if rows and not assume_gone and not proven(client, state, rows, ids):
        receipt = none_visible(len(rows), "delete")
        write_receipt(audit_dir, receipt)
        return receipt
    remember(state, rows, client.identity)
    receipt: dict[str, Any] = {"schema": SCHEMA, "deleted_at": now_iso(), "entries": rows}
    write_receipt(audit_dir, receipt)
    left = [r for r in rows if decide(r).verdict is Verdict.LEFT]
    up = any(r.get("kind") == "deployment" for r in left)
    unanswered_id = any(r.get("code") == NOT_ANSWERED for r in rows)
    local = forget_locally(audit_dir, state, keep_keys=up or unanswered_id, settled=not left)
    receipt = {**receipt, "audit_status": "halted" if left or local else "deleted"}
    if local:
        receipt["entries"] = [*rows, *local]
    write_receipt(audit_dir, receipt)
    return receipt


def unclaimed(state: AuditState) -> set[str]:
    """The ids the platform does not own for this audit: refused a claim, or never asked.

    A claim that was never answered (``claim_pending``) leaves everything an earlier unpublished
    run recorded in the same position as a refusal: this directory made it, the platform did not.
    An id marked kept is never among them.
    """
    if state.claim_pending:
        every = recorded_ids(state)
        mine = {i for kind, found in every.items() if kind != "project" for i in found}
    else:
        mine = set(state.unclaimed_ids)
    return mine - set(state.kept_ids)


def unclaimed_to_delete(state: AuditState, rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """The ids the platform did not claim that are still this directory's to delete.

    Not the project; not an id the platform's own receipt has a row for under any code but
    "recorded, not created here": it answered for it, so it owns it (a resource the owner moved
    or linked is theirs, and what it deleted or found gone is settled -- a claim whose answer
    was lost leaves both in the unclaimed set).
    """
    owned = {str(r.get("id")) for r in rows if r.get("code") != NOT_CREATED_HERE}
    mine = unclaimed(state) - owned
    return {
        kind: [] if kind == "project" else [i for i in found if i in mine]
        for kind, found in recorded_ids(state).items()
    }


__all__ = [
    "cancel_unpublished",
    "delete_unpublished",
    "none_visible",
    "proven",
    "remember",
    "unclaimed",
    "unclaimed_to_delete",
    "walk",
]
