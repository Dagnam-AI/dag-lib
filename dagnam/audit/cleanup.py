"""``audit cancel`` and ``audit delete``: what each does to a published or an unpublished audit.

**A published audit** (``state.json`` holds an ``audit_id``): the platform's cancel and delete
routes are the only paths that touch its resources. This module only RECORDS what the
platform's receipt says (:func:`settle_cancel`, :func:`settle_delete`) and removes this
machine's own files, so it cannot disagree with the platform about what is the audit's. When the
platform does not answer, or answers 404 (it cannot see the audit with this key: another account,
another host), nothing is touched and the receipt says so (:func:`unanswered`); only the
platform's own positive answer marks anything deleted. The one exception is the ids the platform
refused to CLAIM (``AuditState.unclaimed_ids``): this directory made them before the audit
existed, the platform never owned them, and they are stopped and deleted here as an unpublished
audit's are.

**An unpublished audit** (``audit_id`` is ``None``: ``--local-only``, or a run that never
reached ``create_audit``): nothing on the platform knows of it, so the ids this chain of runs
recorded as created are walked directly (:func:`delete_unpublished`,
:func:`cancel_unpublished`), each as delete, re-read expecting not-found, record. A walk in
which EVERY id answers not-found proves nothing (the same answer a key from another account
gets), so it changes nothing local and says so, unless the caller says the audit is gone.

What a receipt row means is decided by :func:`~dagnam.audit.receipt_rows.decide` and the exit
status by :func:`~dagnam.audit.receipt_rows.exit_status`, and nowhere else.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import time
from typing import Any

from dagnam_contracts.audit import CANCELLED_SCHEMA

from dagnam._core._retry import parse_retry_after
from dagnam._core.exceptions import APIError, DagnamError, TeardownInProgressError
from dagnam.audit.cleanup_kinds import KINDS, STOPS, CleanupClient, delete_one
from dagnam.audit.cleanup_local import (
    CANCELLED_ERROR,
    CANCELLED_FILE,
    DELETED_FILE,
    DEPLOY_DELETED,
    DEPLOY_PAUSED,
    GONE,
    LOCAL_KEYS,
    LOCAL_WORKLOADS,
    SCHEMA,
    forget_locally,
    live,
    mark_cancelled,
    receipt_rows,
    recorded_ids,
    write_receipt,
)
from dagnam.audit.receipt_rows import (
    ALREADY_ABSENT,
    KEPT,
    NOT_ANSWERED,
    STOPPED,
    Verdict,
    blocked,
    decide,
)
from dagnam.audit.state import DELETED_STATE, AuditDeletedError, AuditState, load_state, save_state

NOT_FOUND = 404
TEARDOWN_WAIT = 120.0
"""How long a cancel or delete waits for another walk of the same audit before it gives up."""
TEARDOWN_POLL = 5.0
"""The pause between two asks when the platform sends no ``Retry-After``."""
NOT_A_RECEIPT = "the answer was not a receipt"
NO_SUCH_AUDIT = "the platform has no audit with this id for this key"
NOT_CREATED_HERE = "not_created_here"


@dataclass(frozen=True, slots=True)
class Answer:
    """The platform's answer to a cancel or a delete: a receipt, or why there is none."""

    receipt: dict[str, Any] | None = None
    failure: str | None = None
    missing: bool = False
    """The answer was a 404: this key cannot see the audit. It decides nothing."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def unanswered(audit_id: str | None, verb: str, why: str) -> dict[str, Any]:
    """The receipt of a ``cancel`` or ``delete`` the platform did not answer: nothing was touched.

    One ``audit`` row, ``blocked [not_answered]``; no remote call was made and no local file is
    removed, so running the command again is exactly as safe as the first time.
    """
    reason = f"the platform did not answer the {verb} ({why}); nothing was touched: run it again"
    return {
        "schema": CANCELLED_SCHEMA if verb == "cancel" else SCHEMA,
        "deleted_at": _now(),
        "entries": [blocked("audit", audit_id, reason, NOT_ANSWERED)],
    }


def deleted_elsewhere(verb: str) -> dict[str, Any]:
    """The receipt of a delete the caller says is already done (``--already-deleted``).

    Nothing remote is touched, and the receipt has no rows, because the caller knows of none.
    """
    return {
        "schema": CANCELLED_SCHEMA if verb == "cancel" else SCHEMA,
        "deleted_at": _now(),
        "audit_status": "deleted",
        "entries": [],
    }


def ask_platform(
    call: Callable[[str], dict[str, Any]],
    audit_id: str,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> Answer:
    """The platform's own cancel or delete of a published audit, asked once.

    A 404 is :attr:`Answer.missing`, not a decision. A 409 saying another walk holds the audit is
    waited out (the platform's ``Retry-After``, else :data:`TEARDOWN_POLL`) up to
    :data:`TEARDOWN_WAIT`, then asked again; past that it is a failure to answer. A 200 that is
    not a receipt (no list of rows: a proxy, another host) is a failure too, never "nothing is
    left": an empty list is a receipt, a missing one is not.
    """
    waited = 0.0
    while True:
        try:
            receipt = call(audit_id)
            break
        except TeardownInProgressError as exc:
            pause = parse_retry_after(exc.retry_after_header, cap=TEARDOWN_POLL) or TEARDOWN_POLL
            if waited + pause > TEARDOWN_WAIT:
                return Answer(
                    failure=f"another cancel or delete of this audit is still running: {exc}"
                )
            sleep(pause)
            waited += pause
        except DagnamError as exc:
            if isinstance(exc, APIError) and exc.status_code == NOT_FOUND:
                return Answer(failure=NO_SUCH_AUDIT, missing=True)
            return Answer(failure=str(exc))
    if not any(isinstance(receipt.get(key), list) for key in ("items", "entries")):
        return Answer(failure=NOT_A_RECEIPT)
    return Answer(receipt=receipt)


def _invisible(rows: list[dict[str, Any]]) -> bool:
    """Every id answered not-found: what a key from another account is told for all of them."""
    return bool(rows) and all(row.get("status") == ALREADY_ABSENT for row in rows)


def _none_visible(count: int, verb: str) -> dict[str, Any]:
    reason = f"none of the {count} ids this directory recorded is visible to this key"
    return unanswered(None, verb, f"{reason}: another account or host?")


def settle_cancel(state: AuditState, receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    """Record a published cancel's receipt in the state, and return it untouched.

    Only a row that says it stopped something marks the local run or endpoint: ``stopped``
    (paused or cancelled) and ``deleted`` / ``already_absent`` (gone). ``already_stopped``,
    ``kept`` and ``blocked`` mark nothing, so a run that finished stays resumable.
    """
    rows = receipt_rows(receipt)
    mark_cancelled(
        state,
        {str(r.get("id")) for r in rows if r.get("status") == STOPPED and decide(r).marks},
        {str(r.get("id")) for r in rows if r.get("status") in GONE},
    )
    return receipt


def cancel_unpublished(
    state: AuditState, client: CleanupClient, *, only: Collection[str] | None = None
) -> dict[str, Any]:
    """Stop every live run and endpoint the state records (those in ``only``, when given); the receipt.

    Every stop is tried and every outcome is a row, so one that fails never keeps the next from
    being tried. Only what stopped or was found gone is marked; a run that had already finished
    answers ``already_stopped`` and stays resumable. When every id answers not-found nothing is
    marked: a key from another account is told the same.
    """
    entries = [
        STOPS[kind](client, item_id)
        for step in state.all_steps()
        for kind, item_id in live(step)
        if only is None or item_id in only
    ]
    if only is None and _invisible(entries):
        return _none_visible(len(entries), "cancel")
    settle_cancel(state, {"entries": entries})
    return {"schema": CANCELLED_SCHEMA, "deleted_at": _now(), "entries": entries}


def _walk(client: CleanupClient, ids: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Each id deleted and re-read, children first; the project only if nothing else is left."""
    rows: list[dict[str, Any]] = []
    for kind, delete, get, absent in KINDS:
        for item_id in ids[kind]:
            if kind == "project" and any(decide(r).verdict is Verdict.LEFT for r in rows):
                reason = "left in place: something in it was not removed"
                rows.append(
                    {
                        "kind": kind,
                        "id": item_id,
                        "status": KEPT,
                        "code": "project_held",
                        "reason": reason,
                    }
                )
            else:
                rows.append(delete_one(client, kind, delete, get, absent, item_id))
    return rows


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
    goes through its own purge route (``blocked [platform_only]`` when the platform has none),
    never through the registry entry around it, and the project stays while anything in it does.
    The local rows and keys go unless an endpoint is still up; the state is marked deleted only
    when nothing is left. When EVERY id answers not-found the walk changes nothing local (see
    the module docstring) unless ``assume_gone``.
    """
    rows = _walk(client, recorded_ids(state))
    if _invisible(rows) and not assume_gone:
        receipt = _none_visible(len(rows), "delete")
        write_receipt(audit_dir, receipt)
        return receipt
    receipt: dict[str, Any] = {"schema": SCHEMA, "deleted_at": _now(), "entries": rows}
    write_receipt(audit_dir, receipt)
    left = [r for r in rows if decide(r).verdict is Verdict.LEFT]
    up = any(r.get("kind") == "deployment" for r in left)
    local = forget_locally(audit_dir, state, keep_keys=up, settled=not left)
    receipt = {**receipt, "audit_status": "halted" if left or local else "deleted"}
    if local:
        receipt["entries"] = [*rows, *local]
    write_receipt(audit_dir, receipt)
    return receipt


def unclaimed(state: AuditState) -> set[str]:
    """The ids the platform does not own for this audit: refused a claim, or never asked.

    A claim that was never answered (``claim_pending``) leaves everything an earlier unpublished
    run recorded in the same position as a refusal: this directory made it, the platform did not.
    """
    if state.claim_pending:
        every = recorded_ids(state)
        return {i for kind, found in every.items() if kind != "project" for i in found}
    return set(state.unclaimed_ids)


def _unclaimed(state: AuditState, rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """The ids the platform refused to claim that are still this directory's to delete.

    Not the project, and not an id the platform's own receipt kept for any reason but "recorded,
    not created here": a resource the owner moved or linked is theirs, whatever a claim said.
    """
    held = {
        str(r.get("id"))
        for r in rows
        if r.get("status") == KEPT and r.get("code") != NOT_CREATED_HERE
    }
    mine = unclaimed(state) - held
    return {
        kind: [] if kind == "project" else [i for i in found if i in mine]
        for kind, found in recorded_ids(state).items()
    }


def settle_delete(
    audit_dir: Path, state: AuditState, receipt: Mapping[str, Any], client: CleanupClient
) -> dict[str, Any]:
    """Record a published delete's receipt: write it as sent, and remove local files iff it deleted.

    The platform says whether it deleted the audit (``audit_status``); one that sends none is the
    legacy shape, deleted when no row is left, and one that sends a status this client does not
    know is not deleted. ``halted`` means the platform is not done: NOTHING local is removed,
    and the command exits 1 so the next call finishes it. Once it did delete, what it refused
    to claim is deleted here (:func:`_unclaimed`) and joins the receipt. The ids the platform
    kept are recorded in the state so no listing offers them as the audit's. A deployment left
    behind keeps the keys that call it; local files that cannot be removed (a link) are rows of
    their own on the written receipt.
    """
    rows = receipt_rows(receipt)
    verdicts = {decide(row).verdict for row in rows}
    reported = receipt.get("audit_status")
    deleted = reported == "deleted" or (
        reported is None and not verdicts & {Verdict.LEFT, Verdict.UNKNOWN}
    )
    write_receipt(audit_dir, receipt)
    extra: list[dict[str, Any]] = []
    walked: set[str] = set()
    extra: list[dict[str, Any]] = []
    if deleted:
        mine = _unclaimed(state, rows)
        walked = {i for found in mine.values() for i in found}
        extra = _walk(client, mine)
    state.kept_ids += [
        str(r.get("id"))
        for r in rows
        if r.get("status") == KEPT
        and str(r.get("id")) not in state.kept_ids
        and str(r.get("id")) not in walked
    ]
    if not deleted:
        save_state(audit_dir, state)
        return dict(receipt)
    left = [r for r in [*rows, *extra] if decide(r).verdict is Verdict.LEFT]
    up = any(r.get("kind") == "deployment" for r in left)
    local = forget_locally(audit_dir, state, keep_keys=up, settled=not left)
    full = {**receipt, "entries": [*rows, *extra, *local]}
    if extra or local:
        write_receipt(audit_dir, full)
    return full


def cancel_audit(
    audit_dir: Path,
    client: CleanupClient,
    *,
    on_missing: Callable[[], None] = lambda: None,
    sleep: Callable[[float], None] = time.sleep,
) -> Mapping[str, Any]:
    """Cancel the audit in ``audit_dir`` (the caller holds its lock); write ``cancelled.json``.

    Published: one call to the platform's cancel, its receipt recorded and written as sent; what
    the platform refused to claim is stopped here. A 404 or a failure leaves everything as it
    was, with an ``audit blocked [not_answered]`` row (``on_missing`` is called on a 404, for the
    caller to say whose account that was). Unpublished: the recorded runs and endpoints are
    stopped here.

    Raises:
        AuditDeletedError: the directory holds a deleted audit.
    """
    state = load_state(audit_dir)
    if state.halted == DELETED_STATE:
        raise AuditDeletedError(f"{audit_dir} is a deleted audit: there is nothing to cancel")
    receipt: Mapping[str, Any]
    if state.audit_id is None:
        receipt = cancel_unpublished(state, client)
    else:
        answer = ask_platform(client.cancel_audit, state.audit_id, sleep=sleep)
        if answer.receipt is None:
            if answer.missing:
                on_missing()
            receipt = unanswered(state.audit_id, "cancel", answer.failure or NOT_A_RECEIPT)
        else:
            receipt = settle_cancel(state, answer.receipt)
            if mine := unclaimed(state):
                extra = cancel_unpublished(state, client, only=mine)
                receipt = {**receipt, "entries": [*receipt_rows(receipt), *receipt_rows(extra)]}
    write_receipt(audit_dir, receipt, CANCELLED_FILE)
    save_state(audit_dir, state)
    return receipt


def delete_audit(
    audit_dir: Path,
    client: CleanupClient,
    *,
    on_missing: Callable[[], None] = lambda: None,
    assume_gone: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Delete the audit in ``audit_dir`` (the caller holds its lock); write ``deleted.json``.

    Published: one call to the platform's delete, whose receipt is written as sent; this
    machine's rows and keys are removed only when the platform says it deleted the audit. A 404
    or a failure touches nothing and says so (a 404 means only that this key cannot see the audit:
    ``assume_gone`` is the caller's word that it is deleted, and then the local files go). Never a
    direct call but for what the platform refused to claim. Unpublished: the recorded ids are
    deleted here. A directory whose audit is already deleted is not asked about again: only local
    leftovers are retried.
    """
    state = load_state(audit_dir)
    if state.halted == DELETED_STATE:
        receipt = {**deleted_elsewhere("delete"), "entries": forget_locally(audit_dir, state)}
        write_receipt(audit_dir, receipt)
        return receipt
    if state.audit_id is None:
        return delete_unpublished(audit_dir, client, state, assume_gone=assume_gone)
    answer = ask_platform(client.delete_audit, state.audit_id, sleep=sleep)
    if answer.receipt is not None:
        return settle_delete(audit_dir, state, answer.receipt, client)
    if answer.missing:
        on_missing()
    if answer.missing and assume_gone:
        receipt = {**deleted_elsewhere("delete"), "entries": forget_locally(audit_dir, state)}
    else:
        receipt = unanswered(state.audit_id, "delete", answer.failure or NOT_A_RECEIPT)
    write_receipt(audit_dir, receipt)
    return receipt


__all__ = [
    "CANCELLED_ERROR",
    "CANCELLED_FILE",
    "DELETED_FILE",
    "DELETED_STATE",
    "DEPLOY_DELETED",
    "DEPLOY_PAUSED",
    "GONE",
    "LOCAL_KEYS",
    "LOCAL_WORKLOADS",
    "NO_SUCH_AUDIT",
    "SCHEMA",
    "TEARDOWN_POLL",
    "TEARDOWN_WAIT",
    "Answer",
    "AuditDeletedError",
    "ask_platform",
    "cancel_audit",
    "cancel_unpublished",
    "delete_audit",
    "delete_unpublished",
    "deleted_elsewhere",
    "forget_locally",
    "live",
    "mark_cancelled",
    "receipt_rows",
    "recorded_ids",
    "settle_cancel",
    "settle_delete",
    "unanswered",
    "unclaimed",
    "write_receipt",
]
