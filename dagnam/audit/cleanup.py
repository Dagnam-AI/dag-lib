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
audit's are (:mod:`dagnam.audit.cleanup_walk`) -- except an id the platform said is in use
elsewhere, which is kept and never touched.

What a receipt row means is decided by :func:`~dagnam.audit.receipt_rows.decide` and the exit
status by :func:`~dagnam.audit.receipt_rows.exit_status`, and nowhere else.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from dagnam._core._retry import parse_retry_after
from dagnam._core.exceptions import APIError, DagnamError, TeardownInProgressError
from dagnam.audit.cleanup_kinds import CleanupClient
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
    deleted_elsewhere,
    forget_locally,
    live,
    mark_cancelled,
    receipt_rows,
    recorded_ids,
    settle_cancel,
    unanswered,
    write_receipt,
)
from dagnam.audit.cleanup_walk import (
    cancel_unpublished,
    delete_unpublished,
    remember,
    unclaimed,
    unclaimed_to_delete,
    walk,
)
from dagnam.audit.receipt_rows import KEPT, Verdict, decide
from dagnam.audit.state import DELETED_STATE, AuditDeletedError, AuditState, load_state, save_state

NOT_FOUND = 404
TEARDOWN_WAIT = 120.0
"""How long a cancel or delete waits for another walk of the same audit before it gives up."""
TEARDOWN_POLL = 5.0
"""The pause between two asks when the platform sends no usable ``Retry-After``."""
NOT_A_RECEIPT = "the answer was not a receipt"
NO_SUCH_AUDIT = "the platform has no audit with this id for this key"


@dataclass(frozen=True, slots=True)
class Answer:
    """The platform's answer to a cancel or a delete: a receipt, or why there is none."""

    receipt: dict[str, Any] | None = None
    failure: str | None = None
    missing: bool = False
    """The answer was a 404: this key cannot see the audit. It decides nothing."""


def _pause(exc: TeardownInProgressError) -> float:
    """How long the platform asked to wait: its ``Retry-After`` (any size up to the cap), else the poll."""
    asked = parse_retry_after(exc.retry_after_header, cap=TEARDOWN_WAIT)
    return asked if asked is not None and asked > 0 else TEARDOWN_POLL  # False for nan too


def ask_platform(
    call: Callable[[str], dict[str, Any]],
    audit_id: str,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> Answer:
    """The platform's own cancel or delete of a published audit, asked once.

    A 404 is :attr:`Answer.missing`, not a decision. A 409 saying another walk holds the audit
    (or a 503 saying its lock is unavailable, which also carries a ``Retry-After``) is waited
    out -- the platform's own pause, whatever its size, else :data:`TEARDOWN_POLL` -- up to
    :data:`TEARDOWN_WAIT` in all, then asked again; past that it is a failure to answer. A 200
    that is not a receipt (no list of rows: a proxy, another host) is a failure too, never
    "nothing is left": an empty list is a receipt, a missing one is not.
    """
    waited = 0.0
    while True:
        try:
            receipt = call(audit_id)
            break
        except TeardownInProgressError as exc:
            pause = _pause(exc)
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


def settle_delete(
    audit_dir: Path, state: AuditState, receipt: Mapping[str, Any], client: CleanupClient
) -> dict[str, Any]:
    """Record a published delete's receipt: write it as sent, and remove local files iff it deleted.

    The platform says whether it deleted the audit (``audit_status``); one that sends none is the
    legacy shape, deleted when no row is left, and one that sends a status this client does not
    know is not deleted. ``halted`` means the platform is not done: NOTHING local is removed,
    and the command exits 1 so the next call finishes it. Once it did delete, what it refused
    to claim is deleted here (:func:`~dagnam.audit.cleanup_walk.unclaimed_to_delete`) and joins
    the receipt; what that walk cannot remove is recorded as kept (the audit is gone, so the
    directory is deleted all the same: the receipt names it for the owner) rather than holding
    the directory out of the deleted state for ever. The ids the platform kept are recorded in
    the state so no listing offers them as the audit's. A deployment the platform left behind
    keeps the keys that call it; local files that cannot be removed (a link) are rows of their
    own on the written receipt.
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
    if deleted:
        mine = unclaimed_to_delete(state, rows)
        walked = {i for found in mine.values() for i in found}
        extra = walk(client, mine, state.project_id)
    state.kept_ids += [
        str(r.get("id"))
        for r in rows
        if r.get("status") == KEPT
        and str(r.get("id")) not in state.kept_ids
        and str(r.get("id")) not in walked
    ]
    remember(state, extra)
    if not deleted:
        save_state(audit_dir, state)
        return dict(receipt)
    state.kept_ids += [
        str(r.get("id"))
        for r in extra
        if decide(r).verdict is Verdict.LEFT and str(r.get("id")) not in state.kept_ids
    ]
    left = [r for r in rows if decide(r).verdict is Verdict.LEFT]
    stuck = [r for r in extra if decide(r).verdict is Verdict.LEFT]
    up = any(r.get("kind") == "deployment" for r in [*left, *stuck])
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
            seen = {str(r.get("id")) for r in receipt_rows(receipt)}  # the platform's own: not ours
            if mine := unclaimed(state) - seen:
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
