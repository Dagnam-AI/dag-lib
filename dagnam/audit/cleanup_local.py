"""What a cancel or a delete does on this machine, and the shape of what it writes.

``state.json`` keeps every id it recorded (:func:`recorded_ids`); ``deleted.json`` and
``cancelled.json`` are display copies, written atomically (:func:`write_receipt`) and never read
for a decision. :func:`mark_cancelled` records what a cancel stopped, :func:`forget_locally`
removes this machine's rows and deployment keys. Nothing here calls the platform.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from contextlib import suppress
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from dagnam_contracts.audit import CANCELLED_SCHEMA, DELETED_SCHEMA

from dagnam.audit.cleanup_kinds import KINDS
from dagnam.audit.receipt_rows import (
    ALREADY_ABSENT,
    DELETED,
    NOT_ANSWERED,
    STOPPED,
    blocked,
    decide,
)
from dagnam.audit.secrets import SECRETS_FILE, SecretStore
from dagnam.audit.state import DELETED_STATE, AuditState, StepState, save_state
from dagnam.audit.steps_train import RUN_SETTLED
from dagnam.audit.workspace import (
    UnsafeWorkloadsError,
    is_plain_name,
    remove_workload,
    workload_names,
    workloads_dir,
    write_atomic,
)

SCHEMA = DELETED_SCHEMA
"""The schema ``deleted.json`` carries; the contract owns the id."""
DELETED_FILE = "deleted.json"
CANCELLED_FILE = "cancelled.json"
"""``audit cancel``'s receipt. A cancel stops artifacts; only a delete removes them.

It is stamped :data:`~dagnam_contracts.audit.CANCELLED_SCHEMA` (``dagnam.audit.cancelled/1``):
a different document from a delete receipt, and the platform's own is written through verbatim.
"""
RUN_CANCELLED = "cancelled"
DEPLOY_PAUSED = "paused"
DEPLOY_DELETED = "deleted"
"""``StepState.deploy_status`` of an endpoint a cancel found deleted (one that never served)."""
CANCELLED_ERROR = "cancelled: stopped by `dagnam audit cancel`"
"""``StepState.error`` on a candidate a cancel stopped; ``error_code`` reads ``cancelled``."""
TERMINAL_RUN = RUN_SETTLED
"""Run statuses a cancel leaves alone: they already stopped on their own."""
GONE = frozenset({DELETED, ALREADY_ABSENT})
"""Receipt statuses that mean nothing is left of the id."""
LOCAL_WORKLOADS = "local_workload"
LOCAL_KEYS = "local_keys"
"""Receipt kinds of this machine's own leftovers: rows under ``workloads/`` and the key file."""


def recorded_ids(state: AuditState) -> dict[str, list[str]]:
    """Every platform id in the state by kind, in deletion order, without duplicates."""
    ids: dict[str, list[str]] = {kind: [] for kind, *_ in KINDS}
    for step in state.all_steps():
        for kind, value in (
            ("deployment", step.deployment_id),
            ("model_version", step.model_version_id),
            ("training_job", step.training_job_id),
            ("dataset", step.dataset_id),
        ):
            if value is not None and value not in ids[kind]:
                ids[kind].append(value)
    if state.project_id is not None:
        ids["project"].append(state.project_id)
    return {kind: [i for i in found if i not in state.kept_ids] for kind, found in ids.items()}


def write_receipt(audit_dir: Path, receipt: Mapping[str, Any], name: str = DELETED_FILE) -> Path:
    """Write one receipt atomically, exactly as given, and return where it went.

    ``name`` says which receipt it is: a cancel writes :data:`CANCELLED_FILE`, so
    ``deleted.json`` only ever means the artifacts are gone.
    """
    path = audit_dir / name
    write_atomic(path, json.dumps(receipt, indent=2))
    return path


def receipt_rows(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A receipt's rows, whichever key its writer used (``entries``; ``items`` in older files).

    A row that is not an object is kept as an empty one, so it is shown as ``?``, decided as a
    status nobody can read, and fails the command: it is never dropped.
    """
    for key in ("entries", "items"):
        rows = receipt.get(key)
        if isinstance(rows, list):
            return [row if isinstance(row, dict) else {} for row in rows]
    return []


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def unanswered(audit_id: str | None, verb: str, why: str) -> dict[str, Any]:
    """The receipt of a ``cancel`` or ``delete`` the platform did not answer: nothing was touched.

    One ``audit`` row, ``blocked [not_answered]``; no remote call was made and no local file is
    removed, so running the command again is exactly as safe as the first time.
    """
    reason = f"the platform did not answer the {verb} ({why}); nothing was touched: run it again"
    return {
        "schema": CANCELLED_SCHEMA if verb == "cancel" else SCHEMA,
        "deleted_at": now_iso(),
        "entries": [blocked("audit", audit_id, reason, NOT_ANSWERED)],
    }


def deleted_elsewhere(verb: str) -> dict[str, Any]:
    """The receipt of a delete the caller says is already done (``--already-deleted``).

    Nothing remote is touched, and the receipt has no rows, because the caller knows of none.
    """
    return {
        "schema": CANCELLED_SCHEMA if verb == "cancel" else SCHEMA,
        "deleted_at": now_iso(),
        "audit_status": "deleted",
        "entries": [],
    }


def live(step: StepState) -> list[tuple[str, str]]:
    """``(kind, id)`` of what this candidate still has going: its run, its endpoint.

    A run that already settled and a deployment already paused (or found
    deleted) are not live. This is what a cancel stops, and what a forced rescan
    leaves behind when it retires a candidate mid-flight.
    """
    found: list[tuple[str, str]] = []
    if step.training_job_id is not None and step.run_status not in TERMINAL_RUN:
        found.append(("training_job", step.training_job_id))
    if step.deployment_id is not None and step.deploy_status not in (DEPLOY_PAUSED, DEPLOY_DELETED):
        found.append(("deployment", step.deployment_id))
    return found


def mark_cancelled(state: AuditState, stopped: Collection[str], gone: Collection[str] = ()) -> None:
    """Record a cancel in the local state: ``stopped`` names the runs and deployments it stopped.

    ``gone`` names those it found already deleted -- a deployment that never
    served is deleted by a cancel, not paused. So ``audit status`` reads what
    actually happened, and the next ``audit run`` finds a terminal run instead
    of resuming into ``wait_run`` against a job that is gone, or into
    ``wait_active`` against an endpoint that answers 404 -- while a run that had
    already finished, or a deployment the platform refused to pause, keeps the
    status it really has.
    """
    for step in state.all_steps():
        _cancel_step(step, stopped, gone)
    state.halted = {"reason": RUN_CANCELLED}


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


def _cancel_step(step: StepState, stopped: Collection[str], gone: Collection[str]) -> None:
    """One candidate's marks: what the cancel stopped of it, and -- unless it scored -- itself terminal.

    A candidate the cancel stopped nothing of is left alone. Otherwise, unless
    it had already scored, it also gets :data:`CANCELLED_ERROR`, which is what
    makes the next ``audit run`` skip it outright rather than resume into a
    wait against a job or a deployment that is gone: the run could have been
    stopped at any step, not only ``wait_run``. A candidate that scored keeps
    its result (its numbers are the report's) but *not* its deployment: a
    cancel pauses that endpoint like any other, and a status left saying
    ``running`` would make the next ``audit cancel`` pause it a second time.
    """
    job_ended = step.training_job_id is not None and (
        step.training_job_id in stopped or step.training_job_id in gone
    )
    endpoint_stopped = step.deployment_id is not None and step.deployment_id in stopped
    endpoint_gone = step.deployment_id is not None and step.deployment_id in gone
    if job_ended:
        step.run_status = RUN_CANCELLED
    if endpoint_stopped:
        step.deploy_status = DEPLOY_PAUSED
    if endpoint_gone:
        step.deploy_status = DEPLOY_DELETED
    if (job_ended or endpoint_stopped or endpoint_gone) and not step.scored:
        step.error = CANCELLED_ERROR


def _local_row(kind: str, name: str, exc: Exception) -> dict[str, Any]:
    """A leftover on this machine, as a ``blocked`` receipt row."""
    return blocked(kind, name, str(exc))


def _remove_workloads(audit_dir: Path) -> list[dict[str, Any]]:
    """Remove the files a scan wrote under ``workloads/``; the rows of what could not go.

    Only what :func:`~dagnam.audit.workspace.remove_workload` will remove: a
    folder that is not a scan's, or that holds anything else, stays for its
    owner, and a link is never followed (it is a ``blocked`` row).
    """
    try:
        names = sorted(workload_names(audit_dir))
    except UnsafeWorkloadsError as exc:
        return [_local_row(LOCAL_WORKLOADS, "workloads", exc)]
    left: list[dict[str, Any]] = []
    for name in filter(is_plain_name, names):
        try:
            remove_workload(audit_dir, name)
        except UnsafeWorkloadsError as exc:
            left.append(_local_row(LOCAL_WORKLOADS, name, exc))
    with suppress(OSError):  # still holds something that is not ours
        workloads_dir(audit_dir).rmdir()
    return left


def forget_locally(
    audit_dir: Path, state: AuditState, *, keep_keys: bool = False, settled: bool = True
) -> list[dict[str, Any]]:
    """Drop the deployment keys and the derived rows; mark the state deleted if all of it went.

    ``state.json`` itself stays, with every id it recorded. With ``keep_keys`` -- an endpoint
    is still up, and its key is the only way to call it -- the keys and the rows stay too. The
    state is marked deleted only when ``settled`` (nothing of the audit's is left on the platform)
    and every local file went. Returns a ``blocked`` row for each local file that is a link, or
    sits behind one, and so was not removed: the next ``audit delete`` finishes it.
    """
    left: list[dict[str, Any]] = []
    if not keep_keys:
        try:
            secrets = SecretStore(audit_dir)
            for step in state.all_steps():
                if step.key_ref is not None:
                    secrets.forget(step.key_ref)
            (audit_dir / SECRETS_FILE).unlink(missing_ok=True)
        except UnsafeWorkloadsError as exc:
            left.append(_local_row(LOCAL_KEYS, SECRETS_FILE, exc))
        left += _remove_workloads(audit_dir)
        if settled and not left:
            state.halted = dict(DELETED_STATE)
    save_state(audit_dir, state)
    return left


__all__ = [
    "CANCELLED_ERROR",
    "CANCELLED_FILE",
    "DELETED_FILE",
    "DEPLOY_DELETED",
    "DEPLOY_PAUSED",
    "GONE",
    "LOCAL_KEYS",
    "LOCAL_WORKLOADS",
    "SCHEMA",
    "deleted_elsewhere",
    "forget_locally",
    "live",
    "mark_cancelled",
    "now_iso",
    "receipt_rows",
    "recorded_ids",
    "settle_cancel",
    "unanswered",
    "write_receipt",
]
