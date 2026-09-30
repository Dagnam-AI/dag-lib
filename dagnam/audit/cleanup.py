"""``audit cancel`` and ``audit delete``: stop or remove every artifact the state recorded (spec section 10).

Each recorded id goes through the same three moves -- delete, re-read
expecting not-found, record -- driven by :data:`KINDS`, never by a branch per
artifact. Every id ends in one of three receipt states: ``deleted``,
``already_absent`` (nothing there, so running the command twice is safe) or
``blocked`` with the platform's ``reason`` for refusing. Local workload files
and the secret store go last, blocked or not; ``state.json`` keeps the ids a
second ``audit delete`` needs to finish the job.

A published audit is cancelled or deleted by the server first, but the server
only knows the ids the run managed to publish; the state is what this machine
knows. So every recorded id the server's receipt does not settle -- a dataset
whose upload failed before its version was published, a run whose ``submit``
patch never landed -- is stopped or deleted here too, and the receipt says so.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from contextlib import suppress
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
from typing import Any, Protocol

from dagnam_contracts.audit import CANCELLED_SCHEMA, DELETED_SCHEMA

from dagnam._core.exceptions import (
    APIError,
    DagnamError,
    DatasetNotFoundError,
    DeploymentNotFoundError,
    DeploymentStateError,
    ModelNotFoundError,
    ProjectNotFoundError,
    TrainingJobNotFoundError,
)
from dagnam._types import JsonObject
from dagnam.audit.secrets import SECRETS_FILE, SecretStore
from dagnam.audit.state import AuditState, StepState, load_state, save_state
from dagnam.audit.steps import CONFLICT_STATUS
from dagnam.audit.steps_train import RUN_SETTLED
from dagnam.audit.workspace import write_atomic

SCHEMA = DELETED_SCHEMA
"""The schema ``delete_audit`` stamps on ``deleted.json``; the contract owns the id."""
DELETED_FILE = "deleted.json"
CANCELLED_FILE = "cancelled.json"
"""``audit cancel``'s receipt. A cancel stops artifacts; only a delete removes them.

The server stamps it :data:`~dagnam_contracts.audit.CANCELLED_SCHEMA`
(``dagnam.audit.cancelled/1``) -- a different document from a delete receipt,
and it is written through verbatim, never re-stamped here.
"""
RUN_CANCELLED = "cancelled"
DEPLOY_PAUSED = "paused"
CANCELLED_ERROR = "cancelled: stopped by `dagnam audit cancel`"
"""``StepState.error`` on a candidate a cancel stopped; ``error_code`` reads ``cancelled``."""
TERMINAL_RUN = RUN_SETTLED
"""Run statuses a cancel leaves alone: they already stopped on their own."""
FINISHED_STATUS = 400
"""The platform's answer to cancelling a run that already ended: a refusal, not a failure."""
STOPPED = "stopped"
"""A cancel's receipt status for a run it cancelled or a deployment it paused (the server's word)."""
GONE = frozenset({"deleted", "already_absent"})
"""Receipt statuses that settle an id: nothing is left of it to delete."""
KEPT_WITH_PROJECT = frozenset({"project", "model_version"})
"""What the server keeps when it keeps the project (D-F14): the project and its registry."""
SERVER_FINISHES = frozenset({"model_version"})
"""Kinds whose server ``blocked`` row stands: stored weights only the server can purge (a retry
here reads the soft-deleted entry's 404 and would call them ``already_absent``)."""


class CleanupBlockedError(DagnamError):
    """The platform refused a delete; the message is the reason put in the receipt."""


class CleanupClient(Protocol):
    """The ``DagnamClient`` methods deletion drives; a fake implements these and no more."""

    def cancel_training_job(self, job_id: str) -> JsonObject:
        """``POST /api/v1/training/jobs/{id}/cancel``; a job that already ended is a 400."""
        ...

    def pause_deployment(self, deployment_id: str) -> JsonObject:
        """``POST /api/v1/deployments/{id}/pause``; raises ``DeploymentStateError`` when it cannot."""
        ...

    def get_deployment(self, deployment_id: str) -> JsonObject:
        """``GET /api/v1/deployments/{id}``; raises ``DeploymentNotFoundError``."""
        ...

    def delete_deployment(self, deployment_id: str) -> JsonObject | None:
        """``DELETE /api/v1/deployments/{id}``."""
        ...

    def get_model_version(self, version_id: str) -> JsonObject:
        """``GET /api/v1/model-versions/{id}``; raises ``ModelNotFoundError``."""
        ...

    def delete_model_entry(self, model_id: str) -> None:
        """``DELETE /api/v1/models/{id}`` (every version with it)."""
        ...

    def get_training_job(self, job_id: str) -> JsonObject:
        """``GET /api/v1/training/jobs/{id}``; raises ``TrainingJobNotFoundError``."""
        ...

    def bulk_delete_training_jobs(self, job_ids: list[str]) -> JsonObject:
        """``POST /api/v1/training/jobs/bulk-delete``; ``{"deleted": n, "errors": [...]}``."""
        ...

    def get_dataset_meta(self, dataset_id: str, version: str | None = None) -> JsonObject:
        """``GET /api/v1/datasets/{id}/meta``; raises ``DatasetNotFoundError``."""
        ...

    def delete_dataset(self, dataset_id: str) -> None:
        """``DELETE /api/v1/datasets/{id}``."""
        ...

    def get_project(self, project_id: str) -> JsonObject:
        """``GET /api/v1/projects/{id}``; raises ``ProjectNotFoundError``."""
        ...

    def delete_project(self, project_id: str) -> None:
        """``DELETE /api/v1/projects/{id}``."""
        ...


def _delete_model_version(client: CleanupClient, version_id: str) -> None:
    """A version is deleted through its entry (the registry has no per-version delete)."""
    entry_id = client.get_model_version(version_id)["entry_id"]
    client.delete_model_entry(str(entry_id))


def _delete_training_job(client: CleanupClient, job_id: str) -> None:
    """A job is deleted through the bulk route (the platform has no per-job ``DELETE``).

    This is what frees the dataset: the run specification the audit submitted
    (``PostTrainingRun``) points at the job with ``ON DELETE CASCADE`` and at
    the dataset version with ``NO ACTION``, so while the job stands the
    dataset delete answers 409 "referenced by a training run". The route
    answers 200 with a per-id ``errors`` list rather than a status code, so a
    refusal -- the platform deletes terminal jobs only -- surfaces here as
    :class:`CleanupBlockedError` carrying the server's own wording.

    So a run still going is cancelled first (spec P7): refused, it would go on
    training -- and billing -- until its recipe's time bound. A run that
    already ended answers the cancel with a 400, which is nothing to stop.
    """
    with suppress(APIError):
        client.cancel_training_job(job_id)
    errors = client.bulk_delete_training_jobs([job_id]).get("errors")
    if isinstance(errors, list) and errors:
        raise CleanupBlockedError(f"the platform refused the delete: {json.dumps(errors)}")


type Delete = Callable[[CleanupClient, str], object]
type Get = Callable[[CleanupClient, str], object]

KINDS: tuple[tuple[str, Delete, Get, type[DagnamError]], ...] = (
    (
        "deployment",
        lambda c, i: c.delete_deployment(i),
        lambda c, i: c.get_deployment(i),
        DeploymentNotFoundError,
    ),
    (
        "model_version",
        _delete_model_version,
        lambda c, i: c.get_model_version(i),
        ModelNotFoundError,
    ),
    (
        "training_job",
        _delete_training_job,
        lambda c, i: c.get_training_job(i),
        TrainingJobNotFoundError,
    ),
    (
        "dataset",
        lambda c, i: c.delete_dataset(i),
        lambda c, i: c.get_dataset_meta(i),
        DatasetNotFoundError,
    ),
    (
        "project",
        lambda c, i: c.delete_project(i),
        lambda c, i: c.get_project(i),
        ProjectNotFoundError,
    ),
)
"""Deletion order (children before the project): kind, delete, re-read, its not-found error.

The training job sits before the dataset because it is what references it: a
dataset a run names cannot be deleted while that run exists. The run itself
needs no entry -- it is the job's ``ON DELETE CASCADE`` child.
"""


def recorded_ids(state: AuditState) -> dict[str, list[str]]:
    """Every platform id in the state by kind, in deletion order, without duplicates."""
    ids: dict[str, list[str]] = {kind: [] for kind, *_ in KINDS}
    for steps in state.workloads.values():
        for step in steps.values():
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
    return ids


def _delete_one(
    client: CleanupClient,
    kind: str,
    delete: Delete,
    get: Get,
    absent: type[DagnamError],
    item_id: str,
) -> dict[str, Any]:
    """One id's receipt row: ``deleted``, ``already_absent``, or ``blocked`` with a reason.

    A refusal (``CleanupBlockedError``, or a 409 -- the dataset the run holds) is
    recorded rather than raised, so one blocked id does not cost the receipt
    for every other one. The re-read still decides: a refusal for an id that
    is in fact gone is ``already_absent``, not ``blocked``.
    """
    item: dict[str, Any] = {"kind": kind, "id": item_id}
    reason: str | None = None
    try:
        delete(client, item_id)
    except absent:
        return {**item, "status": "already_absent"}
    except CleanupBlockedError as refusal:
        reason = str(refusal)
    except APIError as exc:
        if exc.status_code != CONFLICT_STATUS:
            raise
        reason = exc.message
    try:
        get(client, item_id)
    except absent:
        return {**item, "status": "already_absent" if reason else "deleted"}
    if reason is not None:
        return {**item, "status": "blocked", "reason": reason}
    raise RuntimeError(f"{kind} {item_id} still exists after delete")


def write_receipt(audit_dir: Path, receipt: Mapping[str, Any], name: str = DELETED_FILE) -> Path:
    """Write one receipt atomically, exactly as given, and return where it went.

    The server's own receipt goes through here verbatim when the run published
    (it lists its rows under ``entries``, where this module's local receipt
    uses ``items``); :func:`receipt_rows` reads either. ``name`` says which
    receipt it is: a cancel writes :data:`CANCELLED_FILE`, so ``deleted.json``
    only ever means the artifacts are gone.
    """
    path = audit_dir / name
    write_atomic(path, json.dumps(receipt, indent=2))
    return path


def receipt_rows(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A receipt's rows, whichever key its writer used."""
    for key in ("items", "entries"):
        rows = receipt.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _stop_job(client: CleanupClient, job_id: str) -> dict[str, Any]:
    """Cancel one run; one that already ended is ``blocked`` with the platform's reason, never a crash."""
    row: dict[str, Any] = {"kind": "training_job", "id": job_id}
    try:
        client.cancel_training_job(job_id)
    except TrainingJobNotFoundError:
        return {**row, "status": "already_absent"}
    except APIError as exc:
        if exc.status_code != FINISHED_STATUS:
            raise
        return {**row, "status": "blocked", "reason": exc.message}
    return {**row, "status": STOPPED}


def _stop_deployment(client: CleanupClient, deployment_id: str) -> dict[str, Any]:
    """Pause one deployment; one whose revision never activated cannot be, and says why."""
    row: dict[str, Any] = {"kind": "deployment", "id": deployment_id}
    try:
        client.pause_deployment(deployment_id)
    except DeploymentNotFoundError:
        return {**row, "status": "already_absent"}
    except DeploymentStateError as exc:
        return {**row, "status": "blocked", "reason": str(exc)}
    return {**row, "status": STOPPED}


def cancel_recorded(
    state: AuditState, client: CleanupClient, server: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Stop every run and deployment the state records, mark what stopped, and return the receipt.

    ``server`` is the published audit's own cancel receipt: the ids it answered
    for are left as it left them, and everything else still live -- a run
    whose ``submit`` never reached the account -- is stopped here. Only what
    actually stopped is marked: a run that had already finished stays
    resumable, and a platform refusal is a ``blocked`` row, never a crash.
    """
    entries = receipt_rows(server) if server is not None else []
    seen = {(row.get("kind"), row.get("id")) for row in entries}
    for candidates in state.workloads.values():
        for step in candidates.values():
            live = (
                (
                    "training_job",
                    None if step.run_status in TERMINAL_RUN else step.training_job_id,
                    _stop_job,
                ),
                (
                    "deployment",
                    None if step.deploy_status == DEPLOY_PAUSED else step.deployment_id,
                    _stop_deployment,
                ),
            )
            for kind, item_id, stop in live:
                if item_id is not None and (kind, item_id) not in seen:
                    entries.append(stop(client, item_id))
                    seen.add((kind, item_id))
    mark_cancelled(state, {str(row.get("id")) for row in entries if row.get("status") == STOPPED})
    header = dict(server) if server is not None else {"schema": CANCELLED_SCHEMA}
    return {**header, "deleted_at": header.get("deleted_at") or _now(), "entries": entries}


def mark_cancelled(state: AuditState, stopped: Collection[str]) -> None:
    """Record a cancel in the local state: ``stopped`` names the runs and deployments it stopped.

    So ``audit status`` reads what actually happened, and the next ``audit
    run`` finds a terminal run instead of resuming into ``wait_run`` against a
    job that is gone -- while a run that had already finished, or a deployment
    the platform refused to pause, keeps the status it really has.
    """
    for candidates in state.workloads.values():
        for step in candidates.values():
            _cancel_step(step, stopped)
    state.halted = {"reason": RUN_CANCELLED}


def _cancel_step(step: StepState, stopped: Collection[str]) -> None:
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
    job_stopped = step.training_job_id is not None and step.training_job_id in stopped
    endpoint_stopped = step.deployment_id is not None and step.deployment_id in stopped
    if job_stopped:
        step.run_status = RUN_CANCELLED
    if endpoint_stopped:
        step.deploy_status = DEPLOY_PAUSED
    if (job_stopped or endpoint_stopped) and not step.scored:
        step.error = CANCELLED_ERROR


def forget_locally(audit_dir: Path, state: AuditState) -> None:
    """Drop the deployment keys and the derived rows, and mark the state deleted.

    ``state.json`` itself stays, with every id it recorded: that is all a
    second ``audit delete`` needs to finish what the platform refused.
    """
    secrets = SecretStore(audit_dir)
    for steps in state.workloads.values():
        for step in steps.values():
            if step.key_ref is not None:
                secrets.forget(step.key_ref)
    (audit_dir / SECRETS_FILE).unlink(missing_ok=True)
    shutil.rmtree(audit_dir / "workloads", ignore_errors=True)
    state.halted = {"reason": "deleted"}
    save_state(audit_dir, state)


def _as_the_account_left(
    client: CleanupClient, kind: str, get: Get, absent: type[DagnamError], item_id: str
) -> dict[str, Any]:
    """An id the account already decided about: read, never deleted.

    ``already_absent`` when the account deleted it, ``blocked`` when it kept it
    -- a project that holds the owner's own work, with its registry entries.
    """
    try:
        get(client, item_id)
    except absent:
        return {"kind": kind, "id": item_id, "status": "already_absent"}
    return {"kind": kind, "id": item_id, "status": "blocked", "reason": "kept by the account"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def delete_audit(
    audit_dir: Path,
    client: CleanupClient,
    server: Mapping[str, Any] | None = None,
    *,
    account_deleted: bool = False,
) -> dict[str, Any]:
    """Delete every recorded artifact, write ``deleted.json``, then drop local rows and secrets.

    ``server`` is the published audit's own delete receipt: an id it settled
    (``deleted`` or ``already_absent``) or whose weights it kept (a ``blocked``
    :data:`SERVER_FINISHES` row, which stands as written) is not touched again; every other
    recorded id -- one the server never learned about, or one it could not
    delete -- is deleted here, its row replacing the server's. A project the
    server kept (``blocked``: it holds what the audit did not record) is never
    deleted here, and nor are the registry entries it keeps with it.
    ``account_deleted`` is a published audit the account had already deleted
    (its delete answered 404, from the website say): the account decided about
    the project and its registry then, so those are only read. Returns the
    receipt: one row per id, each ``deleted``, ``already_absent`` or
    ``blocked`` with the platform's ``reason``. The local rows and keys go
    whatever was blocked (:func:`forget_locally`); ``state.json`` keeps the ids
    a later ``audit delete`` retries. Raises ``RuntimeError`` when a delete
    reports success and the id still reads back.
    """
    state = load_state(audit_dir)
    served = receipt_rows(server) if server is not None else []
    settled = {
        (row.get("kind"), row.get("id"))
        for row in served
        if row.get("status") in GONE
        or (row.get("status") == "blocked" and row.get("kind") in SERVER_FINISHES)
    }
    kept = [
        row["id"]
        for row in served
        if row.get("kind") == "project" and row.get("status") == "blocked"
    ]
    items: list[dict[str, Any]] = []
    for kind, delete, get, absent in KINDS:
        for item_id in recorded_ids(state)[kind]:
            if (kind, item_id) in settled:
                continue
            if account_deleted and kind in KEPT_WITH_PROJECT:
                items.append(_as_the_account_left(client, kind, get, absent, item_id))
                continue
            if kept and kind in KEPT_WITH_PROJECT:
                if kind != "project":  # the server's own row already says why the project stays
                    reason = f"kept with project {kept[0]}"
                    items.append(
                        {"kind": kind, "id": item_id, "status": "blocked", "reason": reason}
                    )
                continue
            items.append(_delete_one(client, kind, delete, get, absent, item_id))
    receipt: dict[str, Any]
    if server is None:
        receipt = {"schema": SCHEMA, "deleted_at": _now(), "items": items}
    else:
        walked = {(item["kind"], item["id"]) for item in items}
        kept = [row for row in served if (row.get("kind"), row.get("id")) not in walked]
        receipt = {**server, "entries": kept + items}
    write_receipt(audit_dir, receipt)
    forget_locally(audit_dir, state)
    return receipt


__all__ = [
    "CANCELLED_ERROR",
    "CANCELLED_FILE",
    "DELETED_FILE",
    "GONE",
    "KEPT_WITH_PROJECT",
    "KINDS",
    "SCHEMA",
    "SERVER_FINISHES",
    "STOPPED",
    "CleanupBlockedError",
    "CleanupClient",
    "cancel_recorded",
    "delete_audit",
    "forget_locally",
    "mark_cancelled",
    "receipt_rows",
    "recorded_ids",
    "write_receipt",
]
