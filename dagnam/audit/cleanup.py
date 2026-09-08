"""``audit delete``: remove every platform artifact the state recorded, with a receipt (spec section 10).

Each recorded id goes through the same three moves -- delete, re-read
expecting not-found, record -- driven by :data:`KINDS`, never by a branch per
artifact. Every id ends in one of three receipt states: ``deleted``,
``already_absent`` (nothing there, so running the command twice is safe) or
``blocked`` with the platform's ``reason`` for refusing. Local workload files
and the secret store go last, and only when nothing is blocked -- they are
what a second ``audit delete`` needs to finish the job.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
from typing import Any, Protocol

from dagnam._core.exceptions import (
    APIError,
    DagnamError,
    DatasetNotFoundError,
    DeploymentNotFoundError,
    ModelNotFoundError,
    ProjectNotFoundError,
    TrainingJobNotFoundError,
)
from dagnam._types import JsonObject
from dagnam.audit.secrets import SecretStore
from dagnam.audit.state import AuditState, StepState, load_state
from dagnam.audit.steps_train import RUN_COMPLETED, RUN_FAILED
from dagnam.audit.workspace import write_atomic

SCHEMA = "dagnam.audit.deleted/1"
DELETED_FILE = "deleted.json"
CANCELLED_FILE = "cancelled.json"
"""``audit cancel``'s receipt. A cancel stops artifacts; only a delete removes them."""
RUN_CANCELLED = "cancelled"
DEPLOY_PAUSED = "paused"
CANCELLED_ERROR = "cancelled: stopped by `dagnam audit cancel`"
"""``StepState.error`` on a candidate a cancel stopped; ``error_code`` reads ``cancelled``."""
TERMINAL_RUN = frozenset({RUN_COMPLETED, *RUN_FAILED})
"""Run statuses a cancel leaves alone: they already stopped on their own."""
CONFLICT_STATUS = 409
"""A refusal, not a failure: the id is recorded ``blocked``, the rest still runs."""


class CleanupBlockedError(DagnamError):
    """The platform refused a delete; the message is the reason put in the receipt."""


class CleanupClient(Protocol):
    """The ``DagnamClient`` methods deletion drives; a fake implements these and no more."""

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
    """
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


def mark_cancelled(state: AuditState, unpaused: Collection[str] = ()) -> None:
    """Record a cancel in the local state, whoever performed it.

    Both cancel paths call this -- the local walk over the recorded ids and the
    published cancel the server performs in one call -- so ``audit status``
    reads the same after either, and the next ``audit run`` finds a terminal
    run instead of resuming into ``wait_run`` against a job that is gone.
    ``unpaused`` names the deployments the platform refused to pause; their
    status is left as it was, because they were not paused.
    """
    for candidates in state.workloads.values():
        for step in candidates.values():
            _cancel_step(step, unpaused)
    state.halted = {"reason": RUN_CANCELLED}


def _cancel_step(step: StepState, unpaused: Collection[str]) -> None:
    """One candidate's marks: its run cancelled, its deployment paused, and itself terminal.

    A candidate that never started is left alone. Everything else records
    what the cancel did to it -- and, unless it had already scored, also
    :data:`CANCELLED_ERROR`, which is what makes the next ``audit run`` skip
    the candidate outright rather than resume into a wait against a job or a
    deployment that is gone: the run could have been stopped at any step, not
    only ``wait_run``. A candidate that scored keeps its result (its numbers
    are the report's) but *not* its deployment: a cancel pauses that endpoint
    like any other, and a status left saying ``running`` would make the next
    ``audit cancel`` pause it a second time.
    """
    if step.training_job_id is None and step.deployment_id is None:
        return
    if step.training_job_id is not None and step.run_status not in TERMINAL_RUN:
        step.run_status = RUN_CANCELLED
    if step.deployment_id is not None and step.deployment_id not in unpaused:
        step.deploy_status = DEPLOY_PAUSED
    if not step.scored:
        step.error = CANCELLED_ERROR


def forget_locally(audit_dir: Path, state: AuditState) -> None:
    """Drop the deployment keys and the derived rows: nothing of the audit is left here."""
    secrets = SecretStore(audit_dir)
    for steps in state.workloads.values():
        for step in steps.values():
            if step.key_ref is not None:
                secrets.forget(step.key_ref)
    shutil.rmtree(audit_dir / "workloads", ignore_errors=True)


def delete_audit(audit_dir: Path, client: CleanupClient) -> dict[str, Any]:
    """Delete every recorded artifact, write ``deleted.json``, then drop local rows and secrets.

    Returns the receipt: one item per recorded id, each ``deleted``,
    ``already_absent`` or ``blocked`` with the platform's ``reason``. Local
    workload rows and the deployment keys are dropped only when nothing is
    blocked -- they are what a later ``audit delete`` needs to finish. Raises
    ``RuntimeError`` when a delete reports success and the id still reads back.
    """
    state = load_state(audit_dir)
    items = [
        _delete_one(client, kind, delete, get, absent, item_id)
        for kind, delete, get, absent in KINDS
        for item_id in recorded_ids(state)[kind]
    ]
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "deleted_at": datetime.now(UTC).isoformat(),
        "items": items,
    }
    write_receipt(audit_dir, receipt)
    if not any(item["status"] == "blocked" for item in items):
        forget_locally(audit_dir, state)
    return receipt


__all__ = [
    "CANCELLED_ERROR",
    "CANCELLED_FILE",
    "DELETED_FILE",
    "KINDS",
    "SCHEMA",
    "CleanupBlockedError",
    "CleanupClient",
    "delete_audit",
    "forget_locally",
    "mark_cancelled",
    "receipt_rows",
    "recorded_ids",
    "write_receipt",
]
