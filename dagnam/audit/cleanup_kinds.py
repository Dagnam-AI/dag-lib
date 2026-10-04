"""One artifact at a time: delete it or stop it, and say what happened as a receipt row.

This is the walk of an UNPUBLISHED audit only (``audit_id`` is ``None``: ``--local-only``, or
a run that never reached ``create_audit``). A published audit's resources are touched by the
platform's own delete and cancel routes and by nothing here
(:mod:`dagnam.audit.cleanup`). :data:`KINDS` says how one id of each kind is deleted and how
its absence is confirmed, :data:`STOPS` how a live one is stopped, and :func:`delete_one`
turns either outcome into a row -- never an exception, so one artifact the platform keeps or
fails on does not cost the receipt for every other one.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
import json
from typing import Any, Protocol

from dagnam._core.exceptions import (
    APIError,
    DagnamError,
    DatasetNotFoundError,
    DeploymentNotFoundError,
    ModelError,
    ModelNotFoundError,
    ProjectNotFoundError,
    TrainingJobNotFoundError,
)
from dagnam._types import JsonObject
from dagnam.audit.receipt_rows import ALREADY_STOPPED, PLATFORM_ONLY, blocked
from dagnam.audit.steps import CONFLICT_STATUS

FINISHED_STATUS = 400
METHOD_NOT_ALLOWED = 405
"""The platform's answer to cancelling a run that already ended: a refusal, not a failure."""
PAUSED_STATUSES = frozenset({"paused", "stopped"})
STOPPED = "stopped"
"""A cancel's receipt status for a run it cancelled or a deployment it paused (the server's word)."""
STILL_THERE = "the platform reported the delete done, but the id still reads back"
"""The reason on an id whose delete answered success and whose re-read still finds it."""


class CleanupBlockedError(DagnamError):
    """The platform refused a delete; the message is the reason put in the receipt."""


class PlatformOnlyError(CleanupBlockedError):
    """The platform has no route this client can use for it; only the platform can remove it."""


class CleanupClient(Protocol):
    """The ``DagnamClient`` methods deletion drives; a fake implements these and no more."""

    def cancel_audit(self, audit_id: str) -> JsonObject:
        """``POST /api/v1/audits/{id}/cancel``: the platform's own walk, one receipt."""
        ...

    def delete_audit(self, audit_id: str) -> JsonObject:
        """``DELETE /api/v1/audits/{id}``: the platform's own walk, one receipt."""
        ...

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
        """``DELETE /api/v1/deployments/{id}``; a refusal is a ``DeploymentStateError``."""
        ...

    def get_model_version(self, version_id: str) -> JsonObject:
        """``GET /api/v1/model-versions/{id}``; raises ``ModelNotFoundError``."""
        ...

    def purge_model_version(self, version_id: str) -> JsonObject | None:
        """``DELETE /api/v1/model-versions/{id}``: this one version's weights, never its entry.

        Idempotent; answers one receipt row, refuses (409) a version a deployment serves.
        """
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

    def list_projects(self, **filter_params: str | int) -> JsonObject | str | None:
        """``GET /api/v1/projects``: ``{"items", "pages", ...}``, ``page`` and ``limit`` filters."""
        ...

    def get_project_datasets(self, project_id: str) -> JsonObject:
        """``GET /api/v1/projects/{id}/datasets``: the project's datasets grouped by role."""
        ...

    def delete_project(self, project_id: str) -> None:
        """``DELETE /api/v1/projects/{id}``."""
        ...


def _purge_version(client: CleanupClient, version_id: str) -> JsonObject | None:
    """A version is purged through its own route, never by deleting the registry entry around it.

    A platform without the route answers 404 (or 405), which is also how a version that is
    gone answers: the re-read tells them apart. One that still reads is a version only the
    platform can remove, and is reported so (:class:`PlatformOnlyError`).
    """
    try:
        return client.purge_model_version(version_id)
    except ModelError as exc:
        kept = _kept_by_refusal(version_id, str(exc))
        if kept is None:
            raise
        return kept
    except (ModelNotFoundError, APIError) as exc:
        if isinstance(exc, APIError) and exc.status_code != METHOD_NOT_ALLOWED:
            raise
        _read_version(client, version_id)  # gone: raises its not-found, which says so
        raise PlatformOnlyError(
            f"this platform has no route to remove version {version_id}: remove it on the website"
        ) from None


def _kept_by_refusal(version_id: str, body: str) -> JsonObject | None:
    """The ``kept`` row for a purge the platform refused because a live endpoint serves the version.

    The 409 body is ``{"detail": {"error": "weights_served", "status": "kept", ...}}``: the
    platform's own decision that the version is not this audit's to remove, so it is ``kept``
    (never ``blocked``) and an unpublished delete finishes around it. Any other 409 is a refusal.
    """
    try:
        data = json.loads(body)
    except ValueError:
        return None
    detail = data.get("detail") if isinstance(data, dict) else None
    if not isinstance(detail, dict) or detail.get("status") != "kept":
        return None
    reason = str(detail.get("message") or detail.get("reason") or WEIGHTS_SERVED)
    code = str(detail.get("code") or detail.get("error") or WEIGHTS_SERVED)
    return {
        "kind": "model_version",
        "id": version_id,
        "status": "kept",
        "code": code,
        "reason": reason,
    }


def _read_version(client: CleanupClient, version_id: str) -> JsonObject:
    """The version's row, or its not-found.

    A purged row that still reads (``revoked``) is NOT gone: the platform's own purge answer
    says whether the weights went.
    """
    return client.get_model_version(version_id)


def _delete_training_job(client: CleanupClient, job_id: str) -> None:
    """A job is deleted through the bulk route (the platform has no per-job ``DELETE``).

    This is what frees the dataset: the run specification the audit submitted
    (``PostTrainingRun``) points at the job with ``ON DELETE CASCADE`` and at
    the dataset version with ``NO ACTION``, so while the job stands the
    dataset delete answers 409 "referenced by a training run". The route
    answers 200 with a per-id ``errors`` list rather than a status code, so a
    refusal -- the platform deletes terminal jobs only -- surfaces here as
    :class:`CleanupBlockedError` carrying the server's own wording.

    So a run still going is cancelled first: refused, it would go on training
    -- and billing -- until its recipe's time bound. A run that already ended
    answers the cancel with a 400, which is nothing to stop.
    """
    with suppress(APIError):
        client.cancel_training_job(job_id)
    errors = client.bulk_delete_training_jobs([job_id]).get("errors")
    if isinstance(errors, list) and errors:
        raise CleanupBlockedError(f"the platform refused the delete: {json.dumps(errors)}")


type Delete = Callable[[CleanupClient, str], object]
GONE_STATUSES = frozenset({"deleted", "already_absent"})
"""The statuses of a platform's own answer to a delete that settle the id without a re-read."""
PROJECT_PAGES = 100
"""The most pages of the owner's projects read when looking for where a dataset is linked."""
IN_USE_ELSEWHERE = "in_use_elsewhere"
"""The ``code`` of a dataset left alone because a project this directory did not create uses it."""
WEIGHTS_SERVED = "weights_served"
"""The ``code`` of a version a live endpoint serves: the purge refuses (409) and keeps it."""
type Get = Callable[[CleanupClient, str], JsonObject]
type Stop = Callable[[CleanupClient, str], dict[str, Any]]

KINDS: tuple[tuple[str, Delete, Get, type[DagnamError]], ...] = (
    (
        "deployment",
        lambda c, i: c.delete_deployment(i),
        lambda c, i: c.get_deployment(i),
        DeploymentNotFoundError,
    ),
    (
        "model_version",
        _purge_version,
        _read_version,
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
BY_KIND = {kind: (delete, get, absent) for kind, delete, get, absent in KINDS}
"""The same, by kind."""


_ANSWERS = (DagnamError, KeyError, TypeError)
"""What a platform answer can raise here: the client's own errors, and a body of the wrong shape.

A walk that has to finish and write its receipt cannot let a reply with a missing
field or a non-object body end it, whichever reader tripped on it.
"""


def _say(exc: Exception) -> str:
    """An error as a receipt says it: the client's own message, else what was raised."""
    return str(exc) if isinstance(exc, DagnamError) else repr(exc)


def _reason(exc: Exception, refusal: int = CONFLICT_STATUS) -> str:
    """What the receipt says an id was kept for: the platform's own words when it refused."""
    if isinstance(exc, APIError) and exc.status_code == refusal:
        return exc.message
    return _say(exc)


def datasets_in_other_projects(client: CleanupClient, own_project: str | None) -> set[str]:
    """Every dataset id linked to a project of the owner's that this directory did not create.

    A dataset resource carries no list of the projects it is linked to, so the owner's projects
    are listed (a page of 100 at a time, at most :data:`PROJECT_PAGES`) and each one's datasets
    read. A read that fails raises: whether a dataset is somebody's work is not a guess.
    """
    found: set[str] = set()
    page = 1
    while page <= PROJECT_PAGES:
        listing = client.list_projects(page=page, limit=100)
        if not isinstance(listing, dict):
            raise TypeError("the project list was not an object")
        items, pages = listing.get("items"), listing.get("pages")
        for project in items if isinstance(items, list) else []:
            if isinstance(project, dict) and str(project.get("id")) != own_project:
                found |= _dataset_ids(client.get_project_datasets(str(project["id"])))
        if not isinstance(pages, int) or page >= pages:
            return found
        page += 1
    return found


def _dataset_ids(grouped: JsonObject) -> set[str]:
    """The ids in a project's datasets grouped by role."""
    return {
        str(item["id"])
        for items in grouped.values()
        if isinstance(items, list)
        for item in items
        if isinstance(item, dict)
    }


def delete_one(
    client: CleanupClient,
    kind: str,
    delete: Delete,
    get: Get,
    absent: type[DagnamError],
    item_id: str,
) -> dict[str, Any]:
    """One id's receipt row: ``deleted``, ``already_absent``, or ``blocked`` with a reason.

    Nothing the platform answers raises out of here. A refusal arrives as
    whatever the client maps it to -- a 409 is an ``APIError`` on a dataset and
    a ``DeploymentStateError`` on a deployment -- and a failure as a 5xx or a
    dead connection; either is a ``blocked`` row carrying it, and so is a
    delete that reported success for an id that still reads back. One id never
    costs the receipt for every other one, nor the cancel of a run that is
    still billing; ``state.json`` keeps the id and the next ``audit delete``
    retries it. The re-read still decides: a refusal for an id that is in fact
    gone is ``already_absent``, not ``blocked``.
    """
    item: dict[str, Any] = {"kind": kind, "id": item_id}
    reason: str | None = None
    try:
        answer = delete(client, item_id)
    except absent:
        return {**item, "status": "already_absent"}
    except PlatformOnlyError as exc:
        return blocked(kind, item_id, str(exc), PLATFORM_ONLY)
    except _ANSWERS as exc:
        reason = _reason(exc)
    else:
        status = answer.get("status") if isinstance(answer, dict) else None
        if status in GONE_STATUSES:  # the platform's own receipt row for this one id
            return {**item, "status": status}
        if status == "kept" and isinstance(answer, dict):  # the owner uses it: not ours to remove
            return {**item, **answer}
    try:
        get(client, item_id)
    except absent:
        return {**item, "status": "already_absent" if reason else "deleted"}
    except _ANSWERS as exc:
        reason = reason or f"the delete could not be confirmed: {_say(exc)}"
    return blocked(kind, item_id, reason or STILL_THERE)


def _stop_job(client: CleanupClient, job_id: str) -> dict[str, Any]:
    """Cancel one run: ``stopped``, ``stopped [already_stopped]``, ``already_absent`` or ``blocked``.

    A run that already ended answers 400: it is ``already_stopped``, which marks nothing, so a
    finished run stays resumable. Any other failure (a 5xx, a dead connection) is a ``blocked``
    row, so one run that cannot be stopped never keeps the next one running.
    """
    row: dict[str, Any] = {"kind": "training_job", "id": job_id}
    try:
        client.cancel_training_job(job_id)
    except TrainingJobNotFoundError:
        return {**row, "status": "already_absent"}
    except APIError as exc:
        if exc.status_code == FINISHED_STATUS:
            return {**row, "status": STOPPED, "code": ALREADY_STOPPED}
        return blocked("training_job", job_id, _say(exc))
    except _ANSWERS as exc:
        return blocked("training_job", job_id, _say(exc))
    return {**row, "status": STOPPED}


def _stop_deployment(client: CleanupClient, deployment_id: str) -> dict[str, Any]:
    """Pause one deployment: ``stopped``, ``already_absent`` or ``blocked`` with why -- never a raise.

    One whose revision never activated cannot be paused, and says why.
    """
    row: dict[str, Any] = {"kind": "deployment", "id": deployment_id}
    try:
        client.pause_deployment(deployment_id)
    except DeploymentNotFoundError:
        return {**row, "status": "already_absent"}
    except _ANSWERS as exc:
        if _is_paused(client, deployment_id):  # the owner paused it already: nothing to stop
            return {**row, "status": STOPPED, "code": ALREADY_STOPPED}
        return blocked("deployment", deployment_id, _say(exc))
    return {**row, "status": STOPPED}


def _is_paused(client: CleanupClient, deployment_id: str) -> bool:
    """Whether the endpoint itself says it is paused or stopped (a refused pause is read back once)."""
    try:
        return client.get_deployment(deployment_id).get("status") in PAUSED_STATUSES
    except _ANSWERS:
        return False


STOPS: dict[str, Stop] = {"training_job": _stop_job, "deployment": _stop_deployment}
"""How a live artifact of each kind is stopped: a run is cancelled, a deployment paused."""

__all__ = [
    "BY_KIND",
    "FINISHED_STATUS",
    "IN_USE_ELSEWHERE",
    "KINDS",
    "STILL_THERE",
    "STOPPED",
    "STOPS",
    "CleanupBlockedError",
    "CleanupClient",
    "PlatformOnlyError",
    "datasets_in_other_projects",
    "delete_one",
]
