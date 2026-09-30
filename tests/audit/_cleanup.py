"""An in-memory platform holding an audit's artifacts, for ``audit cancel`` and ``audit delete``.

Each ``get`` raises the kind's own not-found once its id is gone, and the
refusals are the platform's own: a running job cannot be deleted, a finished
one cannot be cancelled, a deployment whose revision never activated cannot be
paused, and a dataset a job still references answers 409.
"""

from __future__ import annotations

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
from dagnam._types import JsonArray, JsonObject
from dagnam.audit.cleanup import CleanupClient

DATASET_IN_USE = "Dataset is referenced by a training run and cannot be deleted"
"""The platform's own 409 wording, from its dataset service."""
_FAKE_KIND = {"deployment": "deployment", "training_job": "job", "dataset": "dataset"}
"""A receipt row's kind -> the fake's own bucket for it."""


class FakeCleanup:
    """A platform holding ids to delete: each ``get`` raises not-found once its id is gone."""

    def __init__(self, **present: list[str]) -> None:
        self.present: dict[str, set[str]] = {
            kind: set(present.get(kind, []))
            for kind in ("deployment", "model", "job", "dataset", "project")
        }
        self.entry_of: dict[str, str] = {}
        """model version id -> entry id (a deleted entry takes its versions with it)."""
        self.call_log: list[tuple[str, str]] = []
        self.sticky: set[str] = set()
        """Ids whose delete succeeds but which a re-read still finds (a server bug to surface)."""
        self.running: set[str] = set()
        """Job ids the platform refuses to delete: it deletes terminal jobs only."""
        self.held_by_job: dict[str, str] = {}
        """dataset id -> job id: the live FK, a 409 for as long as that job exists."""
        self.dataset_error: DagnamError | None = None
        """Raised by ``delete_dataset`` whatever else is true (a server failure)."""
        self.finished: set[str] = set()
        """Job ids that already ended: cancelling one is the platform's 400."""
        self.unpausable: set[str] = set()
        """Deployment ids whose revision never activated: pausing one is a 409."""
        self.server_receipt: JsonObject = {"schema": "x", "deleted_at": "t", "entries": []}
        """What the published audit's cancel or delete answers; a ``deleted`` row takes its id."""

    def _take(self, kind: str, item_id: str, absent: type[DagnamError]) -> None:
        if item_id not in self.present[kind]:
            raise absent(item_id)
        if item_id not in self.sticky:
            self.present[kind].discard(item_id)

    def _need(self, kind: str, item_id: str, absent: type[DagnamError]) -> JsonObject:
        if item_id not in self.present[kind]:
            raise absent(item_id)
        return {"id": item_id}

    def cancel_audit(self, audit_id: str) -> JsonObject:
        self.call_log.append(("cancel_audit", audit_id))
        return dict(self.server_receipt)

    def delete_audit(self, audit_id: str) -> JsonObject:
        """The server's walk: every row it calls ``deleted`` is gone from the platform after."""
        self.call_log.append(("delete_audit", audit_id))
        entries = self.server_receipt["entries"]
        for row in entries if isinstance(entries, list) else []:
            if isinstance(row, dict) and row.get("status") == "deleted":
                kind = _FAKE_KIND.get(str(row.get("kind")), "")
                self.present.get(kind, set()).discard(str(row.get("id")))
        return dict(self.server_receipt)

    def cancel_training_job(self, job_id: str) -> JsonObject:
        self.call_log.append(("cancel_training_job", job_id))
        if job_id not in self.present["job"]:
            raise TrainingJobNotFoundError(job_id)
        if job_id in self.finished:
            raise APIError(400, "Cannot cancel job with status completed")
        self.running.discard(job_id)
        self.finished.add(job_id)
        return {"id": job_id, "status": "cancelled"}

    def pause_deployment(self, deployment_id: str) -> JsonObject:
        self.call_log.append(("pause_deployment", deployment_id))
        if deployment_id not in self.present["deployment"]:
            raise DeploymentNotFoundError(deployment_id)
        if deployment_id in self.unpausable:
            raise DeploymentStateError("Invalid status transition from not_provisioned to paused")
        return {"id": deployment_id, "status": "paused"}

    def get_deployment(self, deployment_id: str) -> JsonObject:
        self.call_log.append(("get_deployment", deployment_id))
        return self._need("deployment", deployment_id, DeploymentNotFoundError)

    def delete_deployment(self, deployment_id: str) -> JsonObject | None:
        self.call_log.append(("delete_deployment", deployment_id))
        self._take("deployment", deployment_id, DeploymentNotFoundError)
        return None

    def get_model_version(self, version_id: str) -> JsonObject:
        self.call_log.append(("get_model_version", version_id))
        entry = self.entry_of.get(version_id)
        if entry is None or entry not in self.present["model"]:
            raise ModelNotFoundError(version_id)
        return {"id": version_id, "entry_id": entry}

    def delete_model_entry(self, model_id: str) -> None:
        self.call_log.append(("delete_model_entry", model_id))
        self._take("model", model_id, ModelNotFoundError)

    def get_training_job(self, job_id: str) -> JsonObject:
        self.call_log.append(("get_training_job", job_id))
        return self._need("job", job_id, TrainingJobNotFoundError)

    def bulk_delete_training_jobs(self, job_ids: list[str]) -> JsonObject:
        """The real route answers 200 with a per-id ``errors`` list, never a 404."""
        self.call_log.append(("bulk_delete_training_jobs", ",".join(job_ids)))
        deleted = 0
        errors: JsonArray = []
        for job_id in job_ids:
            if job_id not in self.present["job"]:
                errors.append({"job_id": job_id, "error": "Not found or not authorized"})
            elif job_id in self.running:
                errors.append({"job_id": job_id, "error": "Cannot delete job with status running"})
            else:
                self._take("job", job_id, TrainingJobNotFoundError)
                deleted += 1
        return {"deleted": deleted, "errors": errors}

    def get_dataset_meta(self, dataset_id: str, version: str | None = None) -> JsonObject:
        self.call_log.append(("get_dataset_meta", dataset_id))
        return self._need("dataset", dataset_id, DatasetNotFoundError)

    def delete_dataset(self, dataset_id: str) -> None:
        self.call_log.append(("delete_dataset", dataset_id))
        if self.dataset_error is not None:
            raise self.dataset_error
        if self.held_by_job.get(dataset_id) in self.present["job"]:
            raise APIError(409, DATASET_IN_USE)
        self._take("dataset", dataset_id, DatasetNotFoundError)

    def get_project(self, project_id: str) -> JsonObject:
        self.call_log.append(("get_project", project_id))
        return self._need("project", project_id, ProjectNotFoundError)

    def delete_project(self, project_id: str) -> None:
        self.call_log.append(("delete_project", project_id))
        self._take("project", project_id, ProjectNotFoundError)


def as_cleanup_client(platform: FakeCleanup) -> CleanupClient:
    """The cleanup fake, typed as the protocol ``delete_audit`` takes."""
    return platform
