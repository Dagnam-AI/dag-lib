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
from dagnam.audit.cleanup_kinds import CleanupClient

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
        self.purge_errors: dict[str, DagnamError] = {}
        """Version id -> what its purge raises (the platform's 409 or 502)."""
        self.claims: list[JsonArray] = []
        """The entries of each ``claim_audit_resources`` request."""
        self.claim_refuses: set[str] = set()
        """Ids the platform refuses to claim."""
        self.claim_codes: dict[str, str] = {}
        """Refused id -> the ``code`` of its refusal (else ``not_claimable``)."""
        self.claim_error: DagnamError | None = None
        self.no_purge_route = False
        """The platform predates the per-version purge route: it answers 404 for it."""
        self.call_log: list[tuple[str, str]] = []
        self.sticky: set[str] = set()
        """Ids whose delete succeeds but which a re-read still finds (a server bug to surface)."""
        self.running: set[str] = set()
        """Job ids the platform refuses to delete: it deletes terminal jobs only."""
        self.held_by_job: dict[str, str] = {}
        """dataset id -> job id: the live FK, a 409 for as long as that job exists."""
        self.other_projects: dict[str, list[str]] = {}
        """Another project of the owner's -> the dataset ids linked into it."""
        self.project_pages: dict[int, JsonObject | str] = {}
        """Page number -> the listing answered for it (else every project of ``other_projects``)."""
        self.project_datasets: dict[str, JsonObject] = {}
        """Project id -> the grouped datasets answered for it (else from ``other_projects``)."""
        self.project_reads_fail: Exception | None = None
        """Raised by every read of the owner's projects (the links cannot be checked)."""
        self.dataset_error: DagnamError | None = None
        """Raised by ``delete_dataset`` whatever else is true (a server failure)."""
        self.undeletable: set[str] = set()
        """Deployment ids the platform refuses to delete: a 409, which the client types."""
        self.unreadable: dict[str, DagnamError] = {}
        """id -> what every read of it raises (a server failure on the re-read)."""
        self.finished: set[str] = set()
        """Job ids that already ended: cancelling one is the platform's 400."""
        self.unpausable: set[str] = set()
        """Deployment ids whose revision never activated: pausing one is a 409."""
        self.statuses: dict[str, str] = {}
        """id -> the ``status`` a read of a job or an endpoint reports (else: a run that has not
        ended is ``running``, an endpoint that is up is ``running``)."""
        self.account_error: DagnamError | None = None
        """Raised by the account's own ``cancel_audit`` / ``delete_audit`` (a 5xx, a timeout)."""
        self.server_receipt: JsonObject = {"schema": "x", "deleted_at": "t", "entries": []}
        """What the published audit's cancel or delete answers; a ``deleted`` row takes its id."""

    def _take(self, kind: str, item_id: str, absent: type[DagnamError]) -> None:
        if item_id not in self.present[kind]:
            raise absent(item_id)
        if item_id not in self.sticky:
            self.present[kind].discard(item_id)

    def _need(self, kind: str, item_id: str, absent: type[DagnamError]) -> JsonObject:
        if item_id in self.unreadable:
            raise self.unreadable[item_id]
        if item_id not in self.present[kind]:
            raise absent(item_id)
        return {"id": item_id}

    def cancel_audit(self, audit_id: str) -> JsonObject:
        self.call_log.append(("cancel_audit", audit_id))
        if self.account_error is not None:
            raise self.account_error
        return dict(self.server_receipt)

    def delete_audit(self, audit_id: str) -> JsonObject:
        """The server's walk: every row it calls ``deleted`` is gone from the platform after."""
        self.call_log.append(("delete_audit", audit_id))
        if self.account_error is not None:
            raise self.account_error
        entries = self.server_receipt.get("entries")
        for row in entries if isinstance(entries, list) else []:
            if isinstance(row, dict) and row.get("status") == "deleted":
                kind = _FAKE_KIND.get(str(row.get("kind")), "")
                self.present.get(kind, set()).discard(str(row.get("id")))
        return dict(self.server_receipt)

    def claim_audit_resources(self, audit_id: str, entries: JsonArray) -> JsonObject:
        self.call_log.append(("claim_audit_resources", audit_id))
        self.claims.append(entries)
        if self.claim_error is not None:
            raise self.claim_error
        results: JsonArray = [
            {
                "kind": e["kind"],
                "id": e["id"],
                "result": "refused" if e["id"] in self.claim_refuses else "claimed",
                "code": self.claim_codes.get(str(e["id"]), "not_claimable")
                if e["id"] in self.claim_refuses
                else "claimed",
            }
            for e in entries
            if isinstance(e, dict)
        ]
        return {"results": results}

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
        found = self._need("deployment", deployment_id, DeploymentNotFoundError)
        return {**found, "status": self.statuses.get(deployment_id, "running")}

    def delete_deployment(self, deployment_id: str) -> JsonObject | None:
        self.call_log.append(("delete_deployment", deployment_id))
        if deployment_id in self.undeletable:
            raise DeploymentStateError("Cannot delete a deployment that is still deploying")
        self._take("deployment", deployment_id, DeploymentNotFoundError)
        return None

    def get_model_version(self, version_id: str) -> JsonObject:
        self.call_log.append(("get_model_version", version_id))
        return self._need("model", version_id, ModelNotFoundError)

    def purge_model_version(self, version_id: str) -> JsonObject | None:
        """The per-version route: this version's weights and row, never its registry entry.

        Answers the platform's own receipt row, as ``DELETE /model-versions/{id}`` does.
        """
        self.call_log.append(("purge_model_version", version_id))
        if self.no_purge_route:
            raise ModelNotFoundError(version_id)
        if version_id in self.purge_errors:
            raise self.purge_errors[version_id]
        self._take("model", version_id, ModelNotFoundError)
        return {"kind": "model_version", "id": version_id, "status": "deleted", "code": "deleted"}

    def get_training_job(self, job_id: str) -> JsonObject:
        self.call_log.append(("get_training_job", job_id))
        found = self._need("job", job_id, TrainingJobNotFoundError)
        ended = "completed" if job_id in self.finished else "running"
        return {**found, "status": self.statuses.get(job_id, ended)}

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

    def list_projects(self, **filter_params: str | int) -> JsonObject | str | None:
        self.call_log.append(("list_projects", str(filter_params.get("page"))))
        if self.project_reads_fail is not None:
            raise self.project_reads_fail
        if self.project_pages:
            return self.project_pages[int(filter_params["page"])]
        items: JsonArray = [{"id": pid} for pid in self.other_projects]
        return {"items": items, "pages": 1}

    def get_project_datasets(self, project_id: str) -> JsonObject:
        self.call_log.append(("get_project_datasets", project_id))
        if project_id in self.project_datasets:
            return self.project_datasets[project_id]
        linked: JsonArray = [{"id": i} for i in self.other_projects.get(project_id, [])]
        return {"training": linked, "validation": []}

    def delete_project(self, project_id: str) -> None:
        self.call_log.append(("delete_project", project_id))
        self._take("project", project_id, ProjectNotFoundError)


def as_cleanup_client(platform: FakeCleanup) -> CleanupClient:
    """The cleanup fake, typed as the protocol ``delete_audit`` takes."""
    return platform
