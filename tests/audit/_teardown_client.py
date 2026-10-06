"""A client over the teardown world: the designed platform's two audit routes plus the generic ones.

The generic routes (delete a dataset, cancel a run, ...) are what an SDK that acts on its own
reaches. They answer as the platform's slices do (a running job cannot be deleted, a dataset a
run still names answers 409) and log every destructive act under the SDK's actor, so a test can
state that the SDK made none for a published audit. ``delete_model_entry`` exists so that a client
which still deletes a whole registry entry is caught doing it.
"""

from __future__ import annotations

from tests.audit._teardown_world import MISSING, SDK, TeardownPlatform, World

from dagnam._core.exceptions import (
    APIError,
    DatasetNotFoundError,
    DeploymentNotFoundError,
    DeploymentStateError,
    ModelNotFoundError,
    ProjectNotFoundError,
    TrainingJobNotFoundError,
)
from dagnam._types import JsonArray, JsonObject
from dagnam.audit.cleanup_kinds import CleanupClient

CLAIM_MAX = 200


class WorldClient:
    """Implements :class:`CleanupClient` over a :class:`World`."""

    api_url = "https://x"

    def __init__(self, platform: TeardownPlatform) -> None:
        self.platform = platform
        self.identity = "key-a"
        self.world: World = platform.world
        self.audit_failure: str | None = None
        """``"500"``: the next audit route does not answer; ``"404"``: it answers 404 (a key that
        cannot see the audit); ``"crash1"``: a delete walk dies after one step."""
        self.claim_failure = False
        """The next claim request fails as a whole (a 5xx)."""
        self.calls: list[str] = []

    # ------------------------------------------------------------------ the audit's own routes
    def _answer(self, name: str, audit_id: str) -> None:
        self.calls.append(f"{name}:{audit_id}")
        if self.audit_failure in ("500", "404"):
            status, self.audit_failure = int(self.audit_failure), None
            raise APIError(status, "server error" if status == 500 else "audit not found")

    def delete_audit(self, audit_id: str) -> JsonObject:
        self._answer("delete_audit", audit_id)
        crash = 1 if self.audit_failure == "crash1" else None
        self.audit_failure = None
        return self.platform.delete(audit_id, crash_at=crash)

    def cancel_audit(self, audit_id: str) -> JsonObject:
        self._answer("cancel_audit", audit_id)
        return self.platform.cancel(audit_id)

    def claim_audit_resources(self, audit_id: str, entries: JsonArray) -> JsonObject:
        """Take the ids that are the caller's and tagged by no audit; refuse the rest.

        A dataset the owner linked to another project is refused, and a claimed run takes its
        version with it. The route knows four kinds; ``model_version`` is a 422 for the request.
        """
        self.calls.append(f"claim_audit_resources:{audit_id}")
        if self.claim_failure:
            self.claim_failure = False
            raise APIError(500, "server error")
        kinds = {"dataset", "training_job", "deployment", "project"}
        listed = [e for e in entries if isinstance(e, dict)]
        if len(entries) > CLAIM_MAX or any(e["kind"] not in kinds for e in listed):
            raise APIError(422, "claim request refused")
        rows: JsonArray = []
        for entry in listed:
            rid = str(entry["id"])
            found = self.world.get(rid)
            project = self.world.audits[audit_id].project
            linked = found is not None and found.kind == "dataset" and bool(found.links - {project})
            ok = found is not None and found.tag is None and not linked
            if ok and found is not None:
                found.tag = audit_id
                inherited = self.world.get(found.version or "")
                if inherited is not None and inherited.tag is None:
                    inherited.tag = audit_id
            rows.append(
                {
                    "kind": entry["kind"],
                    "id": rid,
                    "result": "claimed" if ok else "refused",
                    "code": "claimed" if ok else "in_use_elsewhere" if linked else "not_claimable",
                }
            )
        return {"results": rows}

    # ------------------------------------------------------------------ the generic routes
    def _owned(self, kind: str, rid: str) -> bool:
        found = self.world.get(rid)
        return found is not None and found.kind == kind

    def cancel_training_job(self, job_id: str) -> JsonObject:
        self.calls.append(f"cancel_training_job:{job_id}")
        if not self._owned("training_job", job_id):
            raise TrainingJobNotFoundError(job_id)
        job = self.world.res[job_id]
        if job.status != "running":
            raise APIError(400, "Cannot cancel job with status completed")
        job.status = "cancelled"
        self.world.destroyed.append((SDK, "stop", job_id))
        return {"id": job_id, "status": "cancelled"}

    def pause_deployment(self, deployment_id: str) -> JsonObject:
        self.calls.append(f"pause_deployment:{deployment_id}")
        if not self._owned("deployment", deployment_id):
            raise DeploymentNotFoundError(deployment_id)
        self.world.res[deployment_id].paused = True
        self.world.destroyed.append((SDK, "pause", deployment_id))
        return {"id": deployment_id, "status": "paused"}

    def get_deployment(self, deployment_id: str) -> JsonObject:
        if not self._owned("deployment", deployment_id):
            raise DeploymentNotFoundError(deployment_id)
        return {"id": deployment_id, "status": "running"}

    def delete_deployment(self, deployment_id: str) -> JsonObject | None:
        self.calls.append(f"delete_deployment:{deployment_id}")
        if not self._owned("deployment", deployment_id):
            raise DeploymentNotFoundError(deployment_id)
        if self.world.res[deployment_id].status == "deploying":
            raise DeploymentStateError("Cannot delete a deployment that is still deploying")
        self.world.destroy(SDK, deployment_id)
        return None

    def get_model_version(self, version_id: str) -> JsonObject:
        found = self.world.res.get(version_id)
        if found is None or found.kind != "model_version" or not found.alive:
            raise ModelNotFoundError(version_id)
        return {
            "id": version_id,
            "entry_id": f"entry-{version_id}",
            "status": "revoked" if found.revoked else "ready",
        }

    def delete_model_entry(self, model_id: str) -> None:
        """The registry entry's soft delete: the row goes, the bytes stay, every version goes with it."""
        self.calls.append(f"delete_model_entry:{model_id}")
        vid = model_id.removeprefix("entry-")
        self.world.res[vid].alive = False
        self.world.destroyed.append((SDK, "model_entry", vid))

    def purge_model_version(self, version_id: str) -> JsonObject | None:
        """This one version: its row is revoked and its bytes removed; nothing around it is touched."""
        self.calls.append(f"purge_model_version:{version_id}")
        found = self.world.res.get(version_id)
        if found is None or found.kind != "model_version" or not found.alive:
            raise ModelNotFoundError(version_id)
        found.revoked = True
        if found.bytes:
            found.bytes = False
            self.world.destroyed.append((SDK, "weights", version_id))
        return {"kind": "model_version", "id": version_id, "status": "deleted", "code": "deleted"}

    def get_training_job(self, job_id: str) -> JsonObject:
        if not self._owned("training_job", job_id):
            raise TrainingJobNotFoundError(job_id)
        return {"id": job_id, "status": self.world.res[job_id].status}

    def bulk_delete_training_jobs(self, job_ids: list[str]) -> JsonObject:
        self.calls.append("bulk_delete_training_jobs:" + ",".join(job_ids))
        errors: JsonArray = []
        deleted = 0
        for job_id in job_ids:
            if not self._owned("training_job", job_id):
                errors.append({"job_id": job_id, "error": "Not found or not authorized"})
            elif self.world.res[job_id].status == "running":
                errors.append({"job_id": job_id, "error": "Cannot delete job with status running"})
            else:
                self.world.destroy(SDK, job_id)
                deleted += 1
        return {"deleted": deleted, "errors": errors}

    def get_dataset_meta(self, dataset_id: str, version: str | None = None) -> JsonObject:
        if not self._owned("dataset", dataset_id):
            raise DatasetNotFoundError(dataset_id)
        return {"id": dataset_id}

    def delete_dataset(self, dataset_id: str) -> None:
        self.calls.append(f"delete_dataset:{dataset_id}")
        if not self._owned("dataset", dataset_id):
            raise DatasetNotFoundError(dataset_id)
        if self.world.runs_naming(dataset_id):
            raise APIError(409, "Dataset is referenced by a training run and cannot be deleted")
        self.world.destroy(SDK, dataset_id)

    def get_project(self, project_id: str) -> JsonObject:
        if not self._owned("project", project_id):
            raise ProjectNotFoundError(project_id)
        return {"id": project_id}

    def list_projects(self, **filter_params: str | int) -> JsonObject | str | None:
        """Every project of the owner's, on one page."""
        self.calls.append("list_projects")
        items: JsonArray = [{"id": r.id} for r in self.world.alive("project")]
        return {"items": items, "pages": 1, "total": len(items)}

    def get_project_datasets(self, project_id: str) -> JsonObject:
        """The datasets linked into one project, grouped by role."""
        self.calls.append(f"get_project_datasets:{project_id}")
        linked: JsonArray = [
            {"id": r.id} for r in self.world.alive("dataset") if project_id in r.links
        ]
        return {"training": linked}

    def delete_project(self, project_id: str) -> None:
        self.calls.append(f"delete_project:{project_id}")
        if not self._owned("project", project_id):
            raise ProjectNotFoundError(project_id)
        self.world.destroy(SDK, project_id)


def as_cleanup_client(client: WorldClient) -> CleanupClient:
    """The world client, typed as the protocol the cleanup functions take."""
    return client


__all__ = ["MISSING", "WorldClient", "as_cleanup_client"]
