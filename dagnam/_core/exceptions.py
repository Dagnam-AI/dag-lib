"""Custom exception classes for the dagnam library."""

from dagnam._contracts import ParamError


class DagnamError(Exception):
    """Base exception for all dagnam errors."""


class AuthError(DagnamError):
    """No API key found or authentication failed (401)."""


class DatasetNotFoundError(DagnamError):
    """Dataset ID not found (404)."""

    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        super().__init__(f"Dataset '{dataset_id}' not found")


class DeploymentNotFoundError(DagnamError):
    """Deployment ID not found (404)."""

    def __init__(self, deployment_id: str):
        self.deployment_id = deployment_id
        super().__init__(f"Deployment '{deployment_id}' not found")


class TrainingJobNotFoundError(DagnamError):
    """Training job ID not found (404)."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        super().__init__(f"Training job '{job_id}' not found")


class CheckpointNotFoundError(DagnamError):
    """Checkpoint ID not found (404)."""

    def __init__(self, checkpoint_id: str):
        self.checkpoint_id = checkpoint_id
        super().__init__(f"Checkpoint '{checkpoint_id}' not found")


class CheckpointError(DagnamError):
    """Checkpoint download or verification failed (other than 404)."""


class APIError(DagnamError):
    """General API communication failure."""

    def __init__(self, status_code: int, message: str, *, retry_after_header: str | None = None):
        self.status_code = status_code
        self.message = message
        self.retry_after_header = retry_after_header
        super().__init__(f"API error {status_code}: {message}")


class DownloadTooLargeError(APIError):
    """A download exceeded the configured ``max_download_bytes`` ceiling."""


class ResponseError(APIError):
    """Server returned a malformed, undecodable, or wrong-shape response body."""


class ChecksumError(DagnamError):
    """Downloaded file checksum does not match server-reported checksum."""


class StreamError(DagnamError):
    """SSE stream transport failure (after retry exhaustion)."""


# ---------------------------------------------------------------------------
# Phase 4 — Tier 2 domain exceptions
# ---------------------------------------------------------------------------


class HubError(DagnamError):
    """Generic Hub operation failure."""


class HubModelNotFoundError(DagnamError):
    """Hub model ID not found (404)."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Hub model '{model_id}' not found")


class ProjectNotFoundError(DagnamError):
    """Project ID not found (404)."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        super().__init__(f"Project '{project_id}' not found")


class ArchitectureVersionNotFoundError(DagnamError):
    """Architecture version ID not found (404)."""

    def __init__(self, version_id: str):
        self.version_id = version_id
        super().__init__(f"Architecture version '{version_id}' not found")


class DeploymentValidationError(DagnamError):
    """Deployment payload failed server-side validation (422/400)."""


class DeploymentStateError(DagnamError):
    """Action not permitted for deployment's current lifecycle state (409)."""


class CodegenError(DagnamError):
    """Codegen operation failed."""


class CodegenValidationError(DagnamError):
    """Codegen validation returned errors (422)."""


class ArchitectureValidationError(DagnamError):
    """An architecture failed SDK-local validation before persist.

    Raised by ``resources.save_architecture`` so a model the Studio would
    reject never reaches the backend. ``errors`` carries the structured
    per-node param failures.
    """

    def __init__(self, errors: list[ParamError]) -> None:
        self.errors = errors
        preview = "; ".join(e.message for e in errors[:3])
        suffix = "" if len(errors) <= 3 else f" (+{len(errors) - 3} more)"
        super().__init__(f"Architecture validation failed: {preview}{suffix}")


class UploadError(DagnamError):
    """Dataset upload failed (transport or server side)."""


class QuotaExceededError(DagnamError):
    """Plan/usage limit reached: a storage quota (413) or a plan resource limit (402)."""


class TaskNotFoundError(DagnamError):
    """Async task ID not found (404)."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        super().__init__(f"Task '{task_id}' not found")


class LROTimeoutError(DagnamError):
    """Long-running operation did not reach a terminal state within the timeout."""


class LROFailedError(DagnamError):
    """Long-running operation reached a failure state."""

    def __init__(self, state: str, detail: str | None = None):
        self.state = state
        self.detail = detail
        msg = f"Operation entered failure state '{state}'"
        if detail:
            msg = f"{msg}: {detail}"
        super().__init__(msg)
