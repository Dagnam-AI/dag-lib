"""Custom exception classes for the dagnam library."""

from collections.abc import Mapping

from dagnam_contracts import ParamError

from dagnam._types import JsonObject


class DagnamError(Exception):
    """Base exception for all dagnam errors."""


class AuthError(DagnamError):
    """Authentication failed (401) or the request was refused by an access control (403).

    Covers both "no API key found / invalid or expired key" and a server-side
    access control that refuses the request outright (for example a blocked
    source IP address), where no programmatic retry can succeed.
    """


class DatasetNotFoundError(DagnamError):
    """Dataset ID not found (404), or no built-in dataset has that name."""

    def __init__(self, dataset_id: str, hint: str | None = None):
        self.dataset_id = dataset_id
        message = f"Dataset '{dataset_id}' not found"
        super().__init__(f"{message}. {hint}" if hint else message)


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


class FoundationRunNotFoundError(DagnamError):
    """Foundation fine-tuning run ID not found (404).

    Also raised for a run that exists but belongs to someone else: the API
    answers one 404 for both, so a caller cannot use this error to learn that
    an id is real.
    """

    def __init__(self, run_id: str):
        self.run_id = run_id
        super().__init__(f"Foundation run '{run_id}' not found")


class RunFailedError(DagnamError):
    """A foundation run reached a terminal state other than ``completed``.

    ``status`` is the server's terminal value (``failed``, ``cancelled`` or
    ``timeout``) and ``reason`` its ``error_message`` when it gave one.
    """

    def __init__(self, run_id: str, status: str, reason: str | None = None):
        self.run_id = run_id
        self.status = status
        self.reason = reason
        msg = f"Foundation run '{run_id}' {status}"
        if reason:
            msg = f"{msg}: {reason}"
        super().__init__(msg)


class EvaluationRunNotFoundError(DagnamError):
    """Evaluation run ID not found (404).

    Also raised for a run that exists but belongs to someone else: the API
    answers one 404 for both, so a caller cannot use this error to learn that
    an id is real.
    """

    def __init__(self, run_id: str):
        self.run_id = run_id
        super().__init__(f"Evaluation run '{run_id}' not found")


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
        self.retry_after_cap: float | None = None
        """A larger ceiling than the client's backoff cap for honouring ``retry_after_header``, when
        the platform's answer says the wait is worth it (see ``idempotency_unavailable``)."""
        super().__init__(f"API error {status_code}: {message}")


class PlatformAnswerError(DagnamError):
    """The platform answered 200 with a document that lacks what the answer is for."""


class TeardownInProgressError(APIError):
    """Another cancel or delete of the same audit is still walking it (``409 teardown_in_progress``).

    The platform holds a lock for the walk; the caller waits (``Retry-After`` says how long)
    and asks again.
    """


class TrainingStateError(APIError):
    """A training-job request was refused because of the job's current state (``409``).

    ``reason`` is the platform's machine-readable marker when it sent one:
    ``not_paused`` (resume of a job that is not paused), ``checkpoint_unavailable`` (the paused
    job's saved checkpoint cannot be read) or ``not_accepting_checkpoints`` (a checkpoint push
    to a job that is not running). A 409 without a marker, such as a resume while the previous
    run is still stopping, has ``reason=None``; ``message`` is the platform's own sentence.
    """

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        self.reason = reason
        super().__init__(409, message)


class EndpointsServingError(APIError):
    """A delete stopped, having changed nothing, because an endpoint is still serving (``409 endpoints_serving``).

    Deleting would break any app that calls the endpoint. ``message`` is the sentence that says
    so and what to do instead; ``endpoints`` are the blocking endpoints as objects with ``id``,
    ``name``, ``status`` (``running`` reads as serving, anything else as rolling out) and
    ``last_request_at`` (``None`` when the platform has no request on record; absent when this
    client found the endpoint itself). Unlike
    :class:`TeardownInProgressError` waiting does not help: stop the endpoints, or delete again
    with ``include_endpoints=True``.
    """

    def __init__(
        self, message: str, endpoints: list[JsonObject], *, from_platform: bool = True
    ) -> None:
        self.endpoints = endpoints
        self.from_platform = from_platform
        """``False`` when this client found the endpoints itself, so it knows only their status."""
        super().__init__(409, message)


class DownloadTooLargeError(APIError):
    """A download exceeded the configured ``max_download_bytes`` ceiling."""


class ResponseError(APIError):
    """Server returned a malformed, undecodable, or wrong-shape response body."""


class EmailNotVerifiedError(APIError):
    """A dispatch action was refused because the account's email is unverified (403).

    Expensive actions (training-job create, dataset upload, deployment create)
    require the API key's owning account to have a verified email. Verify it via
    the web app, then retry. ``verification_url`` carries the server-provided
    link when it is present and safe to surface.

    A subclass of :class:`APIError` (these were plain ``APIError`` 403s before
    the dedicated class existed) so existing ``except APIError`` handlers still
    catch it and ``.status_code`` is populated.
    """

    def __init__(self, message: str, *, verification_url: str | None = None) -> None:
        self.verification_url = verification_url
        super().__init__(403, message)


class AccountSuspendedError(APIError):
    """The API key's owning account was administratively suspended (403).

    Raised on any authenticated call while the account is suspended or banned
    by an administrator. Not self-clearing — contact support or the account
    owner; retrying the same call will not succeed on its own. A subclass of
    :class:`APIError` so existing ``except APIError`` handlers still catch it.
    """

    def __init__(self, message: str) -> None:
        super().__init__(403, message)


class AccountLockedError(APIError):
    """The API key's owning account is temporarily locked out (423).

    Raised while a login-lockout window (too many failed interactive login
    attempts) is active on the account. Unlike ``AccountSuspendedError`` this is
    TEMPORARY and self-clearing once the lockout window elapses — kept as a
    distinct class (rather than reusing AccountSuspendedError) so a caller can
    tell "wait and retry" apart from "requires support/admin action" without
    parsing the message text. A subclass of :class:`APIError` so existing
    ``except APIError`` handlers still catch it.
    """

    def __init__(self, message: str) -> None:
        super().__init__(423, message)


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


class ModelError(DagnamError):
    """Generic registry model operation failure."""


class VersionKeptError(ModelError):
    """A version purge the platform refused and answered as a ``kept`` decision (409).

    ``row`` is the receipt row the platform sent in its ``detail`` (``error``, ``message``,
    ``kind``, ``id``, ``status``, ``code``): the weights are a live endpoint's, not ours to remove.
    """

    def __init__(self, row: Mapping[str, object]):
        self.row = row
        super().__init__(str(row.get("message") or row.get("error") or "the purge was refused"))


class ModelNotFoundError(DagnamError):
    """Registry model entry, version, or artifact ID not found (404)."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model '{model_id}' not found")


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


class InvalidURLError(UploadError):
    """Dataset-from-URL ingest rejected the URL as invalid or unsafe (4xx).

    Raised when the backend refuses a ``upload_dataset_from_url`` source URL
    (e.g. an SSRF-blocked host, or a disallowed scheme). A subclass of
    :class:`UploadError` so existing handlers still catch it.
    """


class QuotaExceededError(DagnamError):
    """Plan/usage limit reached: a storage quota (413) or a plan resource limit (402)."""


class InsufficientCreditsError(QuotaExceededError):
    """The account's credits do not cover the work (402 ``insufficient_credits``).

    ``message`` is the platform's own complete sentence and is also ``str(exc)``; it is printed
    once and nothing is appended to it. ``required_credits`` is what the work needs to start;
    ``available_credits`` is the balance, ``None`` when the platform did not disclose it (the
    caller is not the account's owner). ``next_steps`` are the platform's tokens for what to do
    about it (for example ``request_credits``, ``top_up_credits``, ``add_payment_method``,
    ``reduce_job_size``).

    A subclass of :class:`QuotaExceededError` so existing ``except QuotaExceededError`` handlers
    still catch it.
    """

    def __init__(
        self,
        message: str,
        *,
        required_credits: int,
        available_credits: int | None = None,
        next_steps: tuple[str, ...] = (),
    ) -> None:
        self.message = message
        self.required_credits = required_credits
        self.available_credits = available_credits
        self.next_steps = next_steps
        super().__init__(message)


class PayloadTooLargeError(QuotaExceededError):
    """Upload rejected for exceeding the server's per-request size cap (413).

    A subclass of :class:`QuotaExceededError` so existing ``except
    QuotaExceededError`` handlers still catch it, while callers who care about
    the size-cap case specifically can catch this narrower type.
    """


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
