"""Public exception hierarchy for the ``dagnam`` SDK.

Every error the SDK can raise is a subclass of :class:`DagnamError`, so callers
can catch the whole family with ``except dagnam.DagnamError`` or a specific type
with e.g. ``except dagnam.APIError``. These names are re-exported at the package
top level too (``dagnam.APIError``); import them from here or from ``dagnam``
directly — never from the private ``dagnam._core.exceptions`` path.
"""

from __future__ import annotations

from dagnam._core.exceptions import (
    AccountLockedError,
    AccountSuspendedError,
    APIError,
    ArchitectureValidationError,
    ArchitectureVersionNotFoundError,
    AuthError,
    CheckpointError,
    CheckpointNotFoundError,
    ChecksumError,
    CodegenError,
    CodegenValidationError,
    DagnamError,
    DatasetNotFoundError,
    DeploymentNotFoundError,
    DeploymentStateError,
    DeploymentValidationError,
    EmailNotVerifiedError,
    HubError,
    HubModelNotFoundError,
    InvalidURLError,
    LROFailedError,
    LROTimeoutError,
    PayloadTooLargeError,
    ProjectNotFoundError,
    QuotaExceededError,
    ResponseError,
    StreamError,
    TaskNotFoundError,
    TrainingJobNotFoundError,
    UploadError,
)

__all__ = [
    "APIError",
    "AccountLockedError",
    "AccountSuspendedError",
    "ArchitectureValidationError",
    "ArchitectureVersionNotFoundError",
    "AuthError",
    "CheckpointError",
    "CheckpointNotFoundError",
    "ChecksumError",
    "CodegenError",
    "CodegenValidationError",
    "DagnamError",
    "DatasetNotFoundError",
    "DeploymentNotFoundError",
    "DeploymentStateError",
    "DeploymentValidationError",
    "EmailNotVerifiedError",
    "HubError",
    "HubModelNotFoundError",
    "InvalidURLError",
    "LROFailedError",
    "LROTimeoutError",
    "PayloadTooLargeError",
    "ProjectNotFoundError",
    "QuotaExceededError",
    "ResponseError",
    "StreamError",
    "TaskNotFoundError",
    "TrainingJobNotFoundError",
    "UploadError",
]
