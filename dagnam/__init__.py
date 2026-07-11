"""dagnam - Python client library for Dagnam.AI datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dagnam._core.auth import configure, get_api_key, get_api_url
from dagnam._core.client import DagnamClient
from dagnam._core.config import get_config_value
from dagnam._core.lro import LongRunningOperation
from dagnam.data.cache import (
    compute_file_checksum,
    evict_lru,
    get_cache_dir,
    is_cached,
    load_metadata,
    save_checksum,
    save_metadata,
    touch_cache,
)
from dagnam.exceptions import (
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
    HubError,
    HubModelNotFoundError,
    LROFailedError,
    LROTimeoutError,
    ProjectNotFoundError,
    QuotaExceededError,
    StreamError,
    TaskNotFoundError,
    TrainingJobNotFoundError,
    UploadError,
)
from dagnam.resources import account, codegen, datasets, deployments, hub, projects, studio
from dagnam.resources.checkpoints import download_checkpoint
from dagnam.resources.datasets import (
    delete_dataset,
    preview_dataset,
    update_dataset,
    update_dataset_roles,
)
from dagnam.resources.inference import (
    deployment_health,
    inference,
    inference_batch,
    inference_schema,
    inference_stream,
)
from dagnam.resources.projects import (
    download_project_thumbnail,
    upload_project_thumbnail,
)
from dagnam.resources.training import (
    TrainingEvent,
    allowed_strategies,
    cancel_training_job,
    create_training_job,
    delete_training_jobs,
    download_code,
    download_dag,
    estimate_resources,
    get_training_job,
    list_training_jobs,
    restart,
    restore_checkpoint,
    stream_training,
    training_logs,
    training_metrics,
    training_metrics_summary,
)
from dagnam.training import (
    init,
    report_error,
    report_log,
    report_metric,
    report_progress,
    report_system,
    write_training_state,
)

__version__ = "0.6.0"

if TYPE_CHECKING:
    # Declared for type checkers and ``__all__``; loaded lazily at runtime via
    # ``__getattr__`` (see ``_LAZY_EXPORTS``) to keep import time low.
    from dagnam._core.aio import AsyncDagnamClient
    from dagnam.data.dataset import DagnamDataset
    from dagnam.data.load import load_dataset

_LAZY_EXPORTS = {
    "AsyncDagnamClient": ("dagnam._core.aio", "AsyncDagnamClient"),
    "DagnamDataset": ("dagnam.data.dataset", "DagnamDataset"),
    "load_dataset": ("dagnam.data.load", "load_dataset"),
}

__all__ = [
    "APIError",
    "ArchitectureValidationError",
    "ArchitectureVersionNotFoundError",
    "AsyncDagnamClient",
    "AuthError",
    "CheckpointError",
    "CheckpointNotFoundError",
    "ChecksumError",
    "CodegenError",
    "CodegenValidationError",
    "DagnamClient",
    "DagnamDataset",
    "DagnamError",
    "DatasetNotFoundError",
    "DeploymentNotFoundError",
    "DeploymentStateError",
    "DeploymentValidationError",
    "HubError",
    "HubModelNotFoundError",
    "LROFailedError",
    "LROTimeoutError",
    "LongRunningOperation",
    "ProjectNotFoundError",
    "QuotaExceededError",
    "StreamError",
    "TaskNotFoundError",
    "TrainingEvent",
    "TrainingJobNotFoundError",
    "UploadError",
    "__version__",
    "account",
    "allowed_strategies",
    "cancel_training_job",
    "codegen",
    "compute_file_checksum",
    "configure",
    "create_training_job",
    "datasets",
    "delete_dataset",
    "delete_training_jobs",
    "deployment_health",
    "deployments",
    "download_checkpoint",
    "download_code",
    "download_dag",
    "download_project_thumbnail",
    "estimate_resources",
    "evict_lru",
    "get_api_key",
    "get_api_url",
    "get_cache_dir",
    "get_config_value",
    "get_training_job",
    "hub",
    "inference",
    "inference_batch",
    "inference_schema",
    "inference_stream",
    "init",
    "is_cached",
    "list_training_jobs",
    "load_dataset",
    "load_metadata",
    "preview_dataset",
    "projects",
    "report_error",
    "report_log",
    "report_metric",
    "report_progress",
    "report_system",
    "restart",
    "restore_checkpoint",
    "save_checksum",
    "save_metadata",
    "stream_training",
    "studio",
    "touch_cache",
    "training_logs",
    "training_metrics",
    "training_metrics_summary",
    "update_dataset",
    "update_dataset_roles",
    "upload_project_thumbnail",
    "write_training_state",
]


def __getattr__(name: str) -> Any:
    """Load optional/heavy public exports only when callers request them."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
