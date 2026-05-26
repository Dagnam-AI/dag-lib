"""dagnam - Python client library for Dagnam.AI datasets."""

from __future__ import annotations

from typing import Any

from dagnam._core.auth import configure, get_api_key, get_api_url
from dagnam._core.client import DagnamClient
from dagnam._core.config import get_config_value
from dagnam._core.exceptions import ChecksumError
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
from dagnam.resources import codegen, datasets, deployments, hub, projects
from dagnam.resources.checkpoints import download_checkpoint
from dagnam.resources.inference import deployment_health, inference, inference_batch
from dagnam.resources.training import TrainingEvent, stream_training
from dagnam.training import (
    report_error,
    report_log,
    report_metric,
    report_progress,
    report_system,
    write_training_state,
)

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "DagnamDataset": ("dagnam.data.dataset", "DagnamDataset"),
    "load_dataset": ("dagnam._core.load", "load_dataset"),
}

__all__ = [
    "ChecksumError",
    "DagnamClient",
    "DagnamDataset",
    "LongRunningOperation",
    "TrainingEvent",
    "__version__",
    "codegen",
    "compute_file_checksum",
    "configure",
    "datasets",
    "deployment_health",
    "deployments",
    "download_checkpoint",
    "evict_lru",
    "get_api_key",
    "get_api_url",
    "get_cache_dir",
    "get_config_value",
    "hub",
    "inference",
    "inference_batch",
    "is_cached",
    "load_dataset",
    "load_metadata",
    "projects",
    "report_error",
    "report_log",
    "report_metric",
    "report_progress",
    "report_system",
    "save_checksum",
    "save_metadata",
    "stream_training",
    "touch_cache",
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
