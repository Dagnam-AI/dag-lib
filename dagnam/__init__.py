"""dagnam - Python client library for Dagnam.AI datasets."""

from __future__ import annotations

from dagnam._core.auth import configure, get_api_key, get_api_url
from dagnam._core.client import DagnamClient
from dagnam._core.config import get_config_value
from dagnam._core.exceptions import ChecksumError
from dagnam._core.load import load_dataset
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
from dagnam.data.dataset import DagnamDataset
from dagnam.resources import codegen, datasets, deployments, hub, projects
from dagnam.resources.checkpoints import download_checkpoint
from dagnam.resources.inference import deployment_health, inference, inference_batch
from dagnam.resources.training import TrainingEvent, stream_training

__version__ = "0.1.0"

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
    "save_checksum",
    "save_metadata",
    "stream_training",
    "touch_cache",
]
