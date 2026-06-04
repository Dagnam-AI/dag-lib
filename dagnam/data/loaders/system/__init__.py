"""Native system dataset loaders."""

from __future__ import annotations

from dagnam.data.loaders.system.common import SYSTEM_CACHE_ROOT
from dagnam.data.loaders.system.flax import resolve_system_dataset_flax
from dagnam.data.loaders.system.registry import (
    NATIVE_LOADERS,
    resolve_system_dataset,
)
from dagnam.data.loaders.system.tensorflow_datasets import (
    TFDS_NAME_MAP,
    resolve_system_dataset_tf,
    resolve_tfds_name,
)

__all__ = [
    "NATIVE_LOADERS",
    "SYSTEM_CACHE_ROOT",
    "TFDS_NAME_MAP",
    "resolve_system_dataset",
    "resolve_system_dataset_flax",
    "resolve_system_dataset_tf",
    "resolve_tfds_name",
]
