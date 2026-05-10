"""Native system dataset loaders."""

from __future__ import annotations

from dagnam.data.loaders.system.common import _SYSTEM_CACHE_ROOT
from dagnam.data.loaders.system.flax import resolve_system_dataset_flax
from dagnam.data.loaders.system.registry import (
    _NATIVE_LOADERS,
    resolve_system_dataset,
)
from dagnam.data.loaders.system.tensorflow_datasets import (
    _TFDS_NAME_MAP,
    _resolve_tfds_name,
    resolve_system_dataset_tf,
)

__all__ = [
    "_NATIVE_LOADERS",
    "_SYSTEM_CACHE_ROOT",
    "_TFDS_NAME_MAP",
    "_resolve_tfds_name",
    "resolve_system_dataset",
    "resolve_system_dataset_flax",
    "resolve_system_dataset_tf",
]
