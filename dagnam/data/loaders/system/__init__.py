"""Generic descriptor-driven system dataset loaders."""

from __future__ import annotations

from dagnam.data.loaders.system.common import SYSTEM_CACHE_ROOT
from dagnam.data.loaders.system.dispatch import detect_installed_framework, load_system_dataset

resolve_system_dataset = load_system_dataset

__all__ = [
    "SYSTEM_CACHE_ROOT",
    "detect_installed_framework",
    "load_system_dataset",
    "resolve_system_dataset",
]
