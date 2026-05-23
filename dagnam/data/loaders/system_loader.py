"""Compatibility wrapper for ``dagnam.data.loaders.system``."""

from dagnam.data.loaders.system import *
from dagnam.data.loaders.system import NATIVE_LOADERS, SYSTEM_CACHE_ROOT

__all__ = ["NATIVE_LOADERS", "SYSTEM_CACHE_ROOT"]
