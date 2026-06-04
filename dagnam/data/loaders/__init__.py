"""Format-specific dataset loader modules.

The modules are intentionally not imported here so optional framework
dependencies are only required when a specific loader is used.
"""

from __future__ import annotations

# The submodules are imported lazily at runtime via ``__getattr__`` (PEP 562)
# so optional framework dependencies are only required when a specific loader
# is accessed. They are also listed under ``TYPE_CHECKING`` so pyright can
# resolve the names referenced in ``__all__`` (reportUnsupportedDunderAll).
import importlib
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dagnam.data.loaders import (
        audio as audio,
        csv as csv,
        flax as flax,
        image_folder as image_folder,
        json_array as json_array,
        media as media,
        system as system,
        tf as tf,
    )

__all__ = [
    "audio",
    "csv",
    "flax",
    "image_folder",
    "json_array",
    "media",
    "system",
    "tf",
]


def __getattr__(name: str) -> ModuleType:
    """Lazily import a loader submodule named in ``__all__`` (PEP 562).

    Keeping the import lazy means optional framework dependencies are only
    required when a specific loader is actually accessed.
    """
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
