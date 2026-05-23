"""Format-specific dataset loader modules.

The modules are intentionally not imported here so optional framework
dependencies are only required when a specific loader is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dagnam.data.loaders import audio as audio
    from dagnam.data.loaders import csv as csv
    from dagnam.data.loaders import flax as flax
    from dagnam.data.loaders import image_folder as image_folder
    from dagnam.data.loaders import json_array as json_array
    from dagnam.data.loaders import media as media
    from dagnam.data.loaders import system as system
    from dagnam.data.loaders import tf as tf

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
