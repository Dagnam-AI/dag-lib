"""Shared paths for native system dataset loaders."""

from __future__ import annotations

from pathlib import Path

_SYSTEM_CACHE_ROOT = Path.home() / ".dagnam" / "system_datasets"
