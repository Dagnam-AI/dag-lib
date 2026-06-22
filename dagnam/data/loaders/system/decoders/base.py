"""Format decoder protocol for descriptor-driven system datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from dagnam.data.loaders.system.column_store import ColumnStore


class DecodeError(Exception):
    """Raised when a declared system-dataset format cannot be decoded."""


class FormatDecoder(Protocol):
    """Decode one artifact format into a ColumnStore split."""

    def decode(
        self,
        artifact_dir: Path,
        layout: dict[str, object],
        split: str,
    ) -> ColumnStore: ...
