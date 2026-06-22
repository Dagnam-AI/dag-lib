"""Plain-text corpus system-dataset decoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from dagnam.data.loaders.system.column_store import Column, ColumnStore
from dagnam.data.loaders.system.decoders.base import DecodeError


class TextDecoder:
    """Decode a declared text file into a non-empty text column."""

    def decode(self, artifact_dir: Path, layout: dict[str, object], split: str) -> ColumnStore:
        del split
        raw_spec = layout.get("text")
        if not isinstance(raw_spec, dict):
            raise DecodeError("text format requires text layout")
        spec = cast("dict[str, Any]", raw_spec)
        filename = spec.get("file")
        if not isinstance(filename, str):
            raise DecodeError("text format requires layout.text.file")
        path = artifact_dir / filename
        if not path.exists():
            raise DecodeError(f"text format: file does not exist: {path}")
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        return ColumnStore({"text": Column.eager(np.asarray(lines, dtype=object))})
