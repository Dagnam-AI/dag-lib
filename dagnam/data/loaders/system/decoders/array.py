"""Array/npz system-dataset decoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from dagnam.data.loaders.system.column_store import Column, ColumnStore
from dagnam.data.loaders.system.decoders.base import DecodeError


class ArrayDecoder:
    """Decode a declared npz layout into eager columns."""

    def decode(self, artifact_dir: Path, layout: dict[str, object], split: str) -> ColumnStore:
        npz_files = sorted(artifact_dir.glob("*.npz"))
        if not npz_files:
            raise DecodeError(f"array format: no .npz artifact in {artifact_dir}")

        key_field = "key" if split == "train" else "test_key"
        columns: dict[str, Column] = {}
        with np.load(npz_files[0], allow_pickle=True) as data:
            for column_name, raw_spec in layout.items():
                spec = cast("dict[str, Any]", raw_spec)
                key = spec.get(key_field) or spec.get("key")
                if not isinstance(key, str) or key not in data:
                    raise DecodeError(f"array format: column {column_name!r} key {key!r} missing")
                columns[column_name] = Column.eager(np.asarray(data[key]))
        return ColumnStore(columns)
