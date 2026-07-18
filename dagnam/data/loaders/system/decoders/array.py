"""Array/npz system-dataset decoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

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
        # allow_pickle=False (the safe default): a server/author-supplied .npz is
        # untrusted, and object arrays deserialize through pickle — arbitrary code
        # execution inside np.load. Object-array columns raise ValueError on access
        # under this flag; we refuse them explicitly rather than run their pickle.
        with np.load(npz_files[0], allow_pickle=False) as data:
            for column_name, raw_spec in layout.items():
                spec = cast("dict[str, Any]", raw_spec)
                key = spec.get(key_field) or spec.get("key")
                if not isinstance(key, str):
                    raise DecodeError(f"array format: column {column_name!r} key {key!r} missing")
                if spec.get("ragged"):
                    array = self._decode_ragged(data, key, column_name)
                else:
                    if key not in data:
                        raise DecodeError(
                            f"array format: column {column_name!r} key {key!r} missing"
                        )
                    try:
                        array = np.asarray(data[key])
                    except ValueError as exc:
                        raise DecodeError(
                            f"array format: column {column_name!r} requires pickle "
                            "(object arrays are not supported)"
                        ) from exc
                columns[column_name] = Column.eager(array)
        return ColumnStore(columns)

    @staticmethod
    def _decode_ragged(data: Any, key: str, column_name: str) -> npt.NDArray[np.object_]:
        """Reconstruct a ragged column from pickle-free ``values``+``offsets`` arrays.

        A ragged column (variable-length rows, e.g. IMDB token sequences) cannot be
        stored as a single regular ndarray, and an object array would require pickle
        (disallowed). Instead the artifact stores two REGULAR int arrays — ``<key>_values``
        (every row's tokens concatenated) and ``<key>_offsets`` (row-start boundaries,
        length ``n_rows + 1``) — which load safely under ``allow_pickle=False``. We
        rebuild the per-row object array IN MEMORY from offset slices; this construction
        never deserializes untrusted bytes, so it is not a pickle/RCE risk.
        """
        values_key = f"{key}_values"
        offsets_key = f"{key}_offsets"
        if values_key not in data or offsets_key not in data:
            raise DecodeError(
                f"array format: ragged column {column_name!r} requires "
                f"{values_key!r} and {offsets_key!r} arrays"
            )
        values = np.asarray(data[values_key])
        offsets = np.asarray(data[offsets_key])
        if offsets.ndim != 1 or offsets.size < 1:
            raise DecodeError(f"array format: ragged column {column_name!r} has malformed offsets")
        n_rows = int(offsets.shape[0]) - 1
        rows = np.empty(n_rows, dtype=object)
        for i in range(n_rows):
            rows[i] = values[int(offsets[i]) : int(offsets[i + 1])]
        return rows
