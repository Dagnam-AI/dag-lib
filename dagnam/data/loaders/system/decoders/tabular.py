"""Tabular system-dataset decoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import polars as pl

from dagnam.data.loaders.system.column_store import Column, ColumnStore
from dagnam.data.loaders.system.decoders.base import DecodeError


class TabularDecoder:
    """Decode csv/parquet files into declared eager columns."""

    def decode(self, artifact_dir: Path, layout: dict[str, object], split: str) -> ColumnStore:
        del split
        files = sorted([*artifact_dir.glob("*.csv"), *artifact_dir.glob("*.parquet")])
        if not files:
            raise DecodeError(f"tabular format: no csv/parquet artifact in {artifact_dir}")
        data = pl.read_parquet(files[0]) if files[0].suffix == ".parquet" else pl.read_csv(files[0])
        columns: dict[str, Column] = {}
        for name, raw_spec in layout.items():
            spec = cast("dict[str, Any]", raw_spec)
            source = spec.get("column") or name
            if not isinstance(source, str) or source not in data.columns:
                raise DecodeError(f"tabular format: column {name!r} source {source!r} missing")
            columns[name] = Column.eager(data[source].to_numpy())
        return ColumnStore(columns)
