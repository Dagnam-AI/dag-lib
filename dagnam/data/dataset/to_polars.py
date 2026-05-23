"""Polars conversion for DagnamDataset."""

from __future__ import annotations

from typing import cast

import polars as pl
from typing_extensions import override

from dagnam.data.dataset._typing import DatasetMixinBase


class PolarsDatasetMixin(DatasetMixinBase):
    """Polars conversion methods."""

    @override
    def to_polars(self) -> pl.DataFrame:
        """Load the dataset as a polars DataFrame.

        The result is cached after the first call.  Supports CSV, TSV,
        JSON, and JSONL formats.

        Raises:
            ValueError: If the format is not supported.
            FileNotFoundError: If no matching data file exists in the cache.
        """
        if self._data is not None:
            if isinstance(self._data, pl.DataFrame):
                return self._data
            raise TypeError("Cached dataset payload is not a polars DataFrame")

        fmt = self.format.lower()
        if fmt not in ("csv", "tsv", "json", "jsonl"):
            raise ValueError(f"Cannot load format '{self.format}' as DataFrame")

        data_file = self._find_data_file()

        if fmt == "csv":
            self._data = pl.read_csv(data_file)
        elif fmt == "tsv":
            self._data = pl.read_csv(data_file, separator="\t")
        elif fmt == "json":
            self._data = pl.read_json(data_file)
        elif fmt == "jsonl":
            self._data = pl.read_ndjson(data_file)

        return cast(pl.DataFrame, self._data)
