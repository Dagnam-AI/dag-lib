"""pandas conversion for DagnamDataset."""

from __future__ import annotations

import pandas as pd


class PandasDatasetMixin:
    """Pandas conversion methods."""

    def to_pandas(self) -> pd.DataFrame:
        """Load the dataset as a pandas DataFrame.

        The result is cached after the first call.  Supports CSV, TSV,
        JSON, and JSONL formats.

        Raises:
            ValueError: If the format is not supported.
            FileNotFoundError: If no matching data file exists in the cache.
        """
        if self._data is not None:
            return self._data

        fmt = self.format.lower()
        if fmt not in ("csv", "tsv", "json", "jsonl"):
            raise ValueError(f"Cannot load format '{self.format}' as DataFrame")

        data_file = self._find_data_file()

        if fmt == "csv":
            self._data = pd.read_csv(data_file)
        elif fmt == "tsv":
            self._data = pd.read_csv(data_file, sep="\t")
        elif fmt == "json":
            self._data = pd.read_json(data_file)
        elif fmt == "jsonl":
            self._data = pd.read_json(data_file, lines=True)

        return self._data
