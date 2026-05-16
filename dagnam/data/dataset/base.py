"""DagnamDataset core metadata and shared helpers."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import pandas as pd

from dagnam.data.dataset.to_flax import FlaxDatasetMixin
from dagnam.data.dataset.to_pandas import PandasDatasetMixin
from dagnam.data.dataset.to_pytorch import PytorchDatasetMixin
from dagnam.data.dataset.to_tensorflow import TensorflowDatasetMixin


class DagnamDataset(
    PandasDatasetMixin,
    PytorchDatasetMixin,
    TensorflowDatasetMixin,
    FlaxDatasetMixin,
):
    """Represents a loaded dataset with metadata and conversion methods.

    Data is loaded lazily; file parsing is deferred until a converter method
    such as ``to_pandas()`` or ``to_pytorch_loader()`` is called.
    """

    def __init__(
        self,
        meta: dict,
        data_dir: Path,
        _native_train: Any = None,
        _native_test: Any = None,
        _native_train_tf: Any = None,
        _native_test_tf: Any = None,
        _native_train_flax: Any = None,
        _native_test_flax: Any = None,
    ) -> None:
        self.id: str = meta["id"]
        self.name: str = meta["name"]
        self.format: str = meta["format"]
        self.dataset_type: str = meta["dataset_type"]
        self.num_samples: int = meta["num_samples"]
        self.num_classes: int = meta["num_classes"]
        self.feature_schema: dict | None = meta.get("feature_schema")
        self.class_names: list[str] | None = meta.get("class_names")
        self._data_dir: Path = data_dir
        self._data: pd.DataFrame | None = None
        self._native_train: Any = _native_train
        self._native_test: Any = _native_test
        # Framework-native dataset objects populated by system_loader when
        # tensorflow_datasets or jax-native loaders are available (16.72-bb/16.82-bb).
        self._native_train_tf: Any = _native_train_tf
        self._native_test_tf: Any = _native_test_tf
        self._native_train_flax: Any = _native_train_flax
        self._native_test_flax: Any = _native_test_flax
        self._raw_meta: dict = meta

    @property
    def info(self) -> dict:
        """Return a summary dictionary with the 8 required keys."""
        return {
            "id": self.id,
            "name": self.name,
            "format": self.format,
            "type": self.dataset_type,
            "samples": self.num_samples,
            "classes": self.num_classes,
            "class_names": self.class_names,
            "schema": self.feature_schema,
        }

    @staticmethod
    def _pad_sequences(sequences, maxlen: int = 200, num_words: int = 20000):
        """Pad/truncate variable-length integer sequences (e.g. IMDB)."""
        import numpy as np

        result = np.zeros((len(sequences), maxlen), dtype=np.int32)
        for i, seq in enumerate(sequences):
            filtered = [w if w < num_words else 0 for w in seq]
            trunc = filtered[:maxlen]
            result[i, : len(trunc)] = trunc
        return result

    def iter_samples(
        self,
        split: str = "train",
        decoded: bool = True,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        """Yield raw samples for a split when data is already available."""
        del decoded  # Reserved for future media decoding support.

        if isinstance(self._data, dict) and split in self._data:
            yield from self._data[split]
            return

        if isinstance(self._data, list):
            yield from self._data
            return

        native = self._native_test if split == "test" else self._native_train
        if isinstance(native, tuple) and len(native) == 2:
            features, labels = native
            for feature, label in zip(features, labels, strict=False):
                yield feature, label
            return

        if native is not None:
            for index in range(len(native)):
                yield native[index]
            return

        if self.format.lower() in ("csv", "tsv", "json", "jsonl"):
            yield from self._iter_tabular_file_samples(
                split=split,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )
            return

        raise ValueError(f"Raw sample iteration is not available for format '{self.format}'")

    def to_arrays(
        self,
        split: str = "train",
        decoded: bool = True,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        """Return features and labels as NumPy arrays for generated pipelines."""
        import numpy as np

        features = []
        labels = []
        for item in self.iter_samples(
            split=split,
            decoded=decoded,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        ):
            if isinstance(item, tuple) and len(item) == 2:
                feature, label = item
            else:
                feature, label = item, None
            features.append(feature)
            labels.append(label)

        if labels and all(label is not None for label in labels):
            return np.asarray(features), np.asarray(labels)
        return np.asarray(features), None

    def _iter_tabular_file_samples(
        self,
        split: str,
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ):
        """Yield numeric feature rows and encoded labels from a tabular file."""
        df = self.to_pandas()
        label_col = self._detect_label_column(df)
        labels = self._encode_label_values(df[label_col])
        feature_cols = [col for col in df.columns if col != label_col]
        features = df[feature_cols].select_dtypes(include="number").values

        indices = self._split_indices(
            len(df),
            split=split,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        for index in indices:
            yield features[index].tolist(), labels[index]

    def _detect_label_column(self, df: pd.DataFrame) -> str:
        if self.feature_schema and "columns" in self.feature_schema:
            for col_info in self.feature_schema["columns"]:
                if col_info.get("type") == "categorical":
                    return col_info["name"]
        return df.columns[-1]

    def _encode_label_values(self, series: pd.Series) -> list:
        if self.class_names:
            mapping = {name: idx for idx, name in enumerate(self.class_names)}
            return series.map(mapping).tolist()
        encoded, _ = pd.factorize(series)
        return encoded.tolist()

    @staticmethod
    def _split_indices(
        n: int,
        split: str,
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> list[int]:
        """Compute deterministic index ranges for train/val/test splits.

        Uses the same shuffle behavior as ``csv_loader`` / ``json_loader``
        so that ``to_arrays()`` and ``to_pytorch_loader()`` produce identical
        splits with the same seed.

        Contract: split order is defined by Python's stdlib
        ``random.Random(seed).shuffle``. Keep all framework loaders on this
        RNG unless the split contract is intentionally versioned.
        """
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        n_train = n - n_val - n_test

        indices = list(range(n))
        # Always shuffle for determinism parity with file-based loaders,
        # which shuffle unconditionally regardless of val/test ratios.
        random.Random(seed).shuffle(indices)

        split_map = {
            "train": indices[:n_train],
            "val": indices[n_train : n_train + n_val],
            "test": indices[n_train + n_val :],
        }
        return split_map[split]

    def _find_data_file(self) -> Path:
        """Locate the data file in ``_data_dir`` by glob pattern.

        Excludes ``meta.json`` from JSON matches.

        Raises:
            FileNotFoundError: If no matching file is found.
        """
        pattern_map: dict[str, list[str]] = {
            "csv": ["*.csv"],
            "tsv": ["*.tsv"],
            "json": ["*.json"],
            "jsonl": ["*.jsonl"],
        }

        fmt = self.format.lower()
        patterns = pattern_map.get(fmt, [f"*.{fmt}"])

        for pattern in patterns:
            for match in self._data_dir.glob(pattern):
                # Exclude meta.json from json matches
                if fmt == "json" and match.name == "meta.json":
                    continue
                return match

        raise FileNotFoundError(f"No data file matching {patterns} in {self._data_dir}")
