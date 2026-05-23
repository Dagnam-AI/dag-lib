"""DagnamDataset core metadata and shared helpers."""

from __future__ import annotations

from pathlib import Path
import random
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING
from typing import cast
from typing_extensions import override

import numpy as np
import numpy.typing as npt
import polars as pl

from dagnam._types import (
    JsonObject,
    JsonValue,
    NativeSplit,
    TensorflowDataset,
    ensure_json_object,
)
from dagnam.data._polars_utils import factorize, numeric_columns
from dagnam.data.dataset.to_flax import FlaxDatasetMixin
from dagnam.data.dataset.to_polars import PolarsDatasetMixin
from dagnam.data.dataset.to_pytorch import PytorchDatasetMixin
from dagnam.data.dataset.to_tensorflow import TensorflowDatasetMixin

if TYPE_CHECKING:
    from dagnam.data.loaders.flax import FlaxBatch

LabelSeries = pl.Series


class DagnamDataset(
    PolarsDatasetMixin,
    PytorchDatasetMixin,
    TensorflowDatasetMixin,
    FlaxDatasetMixin,
):
    """Represents a loaded dataset with metadata and conversion methods.

    Data is loaded lazily; file parsing is deferred until a converter method
    such as ``to_polars()`` or ``to_pytorch_loader()`` is called.
    """

    def __init__(
        self,
        meta: JsonObject,
        data_dir: Path | None,
        _native_train: NativeSplit | None = None,
        _native_test: NativeSplit | None = None,
        _native_train_tf: TensorflowDataset | None = None,
        _native_test_tf: TensorflowDataset | None = None,
        _native_train_flax: list["FlaxBatch"] | None = None,
        _native_test_flax: list["FlaxBatch"] | None = None,
    ) -> None:
        self.id = self._required_str(meta, "id")
        self.name = self._required_str(meta, "name")
        self.format = self._required_str(meta, "format")
        self.dataset_type = self._required_str(meta, "dataset_type")
        self.num_samples = self._required_int(meta, "num_samples")
        self.num_classes = self._required_int(meta, "num_classes")
        self.feature_schema = self._optional_json_object(meta.get("feature_schema"))
        self.class_names = self._optional_str_list(meta.get("class_names"))
        resolved_data_dir = data_dir if data_dir is not None else Path()
        self._data_dir: Path = resolved_data_dir
        self.data_dir: Path = resolved_data_dir
        self._data: pl.DataFrame | dict[str, list[object]] | list[object] | None = None
        self._native_train: NativeSplit | None = _native_train
        self._native_test: NativeSplit | None = _native_test
        # Framework-native dataset objects populated by system_loader when
        # tensorflow_datasets or jax-native loaders are available (16.72-bb/16.82-bb).
        self._native_train_tf = _native_train_tf
        self._native_test_tf = _native_test_tf
        self._native_train_flax = _native_train_flax
        self._native_test_flax = _native_test_flax
        self._raw_meta: JsonObject = meta
        self.raw_meta: JsonObject = meta

    @staticmethod
    def _required_str(meta: JsonObject, key: str) -> str:
        value = meta[key]
        if isinstance(value, str):
            return value
        raise TypeError(f"Dataset metadata field {key!r} must be a string")

    @staticmethod
    def _required_int(meta: JsonObject, key: str) -> int:
        value = meta[key]
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise TypeError(f"Dataset metadata field {key!r} must be an integer")

    @staticmethod
    def _optional_json_object(value: JsonValue) -> JsonObject | None:
        if value is None:
            return None
        return ensure_json_object(value)

    @staticmethod
    def _optional_str_list(value: JsonValue) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return [str(item) for item in value]
        raise TypeError("Dataset metadata field 'class_names' must be a list of strings")

    @property
    def info(self) -> JsonObject:
        """Return a summary dictionary with the 8 required keys."""
        class_names: JsonValue = list(self.class_names) if self.class_names is not None else None
        return {
            "id": self.id,
            "name": self.name,
            "format": self.format,
            "type": self.dataset_type,
            "samples": self.num_samples,
            "classes": self.num_classes,
            "class_names": class_names,
            "schema": self.feature_schema,
        }

    @property
    def native_train(self) -> NativeSplit | None:
        return self._native_train

    @native_train.setter
    def native_train(self, value: NativeSplit | None) -> None:
        self._native_train = value

    @property
    def native_test(self) -> NativeSplit | None:
        return self._native_test

    @native_test.setter
    def native_test(self, value: NativeSplit | None) -> None:
        self._native_test = value

    @property
    def raw_data(self) -> pl.DataFrame | dict[str, list[object]] | list[object] | None:
        return self._data

    @raw_data.setter
    def raw_data(self, value: pl.DataFrame | dict[str, list[object]] | list[object] | None) -> None:
        self._data = value

    @property
    @override
    def native_train_flax(self) -> list["FlaxBatch"] | None:
        return self._native_train_flax

    @native_train_flax.setter
    def native_train_flax(self, value: list["FlaxBatch"] | None) -> None:
        self._native_train_flax = value

    @property
    @override
    def native_test_flax(self) -> list["FlaxBatch"] | None:
        return self._native_test_flax

    @native_test_flax.setter
    def native_test_flax(self, value: list["FlaxBatch"] | None) -> None:
        self._native_test_flax = value

    @property
    @override
    def native_train_tf(self) -> TensorflowDataset | None:
        return self._native_train_tf

    @native_train_tf.setter
    def native_train_tf(self, value: TensorflowDataset | None) -> None:
        self._native_train_tf = value

    @property
    @override
    def native_test_tf(self) -> TensorflowDataset | None:
        return self._native_test_tf

    @native_test_tf.setter
    def native_test_tf(self, value: TensorflowDataset | None) -> None:
        self._native_test_tf = value

    @staticmethod
    @override
    @staticmethod
    def _pad_sequences(
        sequences: Sequence[Sequence[int]],
        maxlen: int = 200,
        num_words: int = 20000,
    ) -> npt.NDArray[np.int32]:
        """Pad/truncate variable-length integer sequences (e.g. IMDB)."""
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
    ) -> Iterator[object]:
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
    ) -> tuple[npt.NDArray[np.object_], npt.NDArray[np.object_] | None]:
        """Return features and labels as NumPy arrays for generated pipelines."""
        features: list[object] = []
        labels: list[object | None] = []
        for item in self.iter_samples(
            split=split,
            decoded=decoded,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        ):
            if isinstance(item, tuple):
                item_tuple = cast(tuple[object, ...], item)
                if len(item_tuple) == 2:
                    feature, label = item_tuple
                else:
                    feature, label = item_tuple, None
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
    ) -> Iterator[tuple[list[object], int]]:
        """Yield numeric feature rows and encoded labels from a tabular file."""
        df = self.to_polars()
        label_col = self.detect_label_column(df)
        labels = self._encode_label_values(df[label_col])
        feature_cols = [col for col in df.columns if col != label_col]
        numeric_cols = numeric_columns(df, feature_cols)
        features = df.select(numeric_cols).to_numpy()

        indices = self._split_indices(
            df.height,
            split=split,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        for index in indices:
            yield features[index].tolist(), labels[index]

    def detect_label_column(self, df: pl.DataFrame) -> str:
        if self.feature_schema and "columns" in self.feature_schema:
            columns = self.feature_schema["columns"]
            if isinstance(columns, list):
                for col_info in columns:
                    if not isinstance(col_info, dict):
                        continue
                    column = ensure_json_object(col_info)
                    if column.get("type") == "categorical":
                        name = column.get("name")
                        if isinstance(name, str):
                            return name
        return df.columns[-1]

    def _encode_label_values(self, series: LabelSeries) -> list[int]:
        if self.class_names:
            mapping = {name: idx for idx, name in enumerate(self.class_names)}
            return [mapping[str(value)] for value in series.to_list()]
        return [int(value) for value in factorize(series).tolist()]

    def encode_label_values(self, series: LabelSeries) -> list[int]:
        return self._encode_label_values(series)

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

    @staticmethod
    def split_indices(
        n: int,
        split: str,
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> list[int]:
        return DagnamDataset._split_indices(n, split, val_ratio, test_ratio, seed)

    @override
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

    def find_data_file(self) -> Path:
        return self._find_data_file()
