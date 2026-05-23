"""Shared typing contract for dataset conversion mixins."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt
import polars as pl

from dagnam._types import JsonObject, NativeSplit, TensorflowDataset

if TYPE_CHECKING:
    from dagnam.data.loaders.flax import FlaxBatch


class DatasetMixinBase(Protocol):
    """Attribute contract supplied by ``DagnamDataset`` to its mixins."""

    id: str
    name: str
    format: str
    dataset_type: str
    num_samples: int
    num_classes: int
    feature_schema: JsonObject | None
    class_names: list[str] | None
    _data_dir: Path
    data_dir: Path
    _data: pl.DataFrame | dict[str, list[object]] | list[object] | None
    _native_train: NativeSplit | None
    _native_test: NativeSplit | None
    _native_train_tf: TensorflowDataset | None
    _native_test_tf: TensorflowDataset | None
    _native_train_flax: list[FlaxBatch] | None
    _native_test_flax: list[FlaxBatch] | None
    _raw_meta: JsonObject
    raw_meta: JsonObject

    def to_polars(self) -> pl.DataFrame:
        raise NotImplementedError

    @property
    def native_train_flax(self) -> list[FlaxBatch] | None:
        raise NotImplementedError

    @property
    def native_test_flax(self) -> list[FlaxBatch] | None:
        raise NotImplementedError

    @property
    def native_train_tf(self) -> TensorflowDataset | None:
        raise NotImplementedError

    @property
    def native_test_tf(self) -> TensorflowDataset | None:
        raise NotImplementedError

    def _find_data_file(self) -> Path:
        raise NotImplementedError

    @staticmethod
    def _pad_sequences(
        sequences: Sequence[Sequence[int]],
        maxlen: int = 200,
        num_words: int = 20000,
    ) -> npt.NDArray[np.int32]:
        raise NotImplementedError
