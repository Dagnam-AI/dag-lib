"""Shared native-dataset builders for the to_flax / to_tensorflow tests.

The ``to_flax`` and ``to_tensorflow`` mixin tests build the same kinds of
``native_*`` datasets (numpy splits, numpy-object splits, and torchvision-style
indexable splits), so those builders live here and are imported by both modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from dagnam._types import NativeSplit
from dagnam.data.dataset import DagnamDataset

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt


def _native_split(features: object, labels: object) -> NativeSplit:
    return cast("NativeSplit", (features, labels))


def array_identity(value: npt.ArrayLike) -> npt.ArrayLike:
    return value


def array_scale(value: npt.ArrayLike) -> npt.ArrayLike:
    return np.asarray(value) * 1.0


def make_native_numpy_ds(tmp_path: Path) -> DagnamDataset:
    ds = DagnamDataset(
        {
            "id": "im1",
            "name": "imdb-like",
            "format": "native",
            "dataset_type": "text",
            "num_samples": 12,
            "num_classes": 2,
            "class_names": [],
        },
        data_dir=None,
    )
    x_train = np.arange(40).reshape(10, 4).astype(np.float32)
    y_train = np.arange(10).astype(np.int64)
    x_test = np.arange(8).reshape(2, 4).astype(np.float32)
    y_test = np.arange(2).astype(np.int64)
    ds.native_train = _native_split(x_train, y_train)
    ds.native_test = _native_split(x_test, y_test)
    return ds


def make_native_obj_ds() -> DagnamDataset:
    """Numpy-object dataset like Keras IMDB (variable-length sequences)."""
    ds = DagnamDataset(
        {
            "id": "obj1",
            "name": "imdb-like",
            "format": "native",
            "dataset_type": "text",
            "num_samples": 4,
            "num_classes": 2,
            "class_names": [],
        },
        data_dir=None,
    )
    x_train = np.array([np.array([1, 2, 3]), np.array([4, 5])], dtype=object)
    y_train = np.array([0, 1], dtype=np.int64)
    x_test = np.array([np.array([6, 7, 8, 9])], dtype=object)
    y_test = np.array([1], dtype=np.int64)
    ds.native_train = _native_split(x_train, y_train)
    ds.native_test = _native_split(x_test, y_test)
    return ds


class _IndexableNativeDs:
    """Mimics torchvision-style native: len() + __getitem__ returning (img, label)."""

    def __init__(self, n: int = 8, with_numpy: bool = True) -> None:
        self.n = n
        self.with_numpy = with_numpy

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> tuple[object, int]:
        arr = np.full((3, 4, 4), float(index), dtype=np.float32)
        if self.with_numpy:

            class _T:
                def __init__(self, a: npt.NDArray[np.float32]) -> None:
                    self._a = a

                def numpy(self) -> npt.NDArray[np.float32]:
                    return self._a

            return _T(arr), index % 2
        return arr, index % 2


def make_indexable_native_ds(with_numpy: bool = True) -> DagnamDataset:
    ds = DagnamDataset(
        {
            "id": "ix1",
            "name": "mnist-like",
            "format": "native",
            "dataset_type": "image",
            "num_samples": 16,
            "num_classes": 2,
            "class_names": [],
        },
        data_dir=None,
    )
    ds.native_train = _IndexableNativeDs(n=12, with_numpy=with_numpy)
    ds.native_test = _IndexableNativeDs(n=4, with_numpy=with_numpy)
    return ds
