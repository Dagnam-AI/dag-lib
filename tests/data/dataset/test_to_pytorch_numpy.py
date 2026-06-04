"""Coverage for to_pytorch _native_numpy_loader + hooks tail."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Protocol, cast

import numpy as np
import pytest

from dagnam._types import NativeSplit
from dagnam.data.dataset import DagnamDataset
from dagnam.data.dataset.hooks import _TransformDataset


class TorchModule(Protocol):
    long: object
    float32: object

    def is_tensor(self, obj: object) -> bool: ...


class ArithmeticValue(Protocol):
    def __mul__(self, value: int) -> object: ...

    def __add__(self, value: int) -> object: ...


def _torch() -> TorchModule:
    return cast("TorchModule", import_module("torch"))


def _native_split(features: object, labels: object) -> NativeSplit:
    return cast("NativeSplit", (features, labels))


def _double(value: object) -> object:
    return cast("ArithmeticValue", value) * 2


def _plus_one(value: object) -> object:
    return cast("ArithmeticValue", value) + 1


def _times_ten(value: object) -> object:
    return cast("ArithmeticValue", value) * 10


@pytest.fixture
def numpy_native_ds() -> DagnamDataset:
    """DagnamDataset wrapping IMDB-style numpy tuples in _native_train/_native_test."""
    ds = DagnamDataset(
        {
            "id": "imdb",
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
    y_train = np.arange(10).astype(np.float32)
    x_test = np.arange(8).reshape(2, 4).astype(np.float32)
    y_test = np.arange(2).astype(np.float32)
    ds.native_train = _native_split(x_train, y_train)
    ds.native_test = _native_split(x_test, y_test)
    return ds


def test_numpy_loader_test_split(numpy_native_ds: DagnamDataset) -> None:
    loader = numpy_native_ds.to_pytorch_loader(split="test", batch_size=2, num_workers=0)
    batches = list(loader)
    assert len(batches) >= 1
    x, y = batches[0]
    torch = _torch()
    assert x.dtype == torch.long
    assert y.dtype == torch.float32


def test_numpy_loader_val_split(numpy_native_ds: DagnamDataset) -> None:
    loader = numpy_native_ds.to_pytorch_loader(
        split="val", batch_size=2, num_workers=0, val_ratio=0.2
    )
    batches = list(loader)
    assert len(batches) >= 1


def test_numpy_loader_train_split(numpy_native_ds: DagnamDataset) -> None:
    loader = numpy_native_ds.to_pytorch_loader(
        split="train", batch_size=2, num_workers=0, val_ratio=0.2, shuffle=False
    )
    batches = list(loader)
    assert len(batches) >= 1


def test_numpy_loader_train_with_transform(numpy_native_ds: DagnamDataset) -> None:
    """Exercise the _TransformDataset wrap branch in _native_numpy_loader."""
    loader = numpy_native_ds.to_pytorch_loader(
        split="train",
        batch_size=2,
        num_workers=0,
        val_ratio=0.2,
        shuffle=False,
        transform=_double,
        target_transform=_plus_one,
    )
    x, y = next(iter(loader))
    torch = _torch()
    assert torch.is_tensor(x)
    assert torch.is_tensor(y)


def test_numpy_loader_pads_object_dtype_sequences() -> None:
    """When numpy arrays are object-dtype, _pad_sequences is called."""
    ds = DagnamDataset(
        {
            "id": "imdb",
            "name": "imdb-like",
            "format": "native",
            "dataset_type": "text",
            "num_samples": 6,
            "num_classes": 2,
            "class_names": [],
        },
        data_dir=None,
    )
    # Variable-length sequences (object dtype like Keras IMDB).
    x_train = np.array([np.array([1, 2, 3]), np.array([4, 5])], dtype=object)
    y_train = np.array([0.0, 1.0])
    x_test = np.array([np.array([6, 7, 8, 9])], dtype=object)
    y_test = np.array([1.0])
    ds.native_train = _native_split(x_train, y_train)
    ds.native_test = _native_split(x_test, y_test)
    loader = ds.to_pytorch_loader(split="test", batch_size=1, num_workers=0)
    x, _ = next(iter(loader))
    torch = _torch()
    assert x.dtype == torch.long


# ---------------------------------------------------------------- hooks tail


def test_transform_dataset_passthrough_non_tuple_item() -> None:
    """When the wrapped dataset yields a single object, only `transform` runs."""

    class _Plain:
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> object:
            return index * 10

    wrapped = _TransformDataset(cast("Sequence[object]", _Plain()), transform=_plus_one)
    assert wrapped[0] == 1
    assert wrapped[1] == 11
    assert len(wrapped) == 2


def test_transform_dataset_preserves_extra_tuple_elements() -> None:
    """When __getitem__ returns >2 elements, the extras pass through unchanged."""

    class _Triple:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, _idx: int) -> object:
            return 1, "label", "meta"

    wrapped = _TransformDataset(cast("Sequence[object]", _Triple()), transform=_times_ten)
    assert wrapped[0] == (10, "label", "meta")


def test_transform_dataset_no_transforms_returns_item() -> None:
    class _Single:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, _idx: int) -> object:
            return "raw"

    wrapped = _TransformDataset(cast("Sequence[object]", _Single()))
    assert wrapped[0] == "raw"
