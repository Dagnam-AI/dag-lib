"""Coverage for to_pytorch _native_numpy_loader + hooks tail."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dagnam.data.dataset import DagnamDataset
from dagnam.data.dataset.hooks import _TransformDataset


@pytest.fixture
def numpy_native_ds():
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
    ds._native_train = (x_train, y_train)
    ds._native_test = (x_test, y_test)
    return ds


def test_numpy_loader_test_split(numpy_native_ds):
    loader = numpy_native_ds.to_pytorch_loader(split="test", batch_size=2, num_workers=0)
    batches = list(loader)
    assert len(batches) >= 1
    x, y = batches[0]
    assert x.dtype == torch.long
    assert y.dtype == torch.float32


def test_numpy_loader_val_split(numpy_native_ds):
    loader = numpy_native_ds.to_pytorch_loader(
        split="val", batch_size=2, num_workers=0, val_ratio=0.2
    )
    batches = list(loader)
    assert len(batches) >= 1


def test_numpy_loader_train_split(numpy_native_ds):
    loader = numpy_native_ds.to_pytorch_loader(
        split="train", batch_size=2, num_workers=0, val_ratio=0.2, shuffle=False
    )
    batches = list(loader)
    assert len(batches) >= 1


def test_numpy_loader_train_with_transform(numpy_native_ds):
    """Exercise the _TransformDataset wrap branch in _native_numpy_loader."""
    loader = numpy_native_ds.to_pytorch_loader(
        split="train",
        batch_size=2,
        num_workers=0,
        val_ratio=0.2,
        shuffle=False,
        transform=lambda x: x * 2,
        target_transform=lambda y: y + 1,
    )
    x, y = next(iter(loader))
    assert torch.is_tensor(x)
    assert torch.is_tensor(y)


def test_numpy_loader_pads_object_dtype_sequences():
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
    ds._native_train = (x_train, y_train)
    ds._native_test = (x_test, y_test)
    loader = ds.to_pytorch_loader(split="test", batch_size=1, num_workers=0)
    x, _ = next(iter(loader))
    assert x.dtype == torch.long


# ---------------------------------------------------------------- hooks tail


def test_transform_dataset_passthrough_non_tuple_item():
    """When the wrapped dataset yields a single object, only `transform` runs."""

    class _Plain:
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return idx * 10

    wrapped = _TransformDataset(_Plain(), transform=lambda v: v + 1)
    assert wrapped[0] == 1
    assert wrapped[1] == 11
    assert len(wrapped) == 2


def test_transform_dataset_preserves_extra_tuple_elements():
    """When __getitem__ returns >2 elements, the extras pass through unchanged."""

    class _Triple:
        def __len__(self):
            return 1

        def __getitem__(self, _idx):
            return 1, "label", "meta"

    wrapped = _TransformDataset(_Triple(), transform=lambda v: v * 10)
    assert wrapped[0] == (10, "label", "meta")


def test_transform_dataset_no_transforms_returns_item():
    class _Single:
        def __len__(self):
            return 1

        def __getitem__(self, _idx):
            return "raw"

    wrapped = _TransformDataset(_Single())
    assert wrapped[0] == "raw"
