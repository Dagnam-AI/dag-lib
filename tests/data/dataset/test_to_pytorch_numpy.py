"""Coverage for to_pytorch _native_numpy_loader + hooks tail."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt
import pytest
from tests.data.dataset._native_helpers import make_indexable_native_ds

from dagnam._types import JsonObject, NativeSplit
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


@pytest.fixture
def int_label_numpy_native_ds() -> DagnamDataset:
    """IMDB-style numpy tuples whose labels are INTEGER class indices (0/1)."""
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
    y_train = (np.arange(10) % 2).astype(np.int64)
    x_test = np.arange(8).reshape(2, 4).astype(np.float32)
    y_test = np.array([0, 1]).astype(np.int64)
    ds.native_train = _native_split(x_train, y_train)
    ds.native_test = _native_split(x_test, y_test)
    return ds


@pytest.mark.parametrize("split", ["test", "train", "val"])
def test_numpy_loader_integer_labels_are_1d_long(
    int_label_numpy_native_ds: DagnamDataset, split: str
) -> None:
    """Integer class labels must become a [B] long target (CrossEntropyLoss), not [B,1] float.

    Regression for G091: `nn.CrossEntropyLoss` rejects a [B,1] float target with
    "0D or 1D target tensor expected, multi-target not supported", which failed
    every pytorch IMDB training job.
    """
    loader = int_label_numpy_native_ds.to_pytorch_loader(
        split=split, batch_size=2, num_workers=0, val_ratio=0.2, shuffle=False
    )
    _x, y = next(iter(loader))
    torch = _torch()
    assert y.dtype == torch.long, f"{split}: integer labels must be long class indices"
    assert y.ndim == 1, f"{split}: target must be 1-D [B], got shape {tuple(y.shape)}"


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


def test_numpy_loader_does_not_pad_rectangular(numpy_native_ds: DagnamDataset) -> None:
    """Rectangular numeric arrays must NOT be padded — width is preserved, not forced to 200."""
    loader = numpy_native_ds.to_pytorch_loader(split="test", batch_size=2, num_workers=0)
    x, _ = next(iter(loader))
    # x_test was (2, 4); a correct guard skips _pad_sequences (which would force width 200).
    assert x.shape[1] == 4


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
    # Ragged sequences are padded/truncated to the fixed maxlen (200).
    assert x.shape[1] == 200


def test_numpy_loader_clamps_object_sequences_to_vocab_size() -> None:
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
    x_train = np.array([np.array([1, 2, 3])], dtype=object)
    y_train = np.array([0.0])
    x_test = np.array([np.array([6, 7, 8, 9])], dtype=object)
    y_test = np.array([1.0])
    ds.native_train = _native_split(x_train, y_train)
    ds.native_test = _native_split(x_test, y_test)

    loader = ds.to_pytorch_loader(split="test", batch_size=1, num_workers=0, vocab_size=8)
    x, _ = next(iter(loader))

    assert x[0, :4].tolist() == [6, 7, 0, 0]


def test_numpy_loader_train_pads_object_dtype_sequences() -> None:
    """The train-split guard also pads ragged object-dtype sequences to maxlen (200)."""
    ds = DagnamDataset(
        {
            "id": "imdb",
            "name": "imdb-like",
            "format": "native",
            "dataset_type": "text",
            "num_samples": 4,
            "num_classes": 2,
            "class_names": [],
        },
        data_dir=None,
    )
    x_train = np.array([np.array([1, 2, 3]), np.array([4, 5]), np.array([6])], dtype=object)
    y_train = np.array([0.0, 1.0, 0.0])
    ds.native_train = _native_split(x_train, y_train)
    loader = ds.to_pytorch_loader(
        split="train", batch_size=1, num_workers=0, val_ratio=0.0, shuffle=False
    )
    x, _ = next(iter(loader))
    torch = _torch()
    assert x.dtype == torch.long
    assert x.shape[1] == 200


# ---------------------------------------------------------------- map-style native loader


def test_native_pytorch_loader_test_split() -> None:
    """Torchvision-style indexable native dataset → test split uses native_test."""
    ds = make_indexable_native_ds(with_numpy=False)  # plain ndarray samples (collatable)
    loader = ds.to_pytorch_loader(split="test", batch_size=2, num_workers=0)
    x, y = next(iter(loader))
    torch = _torch()
    assert torch.is_tensor(x)
    assert torch.is_tensor(y)


def test_native_pytorch_loader_val_split() -> None:
    """Map-style val split goes through random_split."""
    ds = make_indexable_native_ds(with_numpy=False)
    loader = ds.to_pytorch_loader(split="val", batch_size=2, num_workers=0, val_ratio=0.25)
    batches = list(loader)
    assert len(batches) >= 1


def test_native_pytorch_loader_train_split_with_transforms() -> None:
    """Map-style train split via random_split plus _TransformDataset wrap."""
    ds = make_indexable_native_ds(with_numpy=False)
    loader = ds.to_pytorch_loader(
        split="train",
        batch_size=2,
        num_workers=0,
        val_ratio=0.25,
        shuffle=False,
        transform=_double,
    )
    batches = list(loader)
    assert len(batches) >= 1


def test_native_pytorch_loader_missing_native_raises() -> None:
    """Direct call with native_train None raises (defensive guard, line 195)."""
    ds = make_indexable_native_ds()
    ds.native_train = None
    with pytest.raises(ValueError, match="No native PyTorch dataset"):
        ds._native_pytorch_loader(
            split="train",
            batch_size=2,
            num_workers=0,
            shuffle=False,
            val_ratio=0.1,
            seed=0,
        )


def test_native_numpy_loader_non_tuple_train_raises() -> None:
    """Direct call into the numpy loader without tuple arrays raises (line 260)."""
    ds = make_indexable_native_ds()  # native_train is indexable, not a tuple
    with pytest.raises(ValueError, match="Native numpy loader requires train arrays"):
        ds._native_numpy_loader(
            split="train",
            batch_size=2,
            num_workers=0,
            shuffle=False,
            val_ratio=0.1,
            seed=0,
        )


def test_numpy_loader_test_without_tuple_test_uses_empty() -> None:
    """native_train tuple but native_test not a tuple → x_test/y_test = (), () (line 265)."""
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
    x_train = np.arange(40).reshape(10, 4).astype(np.float32)
    y_train = np.arange(10).astype(np.float32)
    ds.native_train = _native_split(x_train, y_train)
    ds.native_test = make_indexable_native_ds().native_train  # non-tuple indexable
    loader = ds.to_pytorch_loader(split="test", batch_size=1, num_workers=0)
    assert list(loader) == []


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


def test_channels_first_image_dataset_transposes_hwc_to_chw() -> None:
    from dagnam.data.dataset.hooks import _ChannelsFirstImageDataset

    base = [(np.zeros((4, 5, 3), np.float32), 1), (np.ones((4, 5, 3), np.float32), 0)]
    ds = _ChannelsFirstImageDataset(cast("Sequence[object]", base))

    assert len(ds) == 2
    item = cast("tuple[object, object]", ds[0])
    data = cast("npt.NDArray[np.float32]", item[0])
    assert data.shape == (3, 4, 5)
    assert item[1] == 1


def test_channels_first_image_dataset_passes_short_and_non_tuple_through() -> None:
    from dagnam.data.dataset.hooks import _ChannelsFirstImageDataset

    one_tuple = _ChannelsFirstImageDataset(cast("Sequence[object]", [(np.zeros((2, 2, 3)),)]))
    short_item = one_tuple[0]
    assert isinstance(short_item, tuple)
    assert len(cast("tuple[object, ...]", short_item)) == 1

    non_tuple = _ChannelsFirstImageDataset(cast("Sequence[object]", ["raw"]))
    assert non_tuple[0] == "raw"


def test_native_pytorch_loader_transposes_bound_image_to_channels_first() -> None:
    from dagnam.data.loaders.system.bound_dataset import BoundNativeDataset
    from dagnam.data.loaders.system.column_store import Column, ColumnStore

    store = ColumnStore(
        {
            "image": Column.eager(np.zeros((6, 8, 8, 3), np.uint8)),
            "label": Column.eager(np.arange(6, dtype=np.int64)),
        }
    )
    binding = {
        "input_column": "image",
        "target_column": "label",
        "input_transform": {"kind": "image_resize", "params": {"size": [8, 8]}},
        "target_transform": {"kind": "class_index", "params": {}},
    }
    bound = BoundNativeDataset(store, binding, [{"name": "image"}])
    meta: JsonObject = {
        "id": "img",
        "name": "img",
        "format": "array",
        "dataset_type": "image",
        "num_samples": 6,
        "num_classes": 3,
    }
    ds = DagnamDataset(meta, None, _native_train=bound, _native_test=bound)

    loader = ds.to_pytorch_loader(split="train", batch_size=2, num_workers=0, binding=binding)
    torch = _torch()
    batch = next(iter(loader))
    x = cast("tuple[object, object]", batch)[0]
    assert tuple(cast("Any", x).shape) == (2, 3, 8, 8)  # channels-first
    del torch
