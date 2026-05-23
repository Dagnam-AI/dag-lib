"""Coverage for to_flax + to_tensorflow mixins on DagnamDataset."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast


import numpy as np
import numpy.typing as npt
import pytest

from dagnam._types import NativeSplit, TensorflowDataset

pytest.importorskip("jax")
pytest.importorskip("tensorflow")

from dagnam.data.dataset import DagnamDataset
from dagnam.data.loaders.flax import FlaxBatch

if TYPE_CHECKING:
    import jax


class JaxNumpyModule(Protocol):
    float32: object
    int32: object

    def asarray(self, value: npt.ArrayLike) -> jax.Array: ...

    def zeros(self, shape: Sequence[int], dtype: object | None = None) -> jax.Array: ...


def _native_split(features: object, labels: object) -> NativeSplit:
    return cast(NativeSplit, (features, labels))


def _array_identity(value: npt.ArrayLike) -> npt.ArrayLike:
    return value


def _array_scale(value: npt.ArrayLike) -> npt.ArrayLike:
    return np.asarray(value) * 1.0


def _jax_batch_identity(features: jax.Array, labels: jax.Array) -> tuple[jax.Array, jax.Array]:
    return features, labels


def _tf_pair_identity(features: object, labels: object) -> tuple[object, object]:
    return features, labels


def _make_native_numpy_ds(tmp_path: Path) -> DagnamDataset:
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


def _make_native_obj_ds() -> DagnamDataset:
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


def _make_indexable_native_ds(with_numpy: bool = True) -> DagnamDataset:
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


# ---------------------------------------------------------------- to_flax_dataset


def test_to_flax_native_numpy_test_split(tmp_path: Path) -> None:
    ds = _make_native_numpy_ds(tmp_path)
    batches = ds.to_flax_dataset(split="test", batch_size=2, shuffle=False)
    assert batches
    assert isinstance(batches[0], FlaxBatch)


def test_to_flax_native_numpy_val_train_splits(tmp_path: Path) -> None:
    ds = _make_native_numpy_ds(tmp_path)
    val = ds.to_flax_dataset(split="val", batch_size=2, shuffle=False, val_ratio=0.2)
    train = ds.to_flax_dataset(split="train", batch_size=2, shuffle=True, val_ratio=0.2)
    assert val
    assert train


def test_to_flax_native_numpy_object_pad(tmp_path: Path) -> None:
    ds = _make_native_obj_ds()
    batches = ds.to_flax_dataset(split="test", batch_size=1, shuffle=False)
    assert batches


def test_to_flax_native_numpy_with_transforms(tmp_path: Path) -> None:
    ds = _make_native_numpy_ds(tmp_path)
    batches = ds.to_flax_dataset(
        split="train",
        batch_size=2,
        shuffle=False,
        val_ratio=0.2,
        transform_fn=_array_scale,
        batch_transform_fn=_jax_batch_identity,
    )
    assert batches


def test_to_flax_native_indexable_train(tmp_path: Path) -> None:
    ds = _make_indexable_native_ds()
    batches = ds.to_flax_dataset(split="train", batch_size=2, shuffle=False, val_ratio=0.25)
    assert batches


def test_to_flax_native_indexable_val(tmp_path: Path) -> None:
    ds = _make_indexable_native_ds()
    batches = ds.to_flax_dataset(split="val", batch_size=2, shuffle=False, val_ratio=0.25)
    assert batches


def test_to_flax_native_indexable_test(tmp_path: Path) -> None:
    ds = _make_indexable_native_ds()
    batches = ds.to_flax_dataset(split="test", batch_size=2, shuffle=False)
    assert batches


def test_to_flax_native_indexable_no_numpy(tmp_path: Path) -> None:
    ds = _make_indexable_native_ds(with_numpy=False)
    batches = ds.to_flax_dataset(split="train", batch_size=2, shuffle=False, val_ratio=0.25)
    assert batches


def test_to_flax_invalid_split(tmp_path: Path) -> None:
    ds = _make_native_numpy_ds(tmp_path)
    with pytest.raises(ValueError, match="Unknown split"):
        ds.to_flax_dataset(split="bogus")


def test_to_flax_unsupported_format(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a\n1\n")
    ds = DagnamDataset(
        {
            "id": "u1",
            "name": "unsupported",
            "format": "parquet",  # unsupported by to_flax
            "dataset_type": "tabular",
            "num_samples": 1,
            "num_classes": 0,
            "class_names": [],
            "filename": "data.csv",
        },
        tmp_path,
    )
    with pytest.raises(ValueError, match="Unsupported format"):
        ds.to_flax_dataset(split="train")


def test_to_flax_tabular_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y,species\n1,2,a\n3,4,b\n5,6,a\n7,8,b\n9,10,a\n")
    ds = DagnamDataset(
        {
            "id": "t1",
            "name": "tab",
            "format": "csv",
            "dataset_type": "tabular",
            "num_samples": 5,
            "num_classes": 2,
            "class_names": ["a", "b"],
            "filename": "data.csv",
        },
        tmp_path,
    )
    batches = ds.to_flax_dataset(
        split="train", batch_size=2, shuffle=False, val_ratio=0.2, test_ratio=0.2
    )
    assert batches


def test_to_flax_image_folder_dispatches(tmp_path: Path) -> None:
    """image_folder path routes through create_flax_dataset image variant."""
    from PIL import Image

    for cls_idx, cls in enumerate(("a", "b")):
        d = tmp_path / cls
        d.mkdir()
        for i in range(3):
            Image.new("RGB", (8, 8), color=(255 * cls_idx, 0, 0)).save(d / f"{i}.jpg", "JPEG")
    ds = DagnamDataset(
        {
            "id": "img1",
            "name": "img",
            "format": "image_folder",
            "dataset_type": "image",
            "num_samples": 6,
            "num_classes": 2,
            "class_names": ["a", "b"],
        },
        tmp_path,
    )
    batches = ds.to_flax_dataset(
        split="train", batch_size=2, shuffle=False, val_ratio=0.2, test_ratio=0.2
    )
    assert batches


# ---------------------------------------------------------------- to_tensorflow_dataset


def test_to_tf_native_numpy_test_split(tmp_path: Path) -> None:
    ds = _make_native_numpy_ds(tmp_path)
    tf_ds = ds.to_tensorflow_dataset(split="test", batch_size=2, shuffle=False)
    next(iter(tf_ds))


def test_to_tf_native_numpy_train_val(tmp_path: Path) -> None:
    ds = _make_native_numpy_ds(tmp_path)
    train = ds.to_tensorflow_dataset(split="train", batch_size=2, shuffle=True, val_ratio=0.2)
    val = ds.to_tensorflow_dataset(split="val", batch_size=2, shuffle=False, val_ratio=0.2)
    next(iter(train))
    next(iter(val))


def test_to_tf_native_obj_array(tmp_path: Path) -> None:
    ds = _make_native_obj_ds()
    tf_ds = ds.to_tensorflow_dataset(split="test", batch_size=1, shuffle=False)
    next(iter(tf_ds))


def test_to_tf_native_indexable_splits(tmp_path: Path) -> None:
    ds = _make_indexable_native_ds()
    for split in ("train", "val", "test"):
        tf_ds = ds.to_tensorflow_dataset(split=split, batch_size=2, shuffle=False, val_ratio=0.25)
        next(iter(tf_ds))


def test_to_tf_native_indexable_with_map_and_batch_map(tmp_path: Path) -> None:
    ds = _make_indexable_native_ds()
    tf_ds = ds.to_tensorflow_dataset(
        split="train",
        batch_size=2,
        shuffle=True,
        val_ratio=0.25,
        map_fn=_tf_pair_identity,
        batch_map_fn=_tf_pair_identity,
    )
    next(iter(tf_ds))


def test_to_tf_invalid_split(tmp_path: Path) -> None:
    ds = _make_native_numpy_ds(tmp_path)
    with pytest.raises(ValueError, match="Unknown split"):
        ds.to_tensorflow_dataset(split="bogus")


def test_to_tf_unsupported_format(tmp_path: Path) -> None:
    ds = DagnamDataset(
        {
            "id": "u1",
            "name": "unsupported",
            "format": "parquet",
            "dataset_type": "tabular",
            "num_samples": 1,
            "num_classes": 0,
            "class_names": [],
        },
        tmp_path,
    )
    with pytest.raises(ValueError, match="Unsupported format"):
        ds.to_tensorflow_dataset(split="train")


def test_to_tf_tabular_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y,species\n1,2,a\n3,4,b\n5,6,a\n7,8,b\n9,10,a\n")
    ds = DagnamDataset(
        {
            "id": "t1",
            "name": "tab",
            "format": "csv",
            "dataset_type": "tabular",
            "num_samples": 5,
            "num_classes": 2,
            "class_names": ["a", "b"],
            "filename": "data.csv",
        },
        tmp_path,
    )
    tf_ds = ds.to_tensorflow_dataset(
        split="train", batch_size=2, shuffle=False, val_ratio=0.2, test_ratio=0.2
    )
    next(iter(tf_ds))


def test_to_tf_image_folder_dispatches(tmp_path: Path) -> None:
    from PIL import Image

    for cls_idx, cls in enumerate(("a", "b")):
        d = tmp_path / cls
        d.mkdir()
        for i in range(3):
            Image.new("RGB", (8, 8), color=(255 * cls_idx, 0, 0)).save(d / f"{i}.jpg", "JPEG")
    ds = DagnamDataset(
        {
            "id": "img1",
            "name": "img",
            "format": "image_folder",
            "dataset_type": "image",
            "num_samples": 6,
            "num_classes": 2,
            "class_names": ["a", "b"],
        },
        tmp_path,
    )
    tf_ds = ds.to_tensorflow_dataset(
        split="train", batch_size=2, shuffle=False, val_ratio=0.2, test_ratio=0.2
    )
    next(iter(tf_ds))


# ---------------------------------------------------------------- native_flax_dataset path


def test_to_flax_native_flax_path(tmp_path: Path) -> None:
    """Set _native_train_flax directly so _native_flax_dataset is hit."""
    import jax.numpy as jnp
    jnp_mod = cast(JaxNumpyModule, jnp)

    ds = DagnamDataset(
        {
            "id": "n1",
            "name": "native-flax",
            "format": "native",
            "dataset_type": "image",
            "num_samples": 8,
            "num_classes": 2,
            "class_names": [],
            "source_type": "system",
        },
        data_dir=None,
    )
    batch_a = FlaxBatch(
        features=jnp_mod.asarray(np.zeros((4, 3, 4, 4), dtype=np.float32)),
        labels=jnp_mod.asarray(np.zeros(4, dtype=np.int64)),
    )
    batch_b = FlaxBatch(
        features=jnp_mod.asarray(np.ones((4, 3, 4, 4), dtype=np.float32)),
        labels=jnp_mod.asarray(np.ones(4, dtype=np.int64)),
    )
    ds.native_train_flax = [batch_a, batch_b]
    ds.native_test_flax = [batch_a]

    for split in ("train", "val", "test"):
        out = ds.to_flax_dataset(
            split=split, batch_size=2, shuffle=split == "train", val_ratio=0.25
        )
        assert out


def test_to_flax_native_flax_with_transforms(tmp_path: Path) -> None:
    import jax.numpy as jnp
    jnp_mod = cast(JaxNumpyModule, jnp)

    ds = DagnamDataset(
        {
            "id": "n1",
            "name": "native-flax",
            "format": "native",
            "dataset_type": "image",
            "num_samples": 4,
            "num_classes": 2,
            "class_names": [],
            "source_type": "system",
        },
        data_dir=None,
    )
    ds.native_train_flax = [
        FlaxBatch(
            features=jnp_mod.asarray(np.zeros((4, 3, 4, 4), dtype=np.float32)),
            labels=jnp_mod.asarray(np.zeros(4, dtype=np.int64)),
        )
    ]
    out = ds.to_flax_dataset(
        split="train",
        batch_size=2,
        shuffle=False,
        val_ratio=0.25,
        transform_fn=_array_identity,
        batch_transform_fn=_jax_batch_identity,
    )
    assert out


def test_to_flax_native_flax_empty_returns_empty(tmp_path: Path) -> None:
    ds = DagnamDataset(
        {
            "id": "n1",
            "name": "empty",
            "format": "native",
            "dataset_type": "image",
            "num_samples": 0,
            "num_classes": 0,
            "class_names": [],
            "source_type": "system",
        },
        data_dir=None,
    )
    ds.native_train_flax = []  # empty list — triggers early return
    out = ds.to_flax_dataset(split="train", batch_size=2, shuffle=False)
    assert out == []


def test_to_flax_native_flax_val_without_train_raises(tmp_path: Path) -> None:
    ds = DagnamDataset(
        {
            "id": "n1",
            "name": "x",
            "format": "native",
            "dataset_type": "image",
            "num_samples": 0,
            "num_classes": 0,
            "class_names": [],
            "source_type": "system",
        },
        data_dir=None,
    )
    # Trip the "no native flax" attribute path via patching _native_flax_dataset directly
    # Easier: set _native_train_flax but force val path via train_flax=None branch
    import jax.numpy as jnp
    jnp_mod = cast(JaxNumpyModule, jnp)

    ds.native_train_flax = None
    # Make sure to_flax doesn't hit the early native_flax path: instead poke _native_flax_dataset
    ds.native_test_flax = [
        FlaxBatch(
            features=jnp_mod.zeros((1, 4), dtype=jnp_mod.float32),
            labels=jnp_mod.zeros((1,), dtype=jnp_mod.int32),
        )
    ]
    with pytest.raises(ValueError, match="No native FLAX"):
        ds.native_flax_dataset(split="val", batch_size=2, shuffle=False)


# ---------------------------------------------------------------- native_tensorflow_dataset path


def test_to_tf_native_tf_path(tmp_path: Path) -> None:
    import tensorflow as tf

    ds = DagnamDataset(
        {
            "id": "ntf1",
            "name": "native-tf",
            "format": "native",
            "dataset_type": "image",
            "num_samples": 8,
            "num_classes": 2,
            "class_names": [],
            "source_type": "system",
        },
        data_dir=None,
    )
    xs = np.arange(8 * 3 * 4 * 4, dtype=np.float32).reshape(8, 3, 4, 4)
    ys = np.arange(8, dtype=np.int64) % 2
    ds.native_train_tf = cast(TensorflowDataset, tf.data.Dataset.from_tensor_slices((xs, ys)))
    ds.native_test_tf = cast(TensorflowDataset, tf.data.Dataset.from_tensor_slices((xs[:2], ys[:2])))

    for split in ("train", "val", "test"):
        out = ds.to_tensorflow_dataset(
            split=split,
            batch_size=2,
            shuffle=split == "train",
            val_ratio=0.25,
        )
        next(iter(out))


def test_to_tf_native_tf_with_map_fns(tmp_path: Path) -> None:
    import tensorflow as tf

    ds = DagnamDataset(
        {
            "id": "ntf2",
            "name": "native-tf",
            "format": "native",
            "dataset_type": "image",
            "num_samples": 4,
            "num_classes": 2,
            "class_names": [],
            "source_type": "system",
        },
        data_dir=None,
    )
    xs = np.zeros((4, 4), dtype=np.float32)
    ys = np.zeros(4, dtype=np.int64)
    ds.native_train_tf = cast(TensorflowDataset, tf.data.Dataset.from_tensor_slices((xs, ys)))
    out = ds.to_tensorflow_dataset(
        split="train",
        batch_size=2,
        shuffle=False,
        val_ratio=0.25,
        map_fn=_tf_pair_identity,
        batch_map_fn=_tf_pair_identity,
    )
    next(iter(out))


def test_native_tf_val_without_train_raises() -> None:
    ds = DagnamDataset(
        {
            "id": "x",
            "name": "x",
            "format": "native",
            "dataset_type": "image",
            "num_samples": 0,
            "num_classes": 0,
            "class_names": [],
            "source_type": "system",
        },
        data_dir=None,
    )
    ds.native_train_tf = None
    with pytest.raises(ValueError, match="No native TF dataset"):
        ds.native_tensorflow_dataset(split="val", batch_size=2, shuffle=False)
    with pytest.raises(ValueError, match="No native TF dataset"):
        ds.native_tensorflow_dataset(split="train", batch_size=2, shuffle=False)
