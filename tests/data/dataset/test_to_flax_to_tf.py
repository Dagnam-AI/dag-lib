"""Coverage for to_flax + to_tensorflow mixins on DagnamDataset."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("tensorflow")

from dagnam.data.dataset import DagnamDataset
from dagnam.data.loaders.flax import FlaxBatch


def _make_native_numpy_ds(tmp_path) -> DagnamDataset:
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
    ds._native_train = (x_train, y_train)
    ds._native_test = (x_test, y_test)
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
    ds._native_train = (x_train, y_train)
    ds._native_test = (x_test, y_test)
    return ds


class _IndexableNativeDs:
    """Mimics torchvision-style native: len() + __getitem__ returning (img, label)."""

    def __init__(self, n: int = 8, with_numpy: bool = True):
        self.n = n
        self.with_numpy = with_numpy

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        arr = np.full((3, 4, 4), float(i), dtype=np.float32)
        if self.with_numpy:

            class _T:
                def __init__(self, a):
                    self._a = a

                def numpy(self):
                    return self._a

            return _T(arr), i % 2
        return arr, i % 2


def _make_indexable_native_ds(with_numpy=True) -> DagnamDataset:
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
    ds._native_train = _IndexableNativeDs(n=12, with_numpy=with_numpy)
    ds._native_test = _IndexableNativeDs(n=4, with_numpy=with_numpy)
    return ds


# ---------------------------------------------------------------- to_flax_dataset


def test_to_flax_native_numpy_test_split(tmp_path):
    ds = _make_native_numpy_ds(tmp_path)
    batches = ds.to_flax_dataset(split="test", batch_size=2, shuffle=False)
    assert batches
    assert isinstance(batches[0], FlaxBatch)


def test_to_flax_native_numpy_val_train_splits(tmp_path):
    ds = _make_native_numpy_ds(tmp_path)
    val = ds.to_flax_dataset(split="val", batch_size=2, shuffle=False, val_ratio=0.2)
    train = ds.to_flax_dataset(split="train", batch_size=2, shuffle=True, val_ratio=0.2)
    assert val
    assert train


def test_to_flax_native_numpy_object_pad(tmp_path):
    ds = _make_native_obj_ds()
    batches = ds.to_flax_dataset(split="test", batch_size=1, shuffle=False)
    assert batches


def test_to_flax_native_numpy_with_transforms(tmp_path):
    ds = _make_native_numpy_ds(tmp_path)
    batches = ds.to_flax_dataset(
        split="train",
        batch_size=2,
        shuffle=False,
        val_ratio=0.2,
        transform_fn=lambda s: s * 1.0,
        batch_transform_fn=lambda x, y: (x, y),
    )
    assert batches


def test_to_flax_native_indexable_train(tmp_path):
    ds = _make_indexable_native_ds()
    batches = ds.to_flax_dataset(split="train", batch_size=2, shuffle=False, val_ratio=0.25)
    assert batches


def test_to_flax_native_indexable_val(tmp_path):
    ds = _make_indexable_native_ds()
    batches = ds.to_flax_dataset(split="val", batch_size=2, shuffle=False, val_ratio=0.25)
    assert batches


def test_to_flax_native_indexable_test(tmp_path):
    ds = _make_indexable_native_ds()
    batches = ds.to_flax_dataset(split="test", batch_size=2, shuffle=False)
    assert batches


def test_to_flax_native_indexable_no_numpy(tmp_path):
    ds = _make_indexable_native_ds(with_numpy=False)
    batches = ds.to_flax_dataset(split="train", batch_size=2, shuffle=False, val_ratio=0.25)
    assert batches


def test_to_flax_invalid_split(tmp_path):
    ds = _make_native_numpy_ds(tmp_path)
    with pytest.raises(ValueError, match="Unknown split"):
        ds.to_flax_dataset(split="bogus")


def test_to_flax_unsupported_format(tmp_path):
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


def test_to_flax_tabular_csv(tmp_path):
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


def test_to_flax_image_folder_dispatches(tmp_path):
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


def test_to_tf_native_numpy_test_split(tmp_path):
    ds = _make_native_numpy_ds(tmp_path)
    tf_ds = ds.to_tensorflow_dataset(split="test", batch_size=2, shuffle=False)
    next(iter(tf_ds))


def test_to_tf_native_numpy_train_val(tmp_path):
    ds = _make_native_numpy_ds(tmp_path)
    train = ds.to_tensorflow_dataset(split="train", batch_size=2, shuffle=True, val_ratio=0.2)
    val = ds.to_tensorflow_dataset(split="val", batch_size=2, shuffle=False, val_ratio=0.2)
    next(iter(train))
    next(iter(val))


def test_to_tf_native_obj_array(tmp_path):
    ds = _make_native_obj_ds()
    tf_ds = ds.to_tensorflow_dataset(split="test", batch_size=1, shuffle=False)
    next(iter(tf_ds))


def test_to_tf_native_indexable_splits(tmp_path):
    ds = _make_indexable_native_ds()
    for split in ("train", "val", "test"):
        tf_ds = ds.to_tensorflow_dataset(split=split, batch_size=2, shuffle=False, val_ratio=0.25)
        next(iter(tf_ds))


def test_to_tf_native_indexable_with_map_and_batch_map(tmp_path):
    import tensorflow as tf

    ds = _make_indexable_native_ds()
    tf_ds = ds.to_tensorflow_dataset(
        split="train",
        batch_size=2,
        shuffle=True,
        val_ratio=0.25,
        map_fn=lambda x, y: (tf.cast(x, tf.float32), y),
        batch_map_fn=lambda x, y: (x, y),
    )
    next(iter(tf_ds))


def test_to_tf_invalid_split(tmp_path):
    ds = _make_native_numpy_ds(tmp_path)
    with pytest.raises(ValueError, match="Unknown split"):
        ds.to_tensorflow_dataset(split="bogus")


def test_to_tf_unsupported_format(tmp_path):
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


def test_to_tf_tabular_csv(tmp_path):
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


def test_to_tf_image_folder_dispatches(tmp_path):
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


def test_to_flax_native_flax_path(tmp_path):
    """Set _native_train_flax directly so _native_flax_dataset is hit."""
    import jax.numpy as jnp

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
        features=jnp.asarray(np.zeros((4, 3, 4, 4), dtype=np.float32)),
        labels=jnp.asarray(np.zeros(4, dtype=np.int64)),
    )
    batch_b = FlaxBatch(
        features=jnp.asarray(np.ones((4, 3, 4, 4), dtype=np.float32)),
        labels=jnp.asarray(np.ones(4, dtype=np.int64)),
    )
    ds._native_train_flax = [batch_a, batch_b]
    ds._native_test_flax = [batch_a]

    for split in ("train", "val", "test"):
        out = ds.to_flax_dataset(
            split=split, batch_size=2, shuffle=split == "train", val_ratio=0.25
        )
        assert out


def test_to_flax_native_flax_with_transforms(tmp_path):
    import jax.numpy as jnp

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
    ds._native_train_flax = [
        FlaxBatch(
            features=jnp.asarray(np.zeros((4, 3, 4, 4), dtype=np.float32)),
            labels=jnp.asarray(np.zeros(4, dtype=np.int64)),
        )
    ]
    out = ds.to_flax_dataset(
        split="train",
        batch_size=2,
        shuffle=False,
        val_ratio=0.25,
        transform_fn=lambda s: s,
        batch_transform_fn=lambda x, y: (x, y),
    )
    assert out


def test_to_flax_native_flax_empty_returns_empty(tmp_path):
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
    ds._native_train_flax = []  # empty list — triggers early return
    out = ds.to_flax_dataset(split="train", batch_size=2, shuffle=False)
    assert out == []


def test_to_flax_native_flax_val_without_train_raises(tmp_path):
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

    ds._native_train_flax = None
    # Make sure to_flax doesn't hit the early native_flax path: instead poke _native_flax_dataset
    ds._native_test_flax = [
        FlaxBatch(
            features=jnp.zeros((1, 4), dtype=jnp.float32), labels=jnp.zeros(1, dtype=jnp.int32)
        )
    ]
    with pytest.raises(ValueError, match="No native FLAX"):
        ds._native_flax_dataset(split="val", batch_size=2, shuffle=False)


# ---------------------------------------------------------------- native_tensorflow_dataset path


def test_to_tf_native_tf_path(tmp_path):
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
    ds._native_train_tf = tf.data.Dataset.from_tensor_slices((xs, ys))
    ds._native_test_tf = tf.data.Dataset.from_tensor_slices((xs[:2], ys[:2]))

    for split in ("train", "val", "test"):
        out = ds.to_tensorflow_dataset(
            split=split,
            batch_size=2,
            shuffle=split == "train",
            val_ratio=0.25,
        )
        next(iter(out))


def test_to_tf_native_tf_with_map_fns(tmp_path):
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
    ds._native_train_tf = tf.data.Dataset.from_tensor_slices((xs, ys))
    out = ds.to_tensorflow_dataset(
        split="train",
        batch_size=2,
        shuffle=False,
        val_ratio=0.25,
        map_fn=lambda x, y: (x, y),
        batch_map_fn=lambda x, y: (x, y),
    )
    next(iter(out))


def test_native_tf_val_without_train_raises():
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
    ds._native_train_tf = None
    with pytest.raises(ValueError, match="No native TF dataset"):
        ds._native_tensorflow_dataset(split="val", batch_size=2, shuffle=False)
    with pytest.raises(ValueError, match="No native TF dataset"):
        ds._native_tensorflow_dataset(split="train", batch_size=2, shuffle=False)
