"""Coverage for the to_tensorflow mixin on DagnamDataset."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from tests.data.dataset._native_helpers import (
    make_indexable_native_ds,
    make_native_numpy_ds,
    make_native_obj_ds,
)

from dagnam._types import TensorflowDataset
from dagnam.data.dataset import DagnamDataset


def _tf_pair_identity(features: object, labels: object) -> tuple[object, object]:
    return features, labels


# ---------------------------------------------------------------- to_tensorflow_dataset


def test_to_tf_native_numpy_test_split(tmp_path: Path) -> None:
    ds = make_native_numpy_ds(tmp_path)
    tf_ds = ds.to_tensorflow_dataset(split="test", batch_size=2, shuffle=False)
    next(iter(tf_ds))


def test_to_tf_native_numpy_train_val(tmp_path: Path) -> None:
    ds = make_native_numpy_ds(tmp_path)
    train = ds.to_tensorflow_dataset(split="train", batch_size=2, shuffle=True, val_ratio=0.2)
    val = ds.to_tensorflow_dataset(split="val", batch_size=2, shuffle=False, val_ratio=0.2)
    next(iter(train))
    next(iter(val))


def test_to_tf_native_obj_array(tmp_path: Path) -> None:
    ds = make_native_obj_ds()
    tf_ds = ds.to_tensorflow_dataset(split="test", batch_size=1, shuffle=False)
    next(iter(tf_ds))


def test_to_tf_native_indexable_splits(tmp_path: Path) -> None:
    ds = make_indexable_native_ds()
    for split in ("train", "val", "test"):
        tf_ds = ds.to_tensorflow_dataset(split=split, batch_size=2, shuffle=False, val_ratio=0.25)
        next(iter(tf_ds))


def test_to_tf_native_indexable_with_map_and_batch_map(tmp_path: Path) -> None:
    ds = make_indexable_native_ds()
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
    ds = make_native_numpy_ds(tmp_path)
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
    ds.native_train_tf = cast("TensorflowDataset", tf.data.Dataset.from_tensor_slices((xs, ys)))
    ds.native_test_tf = cast(
        "TensorflowDataset", tf.data.Dataset.from_tensor_slices((xs[:2], ys[:2]))
    )

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
    ds.native_train_tf = cast("TensorflowDataset", tf.data.Dataset.from_tensor_slices((xs, ys)))
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
