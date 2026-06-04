"""Coverage for tabular flax.py + tf.py loaders."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("jax")
pytest.importorskip("tensorflow")

from dagnam._types import JsonObject
from dagnam.data.dataset import DagnamDataset
from dagnam.data.loaders.flax import FlaxBatch, create_flax_dataset
from dagnam.data.loaders.tf import create_tensorflow_dataset


class TensorLike(Protocol):
    @property
    def shape(self) -> Sequence[int]: ...


class TensorBatch(Protocol):
    def __getitem__(self, index: int) -> TensorLike: ...


def _scale_feature(feature: npt.ArrayLike, label: object) -> tuple[npt.ArrayLike, object]:
    return np.asarray(feature) * 1.0, label


def _batch_identity(batch: FlaxBatch) -> FlaxBatch:
    return batch


def _feature_only(feature: npt.ArrayLike, _label: object) -> npt.ArrayLike:
    return feature


def _tf_pair_identity(features: object, labels: object) -> tuple[object, object]:
    return features, labels


def _csv_ds(tmp_path: Path, with_class_names: bool = True) -> DagnamDataset:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "x,y,species\n"
        + "\n".join(f"{i}.0,{i + 0.5},{'a' if i % 2 == 0 else 'b'}" for i in range(20))
        + "\n"
    )
    meta = cast(
        "JsonObject",
        {
            "id": "tab1",
            "name": "tab",
            "format": "csv",
            "dataset_type": "tabular",
            "num_samples": 20,
            "num_classes": 2,
            "class_names": ["a", "b"] if with_class_names else [],
            "feature_schema": {
                "columns": [
                    {"name": "x", "type": "numeric"},
                    {"name": "y", "type": "numeric"},
                    {"name": "species", "type": "categorical"},
                ]
            },
            "filename": "data.csv",
        },
    )
    return DagnamDataset(meta, tmp_path)


# ---------------------------------------------------------------- flax tabular


def test_flax_loader_basic_with_class_names(tmp_path: Path) -> None:
    ds = _csv_ds(tmp_path)
    batches = create_flax_dataset(
        ds, split="train", batch_size=4, shuffle=False, val_ratio=0.2, test_ratio=0.2, seed=0
    )
    assert batches
    assert isinstance(batches[0], FlaxBatch)
    assert batches[0].features.shape[1] == 2  # x, y numeric


def test_flax_loader_factorizes_labels_when_no_class_names(tmp_path: Path) -> None:
    ds = _csv_ds(tmp_path, with_class_names=False)
    batches = create_flax_dataset(
        ds, split="val", batch_size=2, shuffle=False, val_ratio=0.25, test_ratio=0.25, seed=0
    )
    assert batches


def test_flax_loader_shuffle_and_transform(tmp_path: Path) -> None:
    ds = _csv_ds(tmp_path)
    batches = create_flax_dataset(
        ds,
        split="train",
        batch_size=2,
        shuffle=True,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=0,
        transform_fn=_scale_feature,
        batch_transform_fn=_batch_identity,
    )
    assert batches


def test_flax_loader_transform_returns_single_feature(tmp_path: Path) -> None:
    """transform_fn returning a non-tuple should hit the else branch."""
    ds = _csv_ds(tmp_path)
    batches = create_flax_dataset(
        ds,
        split="train",
        batch_size=2,
        shuffle=False,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=0,
        transform_fn=_feature_only,
    )
    assert batches


def test_flax_loader_test_split(tmp_path: Path) -> None:
    ds = _csv_ds(tmp_path)
    batches = create_flax_dataset(
        ds, split="test", batch_size=2, shuffle=False, val_ratio=0.2, test_ratio=0.2, seed=0
    )
    assert batches


# ---------------------------------------------------------------- tf tabular


def test_tf_loader_basic(tmp_path: Path) -> None:
    ds = _csv_ds(tmp_path)
    tf_ds = create_tensorflow_dataset(
        ds, split="train", batch_size=4, shuffle=False, val_ratio=0.2, test_ratio=0.2, seed=0
    )
    batch = cast("TensorBatch", next(iter(tf_ds)))
    assert batch[0].shape[0] >= 1


def test_tf_loader_shuffle_map_batch_map(tmp_path: Path) -> None:
    ds = _csv_ds(tmp_path)
    tf_ds = create_tensorflow_dataset(
        ds,
        split="train",
        batch_size=2,
        shuffle=True,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=0,
        map_fn=_tf_pair_identity,
        batch_map_fn=_tf_pair_identity,
    )
    next(iter(tf_ds))


def test_tf_loader_factorizes_when_no_class_names(tmp_path: Path) -> None:
    ds = _csv_ds(tmp_path, with_class_names=False)
    tf_ds = create_tensorflow_dataset(
        ds, split="val", batch_size=2, shuffle=False, val_ratio=0.25, test_ratio=0.25, seed=0
    )
    next(iter(tf_ds))


def test_tf_loader_test_split(tmp_path: Path) -> None:
    ds = _csv_ds(tmp_path)
    tf_ds = create_tensorflow_dataset(
        ds, split="test", batch_size=2, shuffle=False, val_ratio=0.2, test_ratio=0.2, seed=0
    )
    next(iter(tf_ds))
