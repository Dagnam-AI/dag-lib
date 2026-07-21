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


def _regression_csv_ds(tmp_path: Path) -> DagnamDataset:
    csv_path = tmp_path / "regression.csv"
    csv_path.write_text(
        "feature,target\n" + "\n".join(f"{float(i)},{float(i) + 0.25}" for i in range(20)) + "\n"
    )
    return DagnamDataset(
        {
            "id": "regression-1",
            "name": "Regression",
            "format": "csv",
            "dataset_type": "tabular",
            "num_samples": 20,
            "num_classes": 0,
            "class_names": None,
            "feature_schema": None,
            "filename": "regression.csv",
        },
        tmp_path,
    )


def _text_csv_ds(tmp_path: Path) -> DagnamDataset:
    csv_path = tmp_path / "text.csv"
    csv_path.write_text(
        "text,label\n"
        + "\n".join(
            f"deterministic short review {index},{'positive' if index % 2 == 0 else 'negative'}"
            for index in range(40)
        )
        + "\n"
    )
    return DagnamDataset(
        {
            "id": "text-1",
            "name": "Text",
            "format": "csv",
            "dataset_type": "text",
            "num_samples": 40,
            "num_classes": 2,
            "class_names": None,
            "feature_schema": {
                "columns": [
                    {"name": "text", "type": "categorical"},
                    {"name": "label", "type": "categorical"},
                ]
            },
            "filename": "text.csv",
        },
        tmp_path,
    )


_TEXT_ROLES = {"text": "text_input", "label": "target"}
_TEXT_BINDING: dict[str, object] = {
    "input_column": "text",
    "target_column": "label",
    "input_transform": {
        "kind": "tokenize",
        "params": {"vocab_size": 256, "sequence_length": 32},
    },
    "target_transform": {"kind": "class_index", "params": {"dtype": "long"}},
}


# ---------------------------------------------------------------- flax tabular


def test_flax_loader_text_binding_tokenizes_user_csv(tmp_path: Path) -> None:
    batches = create_flax_dataset(
        _text_csv_ds(tmp_path),
        split="train",
        batch_size=8,
        shuffle=False,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        column_roles=_TEXT_ROLES,
        binding=_TEXT_BINDING,
    )

    features = np.asarray(batches[0].features)
    labels = np.asarray(batches[0].labels)
    assert features.shape == (8, 32)
    assert features.dtype == np.int32
    assert np.issubdtype(labels.dtype, np.integer)
    assert features.max() < 256


def test_flax_loader_numeric_binding_preserves_float_column_target(tmp_path: Path) -> None:
    batches = create_flax_dataset(
        _regression_csv_ds(tmp_path),
        split="train",
        batch_size=4,
        shuffle=False,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=0,
        column_roles={"feature": "feature", "target": "target"},
        binding={"target_transform": {"kind": "numeric", "params": {"dtype": "float"}}},
    )

    targets = np.asarray(batches[0].labels)
    assert targets.dtype == np.float32
    assert targets.shape == (4, 1)


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


def test_flax_loader_column_roles_accepts_label_role(tmp_path: Path) -> None:
    """G307: column_roles={"col": "label"} must resolve the target column for the
    flax loader. commit ac16b9c switched this loader from detect_label_column
    (which already accepted "label") to split_by_roles, whose TARGET_ROLES silently
    omitted "label" and broke this alias.
    """
    ds = _csv_ds(tmp_path)
    batches = create_flax_dataset(
        ds,
        split="train",
        batch_size=4,
        shuffle=False,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=0,
        column_roles={"x": "feature", "y": "feature", "species": "label"},
    )
    assert batches
    assert batches[0].features.shape[1] == 2


# ---------------------------------------------------------------- tf tabular


def test_tf_loader_text_binding_tokenizes_user_csv(tmp_path: Path) -> None:
    tf_ds = create_tensorflow_dataset(
        _text_csv_ds(tmp_path),
        split="train",
        batch_size=8,
        shuffle=False,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        column_roles=_TEXT_ROLES,
        binding=_TEXT_BINDING,
    )

    features, labels = cast("tuple[object, object]", next(iter(tf_ds)))
    feature_array = np.asarray(features)
    label_array = np.asarray(labels)
    assert feature_array.shape == (8, 32)
    assert feature_array.dtype == np.int32
    assert label_array.dtype == np.int64
    assert feature_array.max() < 256


def test_tf_loader_numeric_binding_preserves_float_column_target(tmp_path: Path) -> None:
    tf_ds = create_tensorflow_dataset(
        _regression_csv_ds(tmp_path),
        split="train",
        batch_size=4,
        shuffle=False,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=0,
        column_roles={"feature": "feature", "target": "target"},
        binding={"target_transform": {"kind": "numeric", "params": {"dtype": "float"}}},
    )

    targets = np.asarray(cast("TensorBatch", next(iter(tf_ds)))[1])
    assert targets.dtype == np.float32
    assert targets.shape == (4, 1)


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


def test_tf_loader_column_roles_accepts_label_role(tmp_path: Path) -> None:
    """G307: same "label" role alias must work for the tf loader (see the flax
    loader's ``test_flax_loader_column_roles_accepts_label_role`` above)."""
    ds = _csv_ds(tmp_path)
    tf_ds = create_tensorflow_dataset(
        ds,
        split="train",
        batch_size=4,
        shuffle=False,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=0,
        column_roles={"x": "feature", "y": "feature", "species": "label"},
    )
    batch = cast("TensorBatch", next(iter(tf_ds)))
    assert batch[0].shape[1] == 2
