"""Tests for native TF/FLAX system dataset dispatch.

Covers the P1 regressions reported in the audit:
  * `_native_tensorflow_dataset` must return a distinct slice for 'val'
    instead of the full training set.
  * `_native_flax_dataset` must rebatch + reshuffle + honor `val_ratio`
    rather than returning the prebatched native list verbatim.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, SupportsInt, cast

import numpy as np
import numpy.typing as npt
import pytest

from dagnam._types import JsonObject, TensorflowDataset
from dagnam.data.dataset import DagnamDataset

if TYPE_CHECKING:
    import jax

    from dagnam.data.loaders.flax import FlaxBatch

_META: JsonObject = {
    "id": "native-probe",
    "name": "ProbeMNIST",
    "format": "image_folder",
    "dataset_type": "image",
    "num_samples": 8,
    "num_classes": 2,
    "source_type": "system",
}


class TensorValue(Protocol):
    def numpy(self) -> SupportsInt: ...


class TensorBatch(Protocol):
    def __getitem__(self, index: int) -> Sequence[TensorValue]: ...


class JaxNumpyModule(Protocol):
    def asarray(self, value: npt.ArrayLike) -> jax.Array: ...


def _dataset_with_tf(tmp_path: Path, native_train_tf: TensorflowDataset) -> DagnamDataset:
    return DagnamDataset(_META, tmp_path, _native_train_tf=native_train_tf)


def _dataset_with_flax(tmp_path: Path, native_train_flax: list[FlaxBatch]) -> DagnamDataset:
    return DagnamDataset(_META, tmp_path, _native_train_flax=native_train_flax)


def _tf_label_values(dataset: Iterable[object]) -> list[int]:
    batches = cast("Iterable[TensorBatch]", dataset)
    return [int(value.numpy()) for batch in batches for value in batch[1]]


# ---------------------------------------------------------------------------
# _native_tensorflow_dataset val split
# ---------------------------------------------------------------------------


def test_native_tf_dataset_val_split_is_distinct(tmp_path: Path) -> None:
    tf = pytest.importorskip("tensorflow")

    # 100 dummy samples, 10 val.
    n = 100
    xs = np.arange(n, dtype=np.float32).reshape(n, 1)
    ys = np.arange(n, dtype=np.int64)
    native_train = tf.data.Dataset.from_tensor_slices((xs, ys))

    ds = _dataset_with_tf(tmp_path, cast("TensorflowDataset", native_train))
    train = ds.native_tensorflow_dataset(
        split="train",
        batch_size=5,
        shuffle=False,
        val_ratio=0.1,
        seed=0,
    )
    val = ds.native_tensorflow_dataset(
        split="val",
        batch_size=5,
        shuffle=False,
        val_ratio=0.1,
        seed=0,
    )

    train_samples = _tf_label_values(cast("Iterable[object]", train))
    val_samples = _tf_label_values(cast("Iterable[object]", val))

    # Distinct partitions, no overlap, correct sizes.
    assert len(val_samples) == 10
    assert len(train_samples) == 90
    assert set(train_samples).isdisjoint(set(val_samples))


def test_native_tf_dataset_train_honors_shuffle_seed(tmp_path: Path) -> None:
    tf = pytest.importorskip("tensorflow")

    xs = np.arange(40, dtype=np.float32).reshape(40, 1)
    ys = np.arange(40, dtype=np.int64)
    native_train = tf.data.Dataset.from_tensor_slices((xs, ys))

    ds = _dataset_with_tf(tmp_path, cast("TensorflowDataset", native_train))
    a = _tf_label_values(
        cast(
            "Iterable[object]",
            ds.native_tensorflow_dataset(
                split="train", batch_size=10, shuffle=True, val_ratio=0.0, seed=123
            ),
        )
    )
    b = _tf_label_values(
        cast(
            "Iterable[object]",
            ds.native_tensorflow_dataset(
                split="train", batch_size=10, shuffle=True, val_ratio=0.0, seed=123
            ),
        )
    )
    assert a == b  # deterministic under same seed


# ---------------------------------------------------------------------------
# _native_flax_dataset reshaping + val split
# ---------------------------------------------------------------------------


def test_native_flax_dataset_reshapes_batches_and_splits_val(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from dagnam.data.loaders.flax import FlaxBatch

    # Prebatched as 2 batches of 50 samples; caller wants batch_size=10 + val.
    xs = np.arange(100, dtype=np.float32).reshape(100, 1)
    ys = np.arange(100, dtype=np.int64)
    jnp_mod = cast("JaxNumpyModule", jnp)
    native = [
        FlaxBatch(features=jnp_mod.asarray(xs[:50]), labels=jnp_mod.asarray(ys[:50])),
        FlaxBatch(features=jnp_mod.asarray(xs[50:]), labels=jnp_mod.asarray(ys[50:])),
    ]

    ds = _dataset_with_flax(tmp_path, native)
    train = ds.native_flax_dataset(
        split="train",
        batch_size=10,
        shuffle=False,
        val_ratio=0.1,
        seed=0,
    )
    val = ds.native_flax_dataset(
        split="val",
        batch_size=10,
        shuffle=False,
        val_ratio=0.1,
        seed=0,
    )

    # 90 train / 10 val, batched at batch_size=10
    train_samples = [int(l) for b in train for l in b.labels.tolist()]
    val_samples = [int(l) for b in val for l in b.labels.tolist()]
    assert len(train_samples) == 90
    assert len(val_samples) == 10
    assert set(train_samples).isdisjoint(set(val_samples))
    # First train batch respects batch_size
    assert train[0].labels.shape[0] == 10
