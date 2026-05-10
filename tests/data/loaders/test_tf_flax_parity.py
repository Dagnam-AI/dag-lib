"""Tests for native TF/FLAX system dataset dispatch.

Covers the P1 regressions reported in the audit:
  * `_native_tensorflow_dataset` must return a distinct slice for 'val'
    instead of the full training set.
  * `_native_flax_dataset` must rebatch + reshuffle + honor `val_ratio`
    rather than returning the prebatched native list verbatim.
  * `resolve_system_dataset_flax` must cope with text-like samples
    (bytes / strings) and not divide them by 255.
"""

from __future__ import annotations

import numpy as np
import pytest

_META = {
    "id": "native-probe",
    "name": "ProbeMNIST",
    "format": "image_folder",
    "dataset_type": "image",
    "num_samples": 8,
    "num_classes": 2,
    "source_type": "system",
}


@pytest.fixture
def ds_factory(tmp_path):
    """Return a factory that builds a DagnamDataset with user-supplied native
    attributes so we can exercise the TF/FLAX native paths without the
    tfds / torchvision round-trip.
    """

    def _make(**kwargs):
        from dagnam.data.dataset import DagnamDataset

        return DagnamDataset(_META, tmp_path, **kwargs)

    return _make


# ---------------------------------------------------------------------------
# _native_tensorflow_dataset val split
# ---------------------------------------------------------------------------


def test_native_tf_dataset_val_split_is_distinct(ds_factory):
    tf = pytest.importorskip("tensorflow")

    # 100 dummy samples, 10 val.
    n = 100
    xs = np.arange(n, dtype=np.float32).reshape(n, 1)
    ys = np.arange(n, dtype=np.int64)
    native_train = tf.data.Dataset.from_tensor_slices((xs, ys))

    ds = ds_factory(_native_train_tf=native_train)
    train = ds._native_tensorflow_dataset(
        split="train",
        batch_size=5,
        shuffle=False,
        val_ratio=0.1,
        seed=0,
    )
    val = ds._native_tensorflow_dataset(
        split="val",
        batch_size=5,
        shuffle=False,
        val_ratio=0.1,
        seed=0,
    )

    train_samples = [int(v.numpy()) for batch in train for v in batch[1]]
    val_samples = [int(v.numpy()) for batch in val for v in batch[1]]

    # Distinct partitions, no overlap, correct sizes.
    assert len(val_samples) == 10
    assert len(train_samples) == 90
    assert set(train_samples).isdisjoint(set(val_samples))


def test_native_tf_dataset_train_honors_shuffle_seed(ds_factory):
    tf = pytest.importorskip("tensorflow")

    xs = np.arange(40, dtype=np.float32).reshape(40, 1)
    ys = np.arange(40, dtype=np.int64)
    native_train = tf.data.Dataset.from_tensor_slices((xs, ys))

    ds = ds_factory(_native_train_tf=native_train)
    a = [
        int(v.numpy())
        for batch in ds._native_tensorflow_dataset(
            split="train", batch_size=10, shuffle=True, val_ratio=0.0, seed=123
        )
        for v in batch[1]
    ]
    b = [
        int(v.numpy())
        for batch in ds._native_tensorflow_dataset(
            split="train", batch_size=10, shuffle=True, val_ratio=0.0, seed=123
        )
        for v in batch[1]
    ]
    assert a == b  # deterministic under same seed


# ---------------------------------------------------------------------------
# _native_flax_dataset reshaping + val split
# ---------------------------------------------------------------------------


def test_native_flax_dataset_reshapes_batches_and_splits_val(ds_factory):
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from dagnam.data.loaders.flax import FlaxBatch

    # Prebatched as 2 batches of 50 samples; caller wants batch_size=10 + val.
    xs = np.arange(100, dtype=np.float32).reshape(100, 1)
    ys = np.arange(100, dtype=np.int64)
    native = [
        FlaxBatch(features=jnp.asarray(xs[:50]), labels=jnp.asarray(ys[:50])),
        FlaxBatch(features=jnp.asarray(xs[50:]), labels=jnp.asarray(ys[50:])),
    ]

    ds = ds_factory(_native_train_flax=native)
    train = ds._native_flax_dataset(
        split="train",
        batch_size=10,
        shuffle=False,
        val_ratio=0.1,
        seed=0,
    )
    val = ds._native_flax_dataset(
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


# ---------------------------------------------------------------------------
# resolve_system_dataset_flax handles text samples
# ---------------------------------------------------------------------------


def test_resolve_system_dataset_flax_encodes_text_without_image_scaling(monkeypatch, tmp_path):
    """tfds IMDB yields bytes/text samples; the loader must not divide by 255."""
    pytest.importorskip("jax")

    # Stub tensorflow_datasets with a text dataset so we don't need the real one.
    import sys
    import types

    fake_tfds = types.ModuleType("tensorflow_datasets")

    def fake_load(name, split, as_supervised=True, data_dir=None):
        # Fake "dataset" — the as_numpy adapter reads from a sentinel generator.
        return ("__fake_ds__", split)

    def fake_as_numpy(ds):
        # Emit three synthetic bytes samples with class labels.
        samples = [
            (b"hello world", 1),
            (b"another review here", 0),
            (b"third", 1),
        ]
        return iter(samples)

    fake_tfds.load = fake_load
    fake_tfds.as_numpy = fake_as_numpy
    monkeypatch.setitem(sys.modules, "tensorflow_datasets", fake_tfds)

    from dagnam.data.loaders.system import resolve_system_dataset_flax

    meta = dict(_META)
    meta["name"] = "imdb"
    meta["dataset_type"] = "text"
    monkeypatch.setattr(
        "dagnam.data.loaders.system._SYSTEM_CACHE_ROOT",
        tmp_path,
    )

    result = resolve_system_dataset_flax(meta)
    assert result._native_train_flax is not None
    batch = result._native_train_flax[0]

    # Features must be an integer array (byte codepoints padded), NOT
    # float-divided-by-255. Max value should exceed 1.0 for ASCII.
    feats = np.asarray(batch.features)
    assert feats.dtype.kind in ("i", "u"), f"expected integer dtype, got {feats.dtype}"
    assert feats.max() > 1, "text loader must not apply image normalization"
