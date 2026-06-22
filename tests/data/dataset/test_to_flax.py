"""Coverage for the to_flax mixin on DagnamDataset."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("jax")

from tests.data.dataset._native_helpers import (
    array_identity,
    array_scale,
    make_indexable_native_ds,
    make_native_numpy_ds,
    make_native_obj_ds,
)

from dagnam.data.dataset import DagnamDataset
from dagnam.data.loaders.flax import FlaxBatch

if TYPE_CHECKING:
    import jax


class JaxNumpyModule(Protocol):
    float32: object
    int32: object

    def asarray(self, value: npt.ArrayLike) -> jax.Array: ...

    def zeros(self, shape: Sequence[int], dtype: object | None = None) -> jax.Array: ...


def _jax_batch_identity(features: jax.Array, labels: jax.Array) -> tuple[jax.Array, jax.Array]:
    return features, labels


# ---------------------------------------------------------------- to_flax_dataset


def test_to_flax_native_numpy_test_split(tmp_path: Path) -> None:
    ds = make_native_numpy_ds(tmp_path)
    batches = ds.to_flax_dataset(split="test", batch_size=2, shuffle=False)
    assert batches
    assert isinstance(batches[0], FlaxBatch)


def test_to_flax_native_numpy_val_train_splits(tmp_path: Path) -> None:
    ds = make_native_numpy_ds(tmp_path)
    val = ds.to_flax_dataset(split="val", batch_size=2, shuffle=False, val_ratio=0.2)
    train = ds.to_flax_dataset(split="train", batch_size=2, shuffle=True, val_ratio=0.2)
    assert val
    assert train


def test_to_flax_native_numpy_object_pad(tmp_path: Path) -> None:
    ds = make_native_obj_ds()
    batches = ds.to_flax_dataset(split="test", batch_size=1, shuffle=False)
    assert batches
    # Ragged object-array sequences are padded/truncated to the fixed maxlen (200).
    assert batches[0].features.shape[1] == 200


def test_to_flax_native_numpy_object_honors_sequence_length(tmp_path: Path) -> None:
    # G079: the embedding-derived sequence_length overrides the default maxlen on
    # the tuple-native ragged path too, so all paths pad to the same fixed length.
    del tmp_path
    ds = make_native_obj_ds()
    batches = ds.to_flax_dataset(split="test", batch_size=1, shuffle=False, sequence_length=12)
    assert batches[0].features.shape[1] == 12


def test_to_flax_native_numpy_object_clamps_to_vocab_size(tmp_path: Path) -> None:
    ds = make_native_obj_ds()
    batches = ds.to_flax_dataset(split="test", batch_size=1, shuffle=False, vocab_size=8)

    assert batches[0].features[0, :4].tolist() == [6, 7, 0, 0]


def test_to_flax_native_numpy_not_padded(tmp_path: Path) -> None:
    """Rectangular numeric arrays keep their width — the padding guard must not fire."""
    ds = make_native_numpy_ds(tmp_path)  # x_train is (10, 4)
    batches = ds.to_flax_dataset(split="train", batch_size=2, shuffle=False, val_ratio=0.2)
    assert batches[0].features.shape[1] == 4


def test_to_flax_native_numpy_with_transforms(tmp_path: Path) -> None:
    ds = make_native_numpy_ds(tmp_path)
    batches = ds.to_flax_dataset(
        split="train",
        batch_size=2,
        shuffle=False,
        val_ratio=0.2,
        transform_fn=array_scale,
        batch_transform_fn=_jax_batch_identity,
    )
    assert batches


def test_to_flax_native_indexable_train(tmp_path: Path) -> None:
    ds = make_indexable_native_ds()
    batches = ds.to_flax_dataset(split="train", batch_size=2, shuffle=False, val_ratio=0.25)
    assert batches


def test_to_flax_native_indexable_val(tmp_path: Path) -> None:
    ds = make_indexable_native_ds()
    batches = ds.to_flax_dataset(split="val", batch_size=2, shuffle=False, val_ratio=0.25)
    assert batches


def test_to_flax_native_indexable_test(tmp_path: Path) -> None:
    ds = make_indexable_native_ds()
    batches = ds.to_flax_dataset(split="test", batch_size=2, shuffle=False)
    assert batches


def test_to_flax_native_indexable_segmentation_mask_label(tmp_path: Path) -> None:
    # G109 regression: a 2-D segmentation mask target must materialize as a mask
    # batch, not raise "only 0-dimensional arrays can be converted to Python scalars"
    # from int(lbl). The mask keeps its [H, W] shape and integer dtype.
    ds = make_indexable_native_ds(label_kind="mask")
    batches = ds.to_flax_dataset(split="test", batch_size=2, shuffle=False)
    lbl = np.asarray(batches[0].labels)
    assert lbl.shape == (2, 4, 4)  # [batch, H, W] segmentation masks
    assert lbl.dtype.kind == "i"  # integer mask ids (int64)


def test_to_flax_native_indexable_float_label_keeps_float(tmp_path: Path) -> None:
    # A non-integer (regression) target keeps its float dtype (no int64 coercion).
    ds = make_indexable_native_ds(label_kind="float")
    batches = ds.to_flax_dataset(split="test", batch_size=2, shuffle=False)
    assert np.asarray(batches[0].labels).dtype.kind == "f"


def test_to_flax_native_indexable_no_numpy(tmp_path: Path) -> None:
    ds = make_indexable_native_ds(with_numpy=False)
    batches = ds.to_flax_dataset(split="train", batch_size=2, shuffle=False, val_ratio=0.25)
    assert batches


def test_to_flax_invalid_split(tmp_path: Path) -> None:
    ds = make_native_numpy_ds(tmp_path)
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


# ---------------------------------------------------------------- native_flax_dataset path


def test_to_flax_native_flax_path(tmp_path: Path) -> None:
    """Set _native_train_flax directly so _native_flax_dataset is hit."""
    import jax.numpy as jnp

    jnp_mod = cast("JaxNumpyModule", jnp)

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


def test_to_flax_native_flax_pads_ragged_sequences(tmp_path: Path) -> None:
    # G079: a native-FLAX text dataset yields ragged (variable-length) token rows.
    # They must be padded/truncated to a fixed length before np.concatenate so the
    # batch is a rectangular integer array a jax.numpy array can hold.
    del tmp_path
    ds = DagnamDataset(
        {
            "id": "n1",
            "name": "native-flax-text",
            "format": "native",
            "dataset_type": "text",
            "num_samples": 3,
            "num_classes": 2,
            "class_names": [],
            "source_type": "system",
        },
        data_dir=None,
    )
    ragged = [
        FlaxBatch(
            features=cast("Any", np.array([[1, 2, 3, 4, 5], [1, 2, 3]], dtype=object)),
            labels=cast("Any", np.asarray([0, 1], dtype=np.int64)),
        ),
        FlaxBatch(
            features=cast("Any", np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=object)),
            labels=cast("Any", np.asarray([1], dtype=np.int64)),
        ),
    ]
    ds.native_train_flax = ragged
    ds.native_test_flax = ragged
    out = ds.to_flax_dataset(split="test", batch_size=8, shuffle=False, sequence_length=6)
    feats = np.asarray(out[0].features)
    assert feats.shape[1] == 6  # padded/truncated to the requested length
    assert feats.dtype.kind in ("i", "u")  # integer tokens, never object


def test_to_flax_native_flax_pads_rectangular_batches_of_different_lengths(tmp_path: Path) -> None:
    # G079 (the REAL platform failure): each FlaxBatch is internally rectangular
    # integer (dtype != object), but DIFFERENT batches have different sequence
    # lengths (e.g. 4816 vs 3819). np.concatenate(axis=0) then fails on the
    # mismatched dim-1 — the object-only guard missed this. They must pad to one
    # length first.
    del tmp_path
    ds = DagnamDataset(
        {
            "id": "n1",
            "name": "native-flax-text",
            "format": "native",
            "dataset_type": "text",
            "num_samples": 3,
            "num_classes": 2,
            "class_names": [],
            "source_type": "system",
        },
        data_dir=None,
    )
    batches = [
        FlaxBatch(
            features=cast("Any", np.ones((2, 5), dtype=np.int64)),
            labels=cast("Any", np.asarray([0, 1], dtype=np.int64)),
        ),
        FlaxBatch(
            features=cast("Any", np.ones((1, 8), dtype=np.int64)),  # different length
            labels=cast("Any", np.asarray([1], dtype=np.int64)),
        ),
    ]
    ds.native_train_flax = batches
    ds.native_test_flax = batches
    out = ds.to_flax_dataset(split="test", batch_size=8, shuffle=False, sequence_length=6)
    feats = np.asarray(out[0].features)
    assert feats.shape == (3, 6)  # 3 samples, padded/truncated to length 6
    assert feats.dtype.kind in ("i", "u")


def test_to_flax_native_flax_tokenizes_string_rows(tmp_path: Path) -> None:
    # G078 (flax defensive): if a native-FLAX batch carries raw text strings, they
    # must be hash-tokenized to fixed-length integer ids, not left as strings.
    del tmp_path
    ds = DagnamDataset(
        {
            "id": "n1",
            "name": "native-flax-strings",
            "format": "native",
            "dataset_type": "text",
            "num_samples": 2,
            "num_classes": 2,
            "class_names": [],
            "source_type": "system",
        },
        data_dir=None,
    )
    strings = [
        FlaxBatch(
            features=cast("Any", np.array(["hello world foo", "bar baz"], dtype=object)),
            labels=cast("Any", np.asarray([0, 1], dtype=np.int64)),
        )
    ]
    ds.native_train_flax = strings
    ds.native_test_flax = strings
    out = ds.to_flax_dataset(split="test", batch_size=8, shuffle=False, sequence_length=5)
    feats = np.asarray(out[0].features)
    assert feats.shape[1] == 5
    assert feats.dtype.kind in ("i", "u")


def test_to_flax_native_flax_with_transforms(tmp_path: Path) -> None:
    import jax.numpy as jnp

    jnp_mod = cast("JaxNumpyModule", jnp)

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
        transform_fn=array_identity,
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

    jnp_mod = cast("JaxNumpyModule", jnp)

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
