"""Coverage for the to_tensorflow mixin on DagnamDataset."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from tests.data.dataset._native_helpers import (
    make_indexable_native_ds,
    make_native_numpy_ds,
    make_native_obj_ds,
)
from tests.typing_helpers import PytestMonkeyPatch

from dagnam._types import JsonObject, NativeSplit, TensorflowDataset
from dagnam.data.dataset import DagnamDataset
from dagnam.data.dataset.to_tensorflow import _cardinality_to_int


class _HasShape(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...


class _TensorBatch(_HasShape, Protocol):
    def numpy(self) -> np.ndarray: ...


def _tf_pair_identity(features: object, labels: object) -> tuple[object, object]:
    return features, labels


def _native_split(features: object, labels: object) -> NativeSplit:
    return cast("NativeSplit", (features, labels))


def _system_native_meta(name: str) -> JsonObject:
    return {
        "id": "sys1",
        "name": name,
        "format": "native",
        "dataset_type": "image",
        "num_samples": 4,
        "num_classes": 2,
        "class_names": [],
        "source_type": "system",
    }


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


def test_to_tf_native_obj_array_pads_to_maxlen(tmp_path: Path) -> None:
    """Ragged object-array sequences are padded/truncated to the fixed maxlen (200)."""
    ds = make_native_obj_ds()
    tf_ds = ds.to_tensorflow_dataset(split="test", batch_size=1, shuffle=False)
    x_batch, _ = cast("tuple[_HasShape, _HasShape]", next(iter(tf_ds)))
    assert x_batch.shape[1] == 200


def test_to_tf_native_obj_array_clamps_to_vocab_size(tmp_path: Path) -> None:
    ds = make_native_obj_ds()
    tf_ds = ds.to_tensorflow_dataset(split="test", batch_size=1, shuffle=False, vocab_size=8)
    x_batch, _ = cast("tuple[_TensorBatch, _HasShape]", next(iter(tf_ds)))

    assert x_batch.numpy()[0, :4].tolist() == [6, 7, 0, 0]


def test_to_tf_native_numpy_not_padded(tmp_path: Path) -> None:
    """Rectangular numeric arrays keep their width — the padding guard must not fire."""
    ds = make_native_numpy_ds(tmp_path)  # x_train is (10, 4)
    tf_ds = ds.to_tensorflow_dataset(split="train", batch_size=2, shuffle=False, val_ratio=0.2)
    x_batch, _ = cast("tuple[_HasShape, _HasShape]", next(iter(tf_ds)))
    assert x_batch.shape[1] == 4


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


# ---------------------------------------------------------------- branch coverage


def test_cardinality_to_int_rejects_non_integer() -> None:
    with pytest.raises(TypeError, match="cardinality"):
        _cardinality_to_int(object())


def test_cardinality_to_int_accepts_int() -> None:
    assert _cardinality_to_int(7) == 7


def test_native_to_tensorflow_raises_without_train() -> None:
    ds = DagnamDataset(_system_native_meta("none"), data_dir=None)
    ds.native_train = None
    with pytest.raises(ValueError, match="No native dataset"):
        ds._native_to_tensorflow(split="train", batch_size=2, shuffle=False, val_ratio=0.1, seed=0)


def test_native_to_tensorflow_tuple_without_test_uses_empty(tmp_path: Path) -> None:
    """Tuple native_train with no tuple native_test → x_test, y_test = (), ()."""
    ds = DagnamDataset(_system_native_meta("imdb"), data_dir=None)
    x_train = np.arange(20).reshape(5, 4).astype(np.float32)
    y_train = np.arange(5).astype(np.int64)
    ds.native_train = _native_split(x_train, y_train)
    ds.native_test = None
    out = ds._native_to_tensorflow(split="test", batch_size=1, shuffle=False, val_ratio=0.1, seed=0)
    # test split with empty x/y produces an empty dataset.
    assert list(out) == []


def test_native_to_tensorflow_sample_not_tuple_raises() -> None:
    class _BadDs:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, _i: int) -> object:
            return np.zeros((2, 2))  # not a tuple

    ds = DagnamDataset(_system_native_meta("bad"), data_dir=None)
    ds.native_train = cast("NativeSplit", _BadDs())
    ds.native_test = None
    with pytest.raises(TypeError, match="feature, label"):
        ds._native_to_tensorflow(split="train", batch_size=1, shuffle=False, val_ratio=0.0, seed=0)


def test_native_to_tensorflow_short_tuple_sample_raises() -> None:
    class _ShortDs:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, _i: int) -> object:
            return (np.zeros((2, 2)),)  # length 1

    ds = DagnamDataset(_system_native_meta("short"), data_dir=None)
    ds.native_train = cast("NativeSplit", _ShortDs())
    ds.native_test = None
    with pytest.raises(TypeError, match="feature, label"):
        ds._native_to_tensorflow(split="train", batch_size=1, shuffle=False, val_ratio=0.0, seed=0)


def test_native_to_tensorflow_non_int_label_raises() -> None:
    class _BadLabelDs:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, _i: int) -> tuple[object, object]:
            return np.zeros((2, 2), dtype=np.float32), "not-an-int"

    ds = DagnamDataset(_system_native_meta("badlabel"), data_dir=None)
    ds.native_train = cast("NativeSplit", _BadLabelDs())
    ds.native_test = None
    with pytest.raises(TypeError, match="integer-compatible"):
        ds._native_to_tensorflow(split="train", batch_size=1, shuffle=False, val_ratio=0.0, seed=0)


def _unknown_cardinality_ds(tmp_path: Path) -> DagnamDataset:
    """Native TF dataset whose cardinality is UNKNOWN (a filtered dataset)."""
    import tensorflow as tf

    ds = DagnamDataset(_system_native_meta("unknown-card"), data_dir=None)
    xs = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    ys = (np.arange(8) % 2).astype(np.int64)
    base = tf.data.Dataset.from_tensor_slices((xs, ys))
    # filter() makes cardinality UNKNOWN, forcing the iterate-to-count fallback.
    # The TF stub types the predicate as single-arg; this element spec is a pair.
    filtered = base.filter(lambda _x, _y: True)  # pyright: ignore[reportArgumentType]
    ds.native_train_tf = cast("TensorflowDataset", filtered)
    return ds


def test_native_tf_unknown_cardinality_val(tmp_path: Path) -> None:
    ds = _unknown_cardinality_ds(tmp_path)
    out = ds.native_tensorflow_dataset(split="val", batch_size=2, shuffle=False, val_ratio=0.25)
    next(iter(out))


def test_native_tf_unknown_cardinality_train(tmp_path: Path) -> None:
    ds = _unknown_cardinality_ds(tmp_path)
    out = ds.native_tensorflow_dataset(split="train", batch_size=2, shuffle=False, val_ratio=0.25)
    next(iter(out))


def test_native_tf_test_split_without_any_dataset_raises() -> None:
    """split='test' with both native_test_tf and native_train_tf None → ds is None."""
    ds = DagnamDataset(_system_native_meta("empty-tf"), data_dir=None)
    ds.native_train_tf = None
    ds.native_test_tf = None
    with pytest.raises(ValueError, match="No native TF dataset for split"):
        ds.native_tensorflow_dataset(split="test", batch_size=2, shuffle=False)


def test_try_upgrade_to_native_tf_already_native_returns_true() -> None:
    import tensorflow as tf

    ds = DagnamDataset(_system_native_meta("already"), data_dir=None)
    xs = np.zeros((2, 4), dtype=np.float32)
    ys = np.zeros(2, dtype=np.int64)
    ds.native_train_tf = cast("TensorflowDataset", tf.data.Dataset.from_tensor_slices((xs, ys)))
    assert ds._try_upgrade_to_native_tf() is True


def test_try_upgrade_to_native_tf_non_system_returns_false() -> None:
    ds = DagnamDataset(_system_native_meta("nonsys"), data_dir=None)
    ds._raw_meta["source_type"] = "user"
    assert ds._try_upgrade_to_native_tf() is False


def test_try_upgrade_to_native_tf_no_tfds_returns_false(
    monkeypatch: PytestMonkeyPatch,
) -> None:
    ds = DagnamDataset(_system_native_meta("notfds"), data_dir=None)
    monkeypatch.setattr("dagnam.data.dataset.to_tensorflow.find_spec", lambda _name: None)
    assert ds._try_upgrade_to_native_tf() is False


def test_to_tensorflow_upgrades_to_native_tf(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """Drive the tfds-upgrade path: find_spec patched, resolvers stubbed."""
    import tensorflow as tf

    xs = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    ys = (np.arange(8) % 2).astype(np.int64)

    upgraded = DagnamDataset(_system_native_meta("mnist"), data_dir=None)
    upgraded.native_train_tf = cast(
        "TensorflowDataset", tf.data.Dataset.from_tensor_slices((xs, ys))
    )
    upgraded.native_test_tf = cast(
        "TensorflowDataset", tf.data.Dataset.from_tensor_slices((xs[:2], ys[:2]))
    )

    monkeypatch.setattr(
        "dagnam.data.dataset.to_tensorflow.find_spec",
        lambda _name: object(),
    )
    monkeypatch.setattr(
        "dagnam.data.loaders.system.resolve_tfds_name",
        lambda _meta: "mnist",
    )
    monkeypatch.setattr(
        "dagnam.data.loaders.system.resolve_system_dataset_tf",
        lambda _meta: upgraded,
    )

    ds = DagnamDataset(_system_native_meta("mnist"), data_dir=None)
    # Legacy PT-native handle present so to_tensorflow_dataset tries the upgrade.
    ds.native_train = _native_split(xs, ys)

    out = ds.to_tensorflow_dataset(split="train", batch_size=2, shuffle=False)
    next(iter(out))


def test_try_upgrade_to_native_tf_unknown_name_returns_false(
    monkeypatch: PytestMonkeyPatch,
) -> None:
    ds = DagnamDataset(_system_native_meta("weird"), data_dir=None)
    monkeypatch.setattr(
        "dagnam.data.dataset.to_tensorflow.find_spec",
        lambda _name: object(),
    )
    monkeypatch.setattr(
        "dagnam.data.loaders.system.resolve_tfds_name",
        lambda _meta: None,
    )
    assert ds._try_upgrade_to_native_tf() is False


def test_try_upgrade_to_native_tf_upgrade_without_train_returns_false(
    monkeypatch: PytestMonkeyPatch,
) -> None:
    empty = DagnamDataset(_system_native_meta("empty"), data_dir=None)
    empty.native_train_tf = None
    monkeypatch.setattr(
        "dagnam.data.dataset.to_tensorflow.find_spec",
        lambda _name: object(),
    )
    monkeypatch.setattr(
        "dagnam.data.loaders.system.resolve_tfds_name",
        lambda _meta: "mnist",
    )
    monkeypatch.setattr(
        "dagnam.data.loaders.system.resolve_system_dataset_tf",
        lambda _meta: empty,
    )
    ds = DagnamDataset(_system_native_meta("empty"), data_dir=None)
    assert ds._try_upgrade_to_native_tf() is False


def test_to_tensorflow_audio_folder_dispatches(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    def fake_create_audio_tf(**kwargs: object) -> str:
        calls.update(kwargs)
        return "audio-tf"

    monkeypatch.setattr(
        "dagnam.data.loaders.audio.create_tensorflow_dataset",
        fake_create_audio_tf,
    )
    ds = DagnamDataset(
        {
            "id": "aud",
            "name": "audio",
            "format": "audio_folder",
            "dataset_type": "audio",
            "num_samples": 2,
            "num_classes": 2,
            "class_names": [],
        },
        tmp_path,
    )
    result = ds.to_tensorflow_dataset(split="train", batch_size=2, shuffle=False)
    assert result == "audio-tf"
    assert calls["split"] == "train"
