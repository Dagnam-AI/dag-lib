"""Tests for dataset transform hooks and raw access APIs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Protocol, SupportsInt, cast

import numpy as np
import numpy.typing as npt
from tests.typing_helpers import PytestMonkeyPatch
from typing_extensions import override

from dagnam._types import JsonObject
from dagnam.data.dataset import DagnamDataset

if TYPE_CHECKING:
    import jax


class TensorLike(Protocol):
    @property
    def shape(self) -> Sequence[int]: ...

    def numpy(self) -> npt.NDArray[np.float64]: ...

    def squeeze(self, dim: int) -> TensorLike: ...

    def tolist(self) -> list[float | int]: ...

    def __add__(self, value: int) -> TensorLike: ...


def _identity(value: object) -> object:
    return value


def _add_ten(value: object) -> int:
    return int(cast("SupportsInt", value)) + 10


def _add_hundred(value: object) -> int:
    return int(cast("SupportsInt", value)) + 100


def _tf_map(value: object) -> object:
    return value


def _flax_transform(value: npt.ArrayLike) -> npt.ArrayLike:
    return value


def _flax_batch_transform(features: jax.Array, labels: jax.Array) -> tuple[jax.Array, jax.Array]:
    return features, labels


def _meta(
    dataset_id: str,
    name: str,
    fmt: str,
    dataset_type: str,
    num_samples: int = 2,
    num_classes: int = 2,
) -> JsonObject:
    return {
        "id": dataset_id,
        "name": name,
        "format": fmt,
        "dataset_type": dataset_type,
        "num_samples": num_samples,
        "num_classes": num_classes,
    }


def test_image_folder_loader_accepts_custom_transform(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    def fake_create_loader(**kwargs: object) -> str:
        calls.update(kwargs)
        return "loader"

    monkeypatch.setattr(
        "dagnam.data.loaders.image_folder.create_pytorch_loader",
        fake_create_loader,
    )

    dataset = DagnamDataset(
        _meta("img", "Images", "image_folder", "image"),
        tmp_path,
    )

    transform = _identity
    target_transform = _identity
    loader = dataset.to_pytorch_loader(
        split="train",
        transform=transform,
        target_transform=target_transform,
        batch_size=4,
    )

    assert loader == "loader"
    assert calls["transform"] is transform
    assert calls["target_transform"] is target_transform
    assert calls["split"] == "train"
    assert calls["batch_size"] == 4


def test_pytorch_batch_transform_wraps_custom_collate(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    def fake_create_loader(**kwargs: object) -> str:
        calls.update(kwargs)
        return "loader"

    monkeypatch.setattr(
        "dagnam.data.loaders.image_folder.create_pytorch_loader",
        fake_create_loader,
    )

    dataset = DagnamDataset(
        _meta("img", "Images", "image_folder", "image"),
        tmp_path,
    )

    def collate_fn(batch: object) -> dict[str, object]:
        return {"items": batch}

    def batch_transform(batch: object) -> dict[str, object]:
        batch = cast("dict[str, object]", batch)
        batch["transformed"] = True
        return batch

    dataset.to_pytorch_loader(
        split="train",
        collate_fn=collate_fn,
        batch_transform=batch_transform,
    )

    collate = cast("Callable[[object], object]", calls["collate_fn"])
    collated = collate([("x", 0)])

    assert collated == {"items": [("x", 0)], "transformed": True}


def test_audio_folder_loader_accepts_waveform_and_spectrogram_transforms(
    monkeypatch: PytestMonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    def fake_create_loader(**kwargs: object) -> str:
        calls.update(kwargs)
        return "audio-loader"

    monkeypatch.setattr(
        "dagnam.data.loaders.audio.create_pytorch_loader",
        fake_create_loader,
    )

    dataset = DagnamDataset(
        _meta("aud", "Audio", "audio_folder", "audio"),
        tmp_path,
    )

    waveform_transform = _identity
    spectrogram_transform = _identity
    target_transform = _identity
    loader = dataset.to_pytorch_loader(
        split="train",
        waveform_transform=waveform_transform,
        spectrogram_transform=spectrogram_transform,
        target_transform=target_transform,
    )

    assert loader == "audio-loader"
    assert calls["waveform_transform"] is waveform_transform
    assert calls["spectrogram_transform"] is spectrogram_transform
    assert calls["target_transform"] is target_transform


def test_native_pytorch_loader_applies_transform_to_map_dataset(tmp_path: Path) -> None:
    from torch.utils.data import Dataset

    class NativeDataset(Dataset[tuple[int, int]]):
        def __len__(self) -> int:
            return 2

        @override
        def __getitem__(self, index: int) -> tuple[int, int]:
            return index, index

    dataset = DagnamDataset(
        _meta("native", "Native", "custom", "image"),
        tmp_path,
        _native_train=NativeDataset(),
        _native_test=NativeDataset(),
    )

    loader = dataset.to_pytorch_loader(
        split="train",
        batch_size=2,
        num_workers=0,
        shuffle=False,
        transform=_add_ten,
        target_transform=_add_hundred,
    )
    batch_inputs, batch_targets = next(iter(loader))

    assert batch_inputs.tolist() == [10, 11]
    assert batch_targets.tolist() == [100, 101]


def test_iter_samples_returns_native_split_items(tmp_path: Path) -> None:
    dataset = DagnamDataset(
        _meta("native", "Native", "custom", "tabular"),
        tmp_path,
    )
    dataset.raw_data = {"train": [({"x": 1}, 0), ({"x": 2}, 1)]}

    assert list(dataset.iter_samples(split="train")) == [({"x": 1}, 0), ({"x": 2}, 1)]


def test_to_arrays_uses_iter_samples(tmp_path: Path) -> None:
    dataset = DagnamDataset(
        _meta("native", "Native", "custom", "tabular"),
        tmp_path,
    )
    dataset.raw_data = {"train": [([1.0, 2.0], 0), ([3.0, 4.0], 1)]}

    features, labels = dataset.to_arrays(split="train")

    assert features.shape == (2, 2)
    assert labels is not None
    assert labels.tolist() == [0, 1]


def test_to_arrays_reads_file_backed_csv_dataset(tmp_path: Path) -> None:
    (tmp_path / "data.csv").write_text(
        "feat1,feat2,label\n1.0,2.0,cat\n3.0,4.0,dog\n5.0,6.0,cat\n",
        encoding="utf-8",
    )
    dataset = DagnamDataset(
        {
            **_meta("csv", "CSV", "csv", "tabular", num_samples=3),
            "class_names": ["cat", "dog"],
        },
        tmp_path,
    )

    features, labels = dataset.to_arrays(split="train", val_ratio=0, test_ratio=0)

    # All rows are present (shuffle is deterministic but reorders)
    assert features.shape == (3, 2)
    assert labels is not None
    assert labels.shape == (3,)
    # Verify the label distribution is preserved
    assert sorted(labels.tolist()) == [0, 0, 1]
    # Verify each feature row's label matches the original mapping
    feature_to_label = {(1.0, 2.0): 0, (3.0, 4.0): 1, (5.0, 6.0): 0}
    for feat_row, label in zip(features.tolist(), labels.tolist(), strict=False):
        assert feature_to_label[tuple(feat_row)] == label


def test_to_arrays_and_to_pytorch_loader_produce_same_split(tmp_path: Path) -> None:
    """Determinism parity: to_arrays() and to_pytorch_loader() with the same
    seed must yield identical samples (same rows in same order)."""
    (tmp_path / "data.csv").write_text(
        "feat1,feat2,label\n1.0,2.0,cat\n3.0,4.0,dog\n5.0,6.0,cat\n7.0,8.0,dog\n",
        encoding="utf-8",
    )
    dataset = DagnamDataset(
        {
            **_meta("csv", "CSV", "csv", "tabular", num_samples=4),
            "class_names": ["cat", "dog"],
        },
        tmp_path,
    )

    features_arr, labels_arr = dataset.to_arrays(split="train", val_ratio=0, test_ratio=0, seed=42)
    assert labels_arr is not None

    loader = dataset.to_pytorch_loader(
        split="train",
        batch_size=4,
        num_workers=0,
        shuffle=False,  # Disable DataLoader shuffle so order matches to_arrays
        val_ratio=0,
        test_ratio=0,
        seed=42,
    )
    features_loader, labels_loader = cast("tuple[TensorLike, TensorLike]", next(iter(loader)))

    np.testing.assert_allclose(features_loader.numpy(), features_arr)
    np.testing.assert_array_equal(labels_loader.numpy(), labels_arr)


def test_iter_samples_reads_file_backed_jsonl_dataset(tmp_path: Path) -> None:
    (tmp_path / "data.jsonl").write_text(
        '{"feat": 1.5, "label": "cat"}\n{"feat": 2.5, "label": "dog"}\n',
        encoding="utf-8",
    )
    dataset = DagnamDataset(
        {
            **_meta("jsonl", "JSONL", "jsonl", "tabular", num_samples=2),
            "class_names": ["cat", "dog"],
        },
        tmp_path,
    )

    # Splits shuffle deterministically — verify set equality rather than order
    samples = cast(
        "list[tuple[list[float], int]]",
        list(dataset.iter_samples(split="train", val_ratio=0, test_ratio=0)),
    )
    assert sorted(samples) == [([1.5], 0), ([2.5], 1)]


def test_tabular_pytorch_loader_applies_batch_transform(tmp_path: Path) -> None:
    (tmp_path / "data.csv").write_text(
        "feat,label\n1.0,cat\n2.0,dog\n",
        encoding="utf-8",
    )
    dataset = DagnamDataset(
        {
            **_meta("csv", "CSV", "csv", "tabular", num_samples=2),
            "class_names": ["cat", "dog"],
        },
        tmp_path,
    )

    def batch_transform(batch: object) -> tuple[TensorLike, TensorLike]:
        features, labels = cast("tuple[TensorLike, TensorLike]", batch)
        return features + 10, labels + 100

    loader = dataset.to_pytorch_loader(
        split="train",
        batch_size=2,
        num_workers=0,
        shuffle=False,
        val_ratio=0,
        test_ratio=0,
        batch_transform=batch_transform,
    )
    features, labels = cast("tuple[TensorLike, TensorLike]", next(iter(loader)))

    assert sorted(features.squeeze(1).tolist()) == [11.0, 12.0]
    assert sorted(labels.tolist()) == [100, 101]


def test_tabular_pytorch_loader_uses_custom_collate_fn(tmp_path: Path) -> None:
    (tmp_path / "data.json").write_text(
        '[{"feat": 1.0, "label": "cat"}, {"feat": 2.0, "label": "dog"}]',
        encoding="utf-8",
    )
    dataset = DagnamDataset(
        {
            **_meta("json", "JSON", "json", "tabular", num_samples=2),
            "class_names": ["cat", "dog"],
        },
        tmp_path,
    )

    def collate_fn(batch: object) -> dict[str, object]:
        rows = cast("Sequence[tuple[object, SupportsInt]]", batch)
        return {"count": len(rows), "labels": [int(label) for _, label in rows]}

    loader = dataset.to_pytorch_loader(
        split="train",
        batch_size=2,
        num_workers=0,
        shuffle=False,
        val_ratio=0,
        test_ratio=0,
        collate_fn=collate_fn,
    )

    batch = cast("dict[str, object]", next(iter(loader)))
    assert batch["count"] == 2
    assert sorted(cast("list[int]", batch["labels"])) == [0, 1]


def test_tensorflow_dataset_accepts_sample_and_batch_map_fns(
    monkeypatch: PytestMonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    def fake_create_dataset(**kwargs: object) -> str:
        calls.update(kwargs)
        return "tf-dataset"

    monkeypatch.setitem(sys.modules, "tensorflow", object())
    monkeypatch.setattr(
        "dagnam.data.loaders.tf.create_tensorflow_dataset",
        fake_create_dataset,
    )

    dataset = DagnamDataset(
        _meta("tab", "Tabular", "csv", "tabular"),
        tmp_path,
    )
    map_fn = _tf_map
    batch_map_fn = _tf_map

    result = dataset.to_tensorflow_dataset(map_fn=map_fn, batch_map_fn=batch_map_fn)

    assert result == "tf-dataset"
    assert calls["map_fn"] is map_fn
    assert calls["batch_map_fn"] is batch_map_fn


def test_flax_dataset_accepts_sample_and_batch_transform_fns(
    monkeypatch: PytestMonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    def fake_create_dataset(**kwargs: object) -> str:
        calls.update(kwargs)
        return "flax-dataset"

    monkeypatch.setitem(sys.modules, "jax", object())
    monkeypatch.setattr(
        "dagnam.data.loaders.flax.create_flax_dataset",
        fake_create_dataset,
    )

    dataset = DagnamDataset(
        _meta("tab", "Tabular", "csv", "tabular"),
        tmp_path,
    )
    transform_fn = _flax_transform
    batch_transform_fn = _flax_batch_transform

    result = dataset.to_flax_dataset(
        transform_fn=transform_fn,
        batch_transform_fn=batch_transform_fn,
    )

    assert result == "flax-dataset"
    assert calls["transform_fn"] is transform_fn
    assert calls["batch_transform_fn"] is batch_transform_fn
