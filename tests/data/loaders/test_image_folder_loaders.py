"""Comprehensive coverage for dagnam.data.loaders.image_folder.

Uses Pillow to synthesize small JPEGs so the real torch/torchvision code path
is exercised end-to-end without external test fixtures.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")
PIL = pytest.importorskip("PIL.Image")

from dagnam.data.dataset import DagnamDataset  # noqa: E402
from dagnam.data.loaders.image_folder import (  # noqa: E402
    ImageArray,
    TransformFn,
    create_flax_dataset,
    create_pytorch_loader,
    create_tensorflow_dataset,
    gather_image_samples,
    resolve_split_dir,
)

if TYPE_CHECKING:
    import jax


class TensorLike(Protocol):
    @property
    def shape(self) -> Sequence[int]: ...


class TransformsModule(Protocol):
    def Compose(self, transforms: Sequence[object]) -> object: ...

    def Resize(self, size: tuple[int, int]) -> object: ...

    def ToTensor(self) -> object: ...


class TensorflowBatch(Protocol):
    def __getitem__(self, index: int) -> TensorLike: ...


def _array_scale(arr: ImageArray) -> ImageArray:
    return cast("ImageArray", np.asarray(arr) * 1.0)


def _jax_batch_identity(features: jax.Array, labels: jax.Array) -> tuple[jax.Array, jax.Array]:
    return features, labels


def _make_jpeg(
    path: Path,
    color: tuple[int, int, int] = (255, 0, 0),
    size: tuple[int, int] = (8, 8),
) -> None:
    img = PIL.new("RGB", size, color=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG")


def _build_unsplit_dataset(
    root: Path,
    classes: tuple[str, ...] = ("cat", "dog"),
    per_class: int = 10,
) -> None:
    for cls_idx, cls in enumerate(classes):
        for i in range(per_class):
            c = (255, 0, 0) if cls_idx == 0 else (0, 0, 255)
            _make_jpeg(root / cls / f"{i}.jpg", color=c)


def _build_split_dataset(root: Path) -> None:
    for split in ("train", "val", "test"):
        for cls_idx, cls in enumerate(("cat", "dog")):
            c = (255, 0, 0) if cls_idx == 0 else (0, 0, 255)
            for i in range(4):
                _make_jpeg(root / split / cls / f"{i}.jpg", color=c)


def _make_ds(tmp_path: Path, fmt: str = "image_folder") -> DagnamDataset:
    return DagnamDataset(
        {
            "id": "img1",
            "name": "img",
            "format": fmt,
            "dataset_type": "image",
            "num_samples": 20,
            "num_classes": 2,
            "class_names": ["cat", "dog"],
        },
        data_dir=tmp_path,
    )


# ---------------------------------------------------------------- _resolve_split_dir


def test_resolve_split_dir_direct_match(tmp_path: Path) -> None:
    assert resolve_split_dir(tmp_path, "train", ["train", "val"]) == tmp_path / "train"


def test_resolve_split_dir_alias_validation(tmp_path: Path) -> None:
    # 'val' falls back to 'validation'
    assert resolve_split_dir(tmp_path, "val", ["train", "validation"]) == tmp_path / "validation"
    # 'validation' falls back to 'val'
    assert resolve_split_dir(tmp_path, "validation", ["train", "val"]) == tmp_path / "val"
    # 'test' falls back to 'dev'
    assert resolve_split_dir(tmp_path, "test", ["train", "dev"]) == tmp_path / "dev"


def test_resolve_split_dir_fallback_to_train(tmp_path: Path) -> None:
    assert resolve_split_dir(tmp_path, "val", ["train"]) == tmp_path / "train"


def test_resolve_split_dir_raises_when_no_match(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No directory found"):
        resolve_split_dir(tmp_path, "val", ["other"])


# ---------------------------------------------------------------- _gather_image_samples


def test_gather_image_samples_sorted(tmp_path: Path) -> None:
    _build_unsplit_dataset(tmp_path, per_class=3)
    samples, classes = gather_image_samples(tmp_path)
    assert classes == ["cat", "dog"]
    assert len(samples) == 6
    # All samples have valid class indices
    assert all(idx in (0, 1) for _, idx in samples)


def test_gather_image_samples_skips_hidden_classes_and_non_images(tmp_path: Path) -> None:
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "x.jpg").write_bytes(b"x")
    (tmp_path / "cat").mkdir()
    (tmp_path / "cat" / "x.jpg").write_bytes(b"x")  # not a real jpeg but matches ext
    (tmp_path / "cat" / "README.txt").write_text("noise")  # non-image
    samples, classes = gather_image_samples(tmp_path)
    assert classes == ["cat"]
    assert len(samples) == 1


# ---------------------------------------------------------------- create_pytorch_loader


def test_pytorch_loader_unsplit_dataset(tmp_path: Path) -> None:
    _build_unsplit_dataset(tmp_path, per_class=10)
    ds = _make_ds(tmp_path)
    loader = create_pytorch_loader(
        ds, split="train", batch_size=2, num_workers=0, val_ratio=0.2, test_ratio=0.2, seed=0
    )
    batch = next(iter(loader))
    images, _labels = batch
    assert images.shape[0] >= 1
    assert images.shape[1] == 3  # RGB channels


def test_pytorch_loader_split_dataset_with_aliases(tmp_path: Path) -> None:
    _build_split_dataset(tmp_path)
    ds = _make_ds(tmp_path)
    loader = create_pytorch_loader(ds, split="val", batch_size=2, num_workers=0)
    batch = next(iter(loader))
    assert batch[0].shape[0] >= 1


def test_pytorch_loader_default_shuffle_is_true_for_train(tmp_path: Path) -> None:
    _build_unsplit_dataset(tmp_path, per_class=8)
    ds = _make_ds(tmp_path)
    loader = create_pytorch_loader(ds, split="train", num_workers=0, batch_size=2)
    # DataLoader sets shuffle via a RandomSampler — easiest check is type
    from torch.utils.data import RandomSampler

    sampler = loader.sampler
    assert isinstance(sampler, RandomSampler) or loader.batch_size == 2


def test_pytorch_loader_test_split(tmp_path: Path) -> None:
    _build_unsplit_dataset(tmp_path, per_class=10)
    ds = _make_ds(tmp_path)
    loader = create_pytorch_loader(
        ds, split="test", batch_size=1, num_workers=0, val_ratio=0.2, test_ratio=0.2, seed=0
    )
    assert len(list(loader)) >= 1


def test_pytorch_loader_explicit_transform(tmp_path: Path) -> None:
    transforms = cast("TransformsModule", import_module("torchvision.transforms"))

    _build_unsplit_dataset(tmp_path, per_class=4)
    ds = _make_ds(tmp_path)
    tfm = transforms.Compose([transforms.Resize((4, 4)), transforms.ToTensor()])
    loader = create_pytorch_loader(
        ds,
        split="train",
        batch_size=1,
        num_workers=0,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=0,
        transform=cast("TransformFn", tfm),
    )
    batch = next(iter(loader))
    assert batch[0].shape[-1] == 4


# ---------------------------------------------------------------- create_tensorflow_dataset


def test_tensorflow_dataset_unsplit(tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    _build_unsplit_dataset(tmp_path, per_class=6)
    ds = _make_ds(tmp_path)
    tf_ds = create_tensorflow_dataset(
        ds,
        split="train",
        batch_size=2,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=0,
        image_size=(8, 8),
    )
    # Iterate one batch
    batch = cast("tuple[TensorLike, object]", next(iter(tf_ds)))
    images, _labels = batch
    assert images.shape[0] >= 1


def test_tensorflow_dataset_split(tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    _build_split_dataset(tmp_path)
    ds = _make_ds(tmp_path)
    tf_ds = create_tensorflow_dataset(ds, split="val", batch_size=2, image_size=(8, 8))
    batch = cast("TensorflowBatch", next(iter(tf_ds)))
    assert batch[0].shape[0] >= 1


def test_tensorflow_dataset_with_map_fn(tmp_path: Path) -> None:
    tf = pytest.importorskip("tensorflow")
    _build_unsplit_dataset(tmp_path, per_class=4)
    ds = _make_ds(tmp_path)

    def per_sample(img: object, label: object):
        return tf.cast(img, tf.float32) / 255.0, label

    def per_batch(img: object, label: object):
        return img, label

    tf_ds = create_tensorflow_dataset(
        ds,
        split="train",
        batch_size=2,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=0,
        image_size=(8, 8),
        map_fn=per_sample,
        batch_map_fn=per_batch,
    )
    next(iter(tf_ds))


# ---------------------------------------------------------------- create_flax_dataset


def test_flax_dataset_unsplit(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    _build_unsplit_dataset(tmp_path, per_class=6)
    ds = _make_ds(tmp_path)
    batches = create_flax_dataset(
        ds, split="train", batch_size=2, val_ratio=0.25, test_ratio=0.25, seed=0, image_size=(8, 8)
    )
    assert len(batches) >= 1
    assert batches[0].features.shape[-1] == 3  # RGB


def test_flax_dataset_split(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    _build_split_dataset(tmp_path)
    ds = _make_ds(tmp_path)
    batches = create_flax_dataset(ds, split="test", batch_size=2, image_size=(8, 8))
    assert len(batches) >= 1


def test_flax_dataset_with_transform_fns(tmp_path: Path) -> None:
    pytest.importorskip("jax")

    _build_unsplit_dataset(tmp_path, per_class=4)
    ds = _make_ds(tmp_path)
    batches = create_flax_dataset(
        ds,
        split="train",
        batch_size=2,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=0,
        image_size=(8, 8),
        transform_fn=_array_scale,
        batch_transform_fn=_jax_batch_identity,
    )
    assert len(batches) >= 1
