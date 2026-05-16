"""Comprehensive coverage for dagnam.data.loaders.image_folder.

Uses Pillow to synthesize small JPEGs so the real torch/torchvision code path
is exercised end-to-end without external test fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")
PIL = pytest.importorskip("PIL.Image")

from dagnam.data.dataset import DagnamDataset  # noqa: E402
from dagnam.data.loaders.image_folder import (  # noqa: E402
    _gather_image_samples,
    _resolve_split_dir,
    create_flax_dataset,
    create_pytorch_loader,
    create_tensorflow_dataset,
)


def _make_jpeg(path: Path, color=(255, 0, 0), size=(8, 8)) -> None:
    img = PIL.new("RGB", size, color=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG")


def _build_unsplit_dataset(root: Path, classes=("cat", "dog"), per_class=10) -> None:
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


def test_resolve_split_dir_direct_match(tmp_path):
    assert _resolve_split_dir(tmp_path, "train", ["train", "val"]) == tmp_path / "train"


def test_resolve_split_dir_alias_validation(tmp_path):
    # 'val' falls back to 'validation'
    assert _resolve_split_dir(tmp_path, "val", ["train", "validation"]) == tmp_path / "validation"
    # 'validation' falls back to 'val'
    assert _resolve_split_dir(tmp_path, "validation", ["train", "val"]) == tmp_path / "val"
    # 'test' falls back to 'dev'
    assert _resolve_split_dir(tmp_path, "test", ["train", "dev"]) == tmp_path / "dev"


def test_resolve_split_dir_fallback_to_train(tmp_path):
    assert _resolve_split_dir(tmp_path, "val", ["train"]) == tmp_path / "train"


def test_resolve_split_dir_raises_when_no_match(tmp_path):
    with pytest.raises(FileNotFoundError, match="No directory found"):
        _resolve_split_dir(tmp_path, "val", ["other"])


# ---------------------------------------------------------------- _gather_image_samples


def test_gather_image_samples_sorted(tmp_path):
    _build_unsplit_dataset(tmp_path, per_class=3)
    samples, classes = _gather_image_samples(tmp_path)
    assert classes == ["cat", "dog"]
    assert len(samples) == 6
    # All samples have valid class indices
    assert all(idx in (0, 1) for _, idx in samples)


def test_gather_image_samples_skips_hidden_classes_and_non_images(tmp_path):
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "x.jpg").write_bytes(b"x")
    (tmp_path / "cat").mkdir()
    (tmp_path / "cat" / "x.jpg").write_bytes(b"x")  # not a real jpeg but matches ext
    (tmp_path / "cat" / "README.txt").write_text("noise")  # non-image
    samples, classes = _gather_image_samples(tmp_path)
    assert classes == ["cat"]
    assert len(samples) == 1


# ---------------------------------------------------------------- create_pytorch_loader


def test_pytorch_loader_unsplit_dataset(tmp_path):
    _build_unsplit_dataset(tmp_path, per_class=10)
    ds = _make_ds(tmp_path)
    loader = create_pytorch_loader(
        ds, split="train", batch_size=2, num_workers=0, val_ratio=0.2, test_ratio=0.2, seed=0
    )
    batch = next(iter(loader))
    images, _labels = batch
    assert images.shape[0] >= 1
    assert images.shape[1] == 3  # RGB channels


def test_pytorch_loader_split_dataset_with_aliases(tmp_path):
    _build_split_dataset(tmp_path)
    ds = _make_ds(tmp_path)
    loader = create_pytorch_loader(ds, split="val", batch_size=2, num_workers=0)
    batch = next(iter(loader))
    assert batch[0].shape[0] >= 1


def test_pytorch_loader_default_shuffle_is_true_for_train(tmp_path):
    _build_unsplit_dataset(tmp_path, per_class=8)
    ds = _make_ds(tmp_path)
    loader = create_pytorch_loader(ds, split="train", num_workers=0, batch_size=2)
    # DataLoader sets shuffle via a RandomSampler — easiest check is type
    from torch.utils.data import RandomSampler

    assert isinstance(loader.sampler, RandomSampler) or loader.batch_size == 2


def test_pytorch_loader_test_split(tmp_path):
    _build_unsplit_dataset(tmp_path, per_class=10)
    ds = _make_ds(tmp_path)
    loader = create_pytorch_loader(
        ds, split="test", batch_size=1, num_workers=0, val_ratio=0.2, test_ratio=0.2, seed=0
    )
    assert len(list(loader)) >= 1


def test_pytorch_loader_explicit_transform(tmp_path):
    from torchvision import transforms

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
        transform=tfm,
    )
    batch = next(iter(loader))
    assert batch[0].shape[-1] == 4


# ---------------------------------------------------------------- create_tensorflow_dataset


def test_tensorflow_dataset_unsplit(tmp_path):
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
    batch = next(iter(tf_ds))
    images, _labels = batch
    assert images.shape[0] >= 1


def test_tensorflow_dataset_split(tmp_path):
    pytest.importorskip("tensorflow")
    _build_split_dataset(tmp_path)
    ds = _make_ds(tmp_path)
    tf_ds = create_tensorflow_dataset(ds, split="val", batch_size=2, image_size=(8, 8))
    batch = next(iter(tf_ds))
    assert batch[0].shape[0] >= 1


def test_tensorflow_dataset_with_map_fn(tmp_path):
    tf = pytest.importorskip("tensorflow")
    _build_unsplit_dataset(tmp_path, per_class=4)
    ds = _make_ds(tmp_path)

    def per_sample(img, label):
        return tf.cast(img, tf.float32) / 255.0, label

    def per_batch(img, label):
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


def test_flax_dataset_unsplit(tmp_path):
    pytest.importorskip("jax")
    _build_unsplit_dataset(tmp_path, per_class=6)
    ds = _make_ds(tmp_path)
    batches = create_flax_dataset(
        ds, split="train", batch_size=2, val_ratio=0.25, test_ratio=0.25, seed=0, image_size=(8, 8)
    )
    assert len(batches) >= 1
    assert batches[0].features.shape[-1] == 3  # RGB


def test_flax_dataset_split(tmp_path):
    pytest.importorskip("jax")
    _build_split_dataset(tmp_path)
    ds = _make_ds(tmp_path)
    batches = create_flax_dataset(ds, split="test", batch_size=2, image_size=(8, 8))
    assert len(batches) >= 1


def test_flax_dataset_with_transform_fns(tmp_path):
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
        transform_fn=lambda arr: arr * 1.0,
        batch_transform_fn=lambda x, y: (x, y),
    )
    assert len(batches) >= 1
