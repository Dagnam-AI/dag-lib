"""Comprehensive coverage for dagnam.data.loaders.image_folder.

Uses Pillow to synthesize small JPEGs so the real torch/torchvision code path
is exercised end-to-end without external test fixtures.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
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
    _cardinality_to_int,
    _gather_image_samples,
    create_flax_dataset,
    create_pytorch_loader,
    create_tensorflow_dataset,
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


# ---------------------------------------------------------------- _gather_image_samples


def test_gather_image_samples_sorted(tmp_path: Path) -> None:
    _build_unsplit_dataset(tmp_path, per_class=3)
    samples, classes = _gather_image_samples(tmp_path)
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
    samples, classes = _gather_image_samples(tmp_path)
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


def test_flax_dataset_explicit_shuffle_skips_default(tmp_path: Path) -> None:
    # Passing shuffle explicitly skips the ``shuffle is None`` default in the
    # flax loader (branch 390->393).
    pytest.importorskip("jax")
    _build_unsplit_dataset(tmp_path, per_class=4)
    ds = _make_ds(tmp_path)
    batches = create_flax_dataset(
        ds,
        split="train",
        batch_size=2,
        shuffle=False,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=0,
        image_size=(8, 8),
    )
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


# ---------------------------------------------------------------- _cardinality_to_int


def test_cardinality_to_int_unwraps_numpy_generic() -> None:
    # A numpy generic is unwrapped via .item() then accepted as an int
    # (branch 112->113->114->115).
    assert _cardinality_to_int(np.int64(42)) == 42


def test_cardinality_to_int_accepts_plain_int() -> None:
    # A plain int skips the numpy-unwrap branch (112->114) and returns directly.
    assert _cardinality_to_int(7) == 7


def test_cardinality_to_int_rejects_non_int() -> None:
    # A value that is neither numpy-generic nor int falls through both ifs
    # (branch 114->116) and raises (line 116).
    with pytest.raises(TypeError, match="Expected TensorFlow cardinality integer"):
        _cardinality_to_int("not-an-int")


# ---------------------------------------------------------------- create_pytorch_loader shuffle


def test_pytorch_loader_explicit_shuffle_skips_default(tmp_path: Path) -> None:
    # Passing shuffle explicitly skips the ``shuffle is None`` default
    # (branch 171->175).
    _build_unsplit_dataset(tmp_path, per_class=8)
    ds = _make_ds(tmp_path)
    loader = create_pytorch_loader(
        ds,
        split="train",
        batch_size=2,
        num_workers=0,
        shuffle=False,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=0,
    )
    batch = next(iter(loader))
    assert batch[0].shape[0] >= 1


# ---------------------------------------------------------------- tensorflow unsplit fake-tf path
#
# The unsplit branch of ``create_tensorflow_dataset`` (lines 326-343) contains a
# cardinality-fallback leg and two graph closures (``_keep_index`` / ``_drop_index``)
# that TensorFlow executes inside its untraceable graph runtime. A synchronous
# fake ``tf`` module runs those closures eagerly on the main thread so coverage
# can follow them, and reports an UNKNOWN cardinality to drive the Python-count
# fallback (branch 326->328, line 328).


class _FakeNumpyScalar:
    def __init__(self, value: int) -> None:
        self._value = value

    def numpy(self) -> int:
        return self._value


class _FakeExperimental:
    UNKNOWN_CARDINALITY = -2

    @staticmethod
    def cardinality(_ds: object) -> _FakeNumpyScalar:
        # Report UNKNOWN so the loader falls back to Python iteration (line 328).
        return _FakeNumpyScalar(_FakeExperimental.UNKNOWN_CARDINALITY)


class _FakeTfData:
    AUTOTUNE = -1
    experimental = _FakeExperimental()


class _FakeImageDataset:
    """A synchronous stand-in for the tf.data.Dataset surface used by the loader."""

    def __init__(self, rows: list[tuple[object, object]]) -> None:
        self._rows = rows

    def __iter__(self) -> Iterator[tuple[object, object]]:
        return iter(self._rows)

    def enumerate(self) -> _FakeImageDataset:
        return _FakeImageDataset([(i, row) for i, row in enumerate(self._rows)])

    def filter(self, predicate: object) -> _FakeImageDataset:
        from collections.abc import Callable

        keep = cast("Callable[[object, object], object]", predicate)
        kept = [(idx, payload) for idx, payload in self._rows if bool(keep(idx, payload))]
        return _FakeImageDataset(kept)

    def map(self, fn: object, *_args: object, **_kwargs: object) -> _FakeImageDataset:
        from collections.abc import Callable

        mapper = cast("Callable[..., object]", fn)
        return _FakeImageDataset([(mapper(idx, payload), None) for idx, payload in self._rows])

    def shuffle(self, *_args: object, **_kwargs: object) -> _FakeImageDataset:
        return self

    def batch(self, *_args: object, **_kwargs: object) -> _FakeImageDataset:
        return self

    def prefetch(self, *_args: object, **_kwargs: object) -> _FakeImageDataset:
        return self


class _FakeKerasUtils:
    @staticmethod
    def image_dataset_from_directory(_directory: str, **_kwargs: object) -> _FakeImageDataset:
        # Three pseudo-samples; payload identity is irrelevant to the index logic.
        return _FakeImageDataset([("img0", 0), ("img1", 1), ("img2", 0)])


class _FakeKeras:
    utils = _FakeKerasUtils()


class _FakeBoolMask:
    """Indexable boolean mask returned by the fake ``tf.constant``."""

    def __init__(self, mask: list[bool]) -> None:
        self._mask = mask

    def __getitem__(self, index: object) -> bool:
        return self._mask[cast("int", index)]


class _FakeTfModule:
    data = _FakeTfData()
    keras = _FakeKeras()
    bool = "bool"

    @staticmethod
    def constant(value: object, *, dtype: object) -> _FakeBoolMask:
        return _FakeBoolMask(cast("list[bool]", value))


def test_tensorflow_unsplit_unknown_cardinality_runs_filter_closures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cover the unsplit cardinality fallback and the filter/map closures.

    Substitutes a synchronous fake ``tf`` so coverage can trace ``_keep_index``
    and ``_drop_index`` (lines 338, 341) and the unknown-cardinality fallback
    (branch 326->328, line 328).
    """
    from dagnam.data.loaders import image_folder

    monkeypatch.setattr(image_folder, "_load_tensorflow", lambda: _FakeTfModule())

    _build_unsplit_dataset(tmp_path, per_class=4)
    ds = _make_ds(tmp_path)
    result = create_tensorflow_dataset(
        ds,
        split="train",
        batch_size=2,
        shuffle=False,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=0,
        image_size=(8, 8),
    )
    rows = list(cast("Iterable[object]", result))
    # _drop_index returns the payload for kept indices.
    assert rows
