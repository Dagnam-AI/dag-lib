"""Native PyTorch/torchvision system dataset loaders."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import hashlib
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast
from urllib.parse import urlparse

import requests

from dagnam._types import IndexedDataset, JsonObject, NativeSplit
from dagnam.data.loaders.system.common import SYSTEM_CACHE_ROOT

if TYPE_CHECKING:
    from dagnam.data.dataset import DagnamDataset

_IMDB_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz"
_IMDB_SHA256 = "69664113be75683a8fe16e3ed0ab59fda8886cb3cd7ada244f7d9544e4676b9f"
IMDB_SHA256 = _IMDB_SHA256
_DOWNLOAD_TIMEOUT = (30, 60)
TransformFn = Callable[[object], object]


class TorchVisionTransformsModule(Protocol):
    """TorchVision transform constructors used by system datasets."""

    def Compose(self, transforms: Sequence[TransformFn]) -> TransformFn: ...

    def ToTensor(self) -> TransformFn: ...

    def Normalize(self, mean: Sequence[float], std: Sequence[float]) -> TransformFn: ...

    def Resize(self, size: tuple[int, int]) -> TransformFn: ...


class TorchVisionDatasetsModule(Protocol):
    """TorchVision dataset constructors used by system datasets."""

    def MNIST(
        self,
        *,
        root: str,
        train: bool,
        download: bool,
        transform: TransformFn | None,
    ) -> IndexedDataset: ...

    def CIFAR10(
        self,
        *,
        root: str,
        train: bool,
        download: bool,
        transform: TransformFn | None,
    ) -> IndexedDataset: ...

    def CIFAR100(
        self,
        *,
        root: str,
        train: bool,
        download: bool,
        transform: TransformFn | None,
    ) -> IndexedDataset: ...

    def FashionMNIST(
        self,
        *,
        root: str,
        train: bool,
        download: bool,
        transform: TransformFn | None,
    ) -> IndexedDataset: ...

    def OxfordIIITPet(
        self,
        *,
        root: str,
        split: str,
        download: bool,
        transform: TransformFn | None,
    ) -> IndexedDataset: ...


class TorchaudioDatasetsModule(Protocol):
    """Torchaudio dataset constructors used by system datasets."""

    def SPEECHCOMMANDS(self, *, root: str, download: bool, subset: str) -> IndexedDataset: ...


class TorchaudioModule(Protocol):
    """Torchaudio module surface used by system datasets."""

    datasets: TorchaudioDatasetsModule


class TorchTextDatasetsModule(Protocol):
    """TorchText dataset constructors used by system datasets."""

    def WikiText2(self, *, root: str, split: str) -> Iterable[str]: ...


def _load_torchvision() -> tuple[TorchVisionDatasetsModule, TorchVisionTransformsModule]:
    return (
        cast(TorchVisionDatasetsModule, import_module("torchvision.datasets")),
        cast(TorchVisionTransformsModule, import_module("torchvision.transforms")),
    )


def _load_torchaudio() -> TorchaudioModule:
    return cast(TorchaudioModule, import_module("torchaudio"))


def _load_torchtext_datasets() -> TorchTextDatasetsModule:
    return cast(TorchTextDatasetsModule, import_module("torchtext.datasets"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified_file(url: str, dest: Path, expected_sha256: str) -> None:
    """Download an HTTPS file and atomically install it only if SHA-256 matches."""
    if urlparse(url).scheme != "https":
        raise ValueError("System dataset downloads must use HTTPS URLs")

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)

        actual = sha256(tmp)
        if actual != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise ValueError(
                f"Downloaded system dataset checksum mismatch: expected {expected_sha256}, got {actual}"
            )
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


_download_verified_file = download_verified_file


def ensure_verified_file(url: str, dest: Path, expected_sha256: str) -> None:
    if dest.exists() and sha256(dest) == expected_sha256:
        return
    dest.unlink(missing_ok=True)
    download_verified_file(url, dest, expected_sha256)


def load_mnist(meta: JsonObject, transform: TransformFn | None = None) -> DagnamDataset:
    from dagnam.data.dataset import DagnamDataset

    datasets, transforms = _load_torchvision()
    cache = SYSTEM_CACHE_ROOT / "mnist"
    cache.mkdir(parents=True, exist_ok=True)

    # When the caller passes a custom transform, honor it as-is (caller owns normalization).
    # Otherwise, preserve the historical bundled default (ToTensor + dataset-specific Normalize).
    base_transform = (
        transform
        if transform is not None
        else transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    )

    train_ds = datasets.MNIST(root=str(cache), train=True, download=True, transform=base_transform)
    test_ds = datasets.MNIST(root=str(cache), train=False, download=True, transform=base_transform)

    return DagnamDataset(meta, cache, _native_train=train_ds, _native_test=test_ds)


def load_cifar10(meta: JsonObject, transform: TransformFn | None = None) -> DagnamDataset:
    from dagnam.data.dataset import DagnamDataset

    datasets, transforms = _load_torchvision()
    cache = SYSTEM_CACHE_ROOT / "cifar10"
    cache.mkdir(parents=True, exist_ok=True)

    base_transform = (
        transform
        if transform is not None
        else transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
    )

    train_ds = datasets.CIFAR10(
        root=str(cache), train=True, download=True, transform=base_transform
    )
    test_ds = datasets.CIFAR10(
        root=str(cache), train=False, download=True, transform=base_transform
    )

    return DagnamDataset(meta, cache, _native_train=train_ds, _native_test=test_ds)


def load_cifar100(meta: JsonObject, transform: TransformFn | None = None) -> DagnamDataset:
    from dagnam.data.dataset import DagnamDataset

    datasets, transforms = _load_torchvision()
    cache = SYSTEM_CACHE_ROOT / "cifar100"
    cache.mkdir(parents=True, exist_ok=True)

    base_transform = (
        transform
        if transform is not None
        else transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
    )

    train_ds = datasets.CIFAR100(
        root=str(cache), train=True, download=True, transform=base_transform
    )
    test_ds = datasets.CIFAR100(
        root=str(cache), train=False, download=True, transform=base_transform
    )

    return DagnamDataset(meta, cache, _native_train=train_ds, _native_test=test_ds)


def load_fashion_mnist(meta: JsonObject, transform: TransformFn | None = None) -> DagnamDataset:
    from dagnam.data.dataset import DagnamDataset

    datasets, transforms = _load_torchvision()
    cache = SYSTEM_CACHE_ROOT / "fashion_mnist"
    cache.mkdir(parents=True, exist_ok=True)

    base_transform = (
        transform
        if transform is not None
        else transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )
    )

    train_ds = datasets.FashionMNIST(
        root=str(cache), train=True, download=True, transform=base_transform
    )
    test_ds = datasets.FashionMNIST(
        root=str(cache), train=False, download=True, transform=base_transform
    )

    return DagnamDataset(meta, cache, _native_train=train_ds, _native_test=test_ds)


def load_imdb(meta: JsonObject, transform: TransformFn | None = None) -> DagnamDataset:
    """Load IMDB via direct npz download (no TensorFlow dependency)."""
    import numpy as np
    from dagnam.data.dataset import DagnamDataset

    cache = SYSTEM_CACHE_ROOT / "imdb"
    cache.mkdir(parents=True, exist_ok=True)
    npz_path = cache / "imdb.npz"

    ensure_verified_file(_IMDB_URL, npz_path, IMDB_SHA256)

    # Build a simple polars DataFrame so the existing to_pytorch_loader
    # file-based path can work.  However, we also set _native_train/test
    # as numpy arrays for direct use.
    # The upstream Keras IMDB npz stores ragged review sequences as object arrays,
    # so NumPy requires pickle support. The pinned SHA-256 check above prevents
    # network or cache tampering before this trusted file is deserialized.
    with np.load(str(npz_path), allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

    return DagnamDataset(
        meta,
        cache,
        _native_train=(x_train, y_train),
        _native_test=(x_test, y_test),
    )


def load_oxford_pets(meta: JsonObject, transform: TransformFn | None = None) -> DagnamDataset:
    from dagnam.data.dataset import DagnamDataset

    datasets, transforms = _load_torchvision()
    cache = SYSTEM_CACHE_ROOT / "oxford_pets"
    cache.mkdir(parents=True, exist_ok=True)

    base_transform = transform
    if base_transform is None:
        base_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    try:
        train_ds = datasets.OxfordIIITPet(
            root=str(cache),
            split="trainval",
            download=True,
            transform=base_transform,
        )
        test_ds = datasets.OxfordIIITPet(
            root=str(cache),
            split="test",
            download=True,
            transform=base_transform,
        )
    except Exception:
        # Fallback: if torchvision doesn't have OxfordIIITPet, return file-based
        return DagnamDataset(meta, cache)

    return DagnamDataset(meta, cache, _native_train=train_ds, _native_test=test_ds)


def load_speech_commands(meta: JsonObject, transform: TransformFn | None = None) -> DagnamDataset:
    from dagnam.data.dataset import DagnamDataset

    cache = SYSTEM_CACHE_ROOT / "speech_commands"
    cache.mkdir(parents=True, exist_ok=True)

    try:
        torchaudio = _load_torchaudio()
        train_ds = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(cache),
            download=True,
            subset="training",
        )
        test_ds = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(cache),
            download=True,
            subset="testing",
        )
        return DagnamDataset(meta, cache, _native_train=train_ds, _native_test=test_ds)
    except (ImportError, Exception):
        # torchaudio not installed or download failed — return file-based
        return DagnamDataset(meta, cache)


def load_wikitext2(meta: JsonObject, transform: TransformFn | None = None) -> DagnamDataset:
    from dagnam.data.dataset import DagnamDataset

    cache = SYSTEM_CACHE_ROOT / "wikitext2"
    cache.mkdir(parents=True, exist_ok=True)

    try:
        torchtext_datasets = _load_torchtext_datasets()

        # torchtext returns iterators, not map-style datasets
        # Store as native for custom handling
        train_iter = torchtext_datasets.WikiText2(root=str(cache), split="train")
        test_iter = torchtext_datasets.WikiText2(root=str(cache), split="test")
        train_items: NativeSplit = cast(NativeSplit, list(train_iter))
        test_items: NativeSplit = cast(NativeSplit, list(test_iter))
        return DagnamDataset(
            meta,
            cache,
            _native_train=train_items,
            _native_test=test_items,
        )
    except (ImportError, Exception):
        return DagnamDataset(meta, cache)
