"""Native PyTorch/torchvision system dataset loaders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dagnam.data.loaders.system.common import _SYSTEM_CACHE_ROOT

if TYPE_CHECKING:
    from dagnam.data.dataset import DagnamDataset


def _load_mnist(meta: dict, transform=None) -> DagnamDataset:
    from torchvision import datasets, transforms

    from dagnam.data.dataset import DagnamDataset

    cache = _SYSTEM_CACHE_ROOT / "mnist"
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


def _load_cifar10(meta: dict, transform=None) -> DagnamDataset:
    from torchvision import datasets, transforms

    from dagnam.data.dataset import DagnamDataset

    cache = _SYSTEM_CACHE_ROOT / "cifar10"
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


def _load_cifar100(meta: dict, transform=None) -> DagnamDataset:
    from torchvision import datasets, transforms

    from dagnam.data.dataset import DagnamDataset

    cache = _SYSTEM_CACHE_ROOT / "cifar100"
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


def _load_fashion_mnist(meta: dict, transform=None) -> DagnamDataset:
    from torchvision import datasets, transforms

    from dagnam.data.dataset import DagnamDataset

    cache = _SYSTEM_CACHE_ROOT / "fashion_mnist"
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


def _load_imdb(meta: dict, transform=None) -> DagnamDataset:
    """Load IMDB via direct npz download (no TensorFlow dependency)."""
    import urllib.request

    import numpy as np

    from dagnam.data.dataset import DagnamDataset

    cache = _SYSTEM_CACHE_ROOT / "imdb"
    cache.mkdir(parents=True, exist_ok=True)
    npz_path = cache / "imdb.npz"

    if not npz_path.exists():
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz"
        urllib.request.urlretrieve(url, str(npz_path))

    # Build a simple pandas DataFrame so the existing to_pytorch_loader
    # file-based path can work.  However, we also set _native_train/test
    # as numpy arrays for direct use.
    with np.load(str(npz_path), allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

    return DagnamDataset(
        meta,
        cache,
        _native_train=(x_train, y_train),
        _native_test=(x_test, y_test),
    )


def _load_oxford_pets(meta: dict, transform=None) -> DagnamDataset:
    from torchvision import datasets, transforms

    from dagnam.data.dataset import DagnamDataset

    cache = _SYSTEM_CACHE_ROOT / "oxford_pets"
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


def _load_speech_commands(meta: dict, transform=None) -> DagnamDataset:
    from dagnam.data.dataset import DagnamDataset

    cache = _SYSTEM_CACHE_ROOT / "speech_commands"
    cache.mkdir(parents=True, exist_ok=True)

    try:
        import torchaudio

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


def _load_wikitext2(meta: dict, transform=None) -> DagnamDataset:
    from dagnam.data.dataset import DagnamDataset

    cache = _SYSTEM_CACHE_ROOT / "wikitext2"
    cache.mkdir(parents=True, exist_ok=True)

    try:
        from torchtext.datasets import WikiText2

        # torchtext returns iterators, not map-style datasets
        # Store as native for custom handling
        train_iter = WikiText2(root=str(cache), split="train")
        test_iter = WikiText2(root=str(cache), split="test")
        return DagnamDataset(
            meta,
            cache,
            _native_train=list(train_iter),
            _native_test=list(test_iter),
        )
    except (ImportError, Exception):
        return DagnamDataset(meta, cache)
