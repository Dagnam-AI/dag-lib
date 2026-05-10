"""System dataset native loader registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dagnam._core.exceptions import DatasetNotFoundError
from dagnam.data.loaders.system.torchvision import (
    _load_cifar10,
    _load_cifar100,
    _load_fashion_mnist,
    _load_imdb,
    _load_mnist,
    _load_oxford_pets,
    _load_speech_commands,
    _load_wikitext2,
)

if TYPE_CHECKING:
    from dagnam.data.dataset import DagnamDataset


def resolve_system_dataset(meta: dict, transform=None) -> DagnamDataset:
    """Load a system dataset using its native library internally.

    Matches on the dataset name (case-insensitive, fuzzy).  Returns a
    ``DagnamDataset`` with ``_native_train`` and ``_native_test`` set.

    Raises:
        DatasetNotFoundError: If no native loader exists for the dataset.
    """
    name = meta.get("name", "").lower()

    # Exact-match first, then substring
    loader = _NATIVE_LOADERS.get(name)
    if loader is None:
        for key, fn in _NATIVE_LOADERS.items():
            if key in name or name in key:
                loader = fn
                break

    if loader is None:
        raise DatasetNotFoundError(
            f"System dataset '{meta.get('name')}' has no native loader. "
            f"Contact support or use a user-uploaded version."
        )

    return loader(meta, transform=transform)


_NATIVE_LOADERS: dict[str, Any] = {
    "mnist handwritten digits": _load_mnist,
    "mnist": _load_mnist,
    "cifar-10": _load_cifar10,
    "cifar10": _load_cifar10,
    "cifar-100": _load_cifar100,
    "cifar100": _load_cifar100,
    "fashion mnist": _load_fashion_mnist,
    "fashion-mnist": _load_fashion_mnist,
    "fashionmnist": _load_fashion_mnist,
    "imdb movie reviews": _load_imdb,
    "imdb": _load_imdb,
    "oxford-iiit pet dataset": _load_oxford_pets,
    "oxford pets": _load_oxford_pets,
    "speech commands": _load_speech_commands,
    "wikitext-2": _load_wikitext2,
    "wikitext2": _load_wikitext2,
}
