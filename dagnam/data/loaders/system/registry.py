"""System dataset native loader registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from dagnam._types import JsonObject
from dagnam._core.exceptions import DatasetNotFoundError
from dagnam.data.loaders.system.torchvision import (
    load_cifar10,
    load_cifar100,
    load_fashion_mnist,
    load_imdb,
    load_mnist,
    load_oxford_pets,
    load_speech_commands,
    load_wikitext2,
)

if TYPE_CHECKING:
    from dagnam.data.dataset import DagnamDataset

TransformFn = Callable[[object], object]
NativeLoader = Callable[[JsonObject, TransformFn | None], "DagnamDataset"]


def resolve_system_dataset(meta: JsonObject, transform: TransformFn | None = None) -> DagnamDataset:
    """Load a system dataset using its native library internally.

    Matches on the dataset name (case-insensitive, fuzzy).  Returns a
    ``DagnamDataset`` with ``_native_train`` and ``_native_test`` set.

    Raises:
        DatasetNotFoundError: If no native loader exists for the dataset.
    """
    name_value = meta.get("name", "")
    name = name_value.lower() if isinstance(name_value, str) else ""

    # Exact-match first, then substring
    loader = NATIVE_LOADERS.get(name)
    if loader is None:
        for key, fn in NATIVE_LOADERS.items():
            if key in name or name in key:
                loader = fn
                break

    if loader is None:
        raise DatasetNotFoundError(
            f"System dataset '{meta.get('name')}' has no native loader. "
            f"Contact support or use a user-uploaded version."
        )

    return loader(meta, transform)


NATIVE_LOADERS: dict[str, NativeLoader] = {
    "mnist handwritten digits": load_mnist,
    "mnist": load_mnist,
    "cifar-10": load_cifar10,
    "cifar10": load_cifar10,
    "cifar-100": load_cifar100,
    "cifar100": load_cifar100,
    "fashion mnist": load_fashion_mnist,
    "fashion-mnist": load_fashion_mnist,
    "fashionmnist": load_fashion_mnist,
    "imdb movie reviews": load_imdb,
    "imdb": load_imdb,
    "oxford-iiit pet dataset": load_oxford_pets,
    "oxford pets": load_oxford_pets,
    "speech commands": load_speech_commands,
    "wikitext-2": load_wikitext2,
    "wikitext2": load_wikitext2,
}
