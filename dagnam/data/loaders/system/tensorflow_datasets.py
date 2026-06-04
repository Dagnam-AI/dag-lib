"""TensorFlow Datasets-backed system dataset loaders."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from dagnam._types import JsonObject, TensorflowDataset
from dagnam.data.loaders.system.common import SYSTEM_CACHE_ROOT
from dagnam.data.loaders.system.registry import resolve_system_dataset

if TYPE_CHECKING:
    from dagnam.data.dataset import DagnamDataset

TFDS_NAME_MAP: dict[str, str] = {
    "mnist": "mnist",
    "mnist handwritten digits": "mnist",
    "cifar-10": "cifar10",
    "cifar10": "cifar10",
    "cifar-100": "cifar100",
    "cifar100": "cifar100",
    "fashion-mnist": "fashion_mnist",
    "fashionmnist": "fashion_mnist",
    "fashion_mnist": "fashion_mnist",
    "imdb": "imdb_reviews",
}


class TensorflowDatasetsModule(Protocol):
    """tensorflow_datasets surface used by this adapter."""

    def load(
        self,
        name: str,
        *,
        split: str,
        as_supervised: bool,
        data_dir: str,
    ) -> TensorflowDataset: ...


def _load_tfds() -> TensorflowDatasetsModule:
    return cast("TensorflowDatasetsModule", import_module("tensorflow_datasets"))


def _load_supervised_split(tfds: Any, name: str, split: str, cache: Path) -> TensorflowDataset:
    try:
        return cast(
            "TensorflowDataset",
            tfds.load(name, split=split, as_supervised=True, data_dir=str(cache)),
        )
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        # Positional fallback for older tensorflow_datasets whose `load`
        # signature predates keyword-only args; `True` is `as_supervised`.
        return cast("TensorflowDataset", tfds.load(name, split, True, cache))  # noqa: FBT003


def resolve_tfds_name(meta: JsonObject) -> str | None:
    """Return the tensorflow_datasets name for a system dataset, or None."""
    raw_name = meta.get("name", "")
    if not isinstance(raw_name, str):
        return None
    name = raw_name.lower()
    if name in TFDS_NAME_MAP:
        return TFDS_NAME_MAP[name]
    for key, tfds_name in TFDS_NAME_MAP.items():
        if key in name or name in key:
            return tfds_name
    return None


def resolve_system_dataset_tf(meta: JsonObject) -> DagnamDataset:
    """Load a system dataset as a native ``tf.data.Dataset`` via ``tensorflow_datasets``.

    Falls back to the PyTorch native loader (which is then converted
    in-memory by ``DagnamDataset._native_to_tensorflow``) if ``tfds`` is not
    installed or the dataset is not recognized.
    """
    from dagnam.data.dataset import DagnamDataset

    tfds_name = resolve_tfds_name(meta)
    if tfds_name is None:
        # Fall back to PT native + in-memory conversion.
        return resolve_system_dataset(meta)

    try:
        tfds = _load_tfds()
    except ImportError:
        # Fall back — caller uses _native_to_tensorflow on PT native.
        return resolve_system_dataset(meta)

    cache = SYSTEM_CACHE_ROOT / tfds_name
    cache.mkdir(parents=True, exist_ok=True)

    train_ds = _load_supervised_split(tfds, tfds_name, "train", cache)
    test_ds = _load_supervised_split(tfds, tfds_name, "test", cache)

    return DagnamDataset(
        meta,
        cache,
        _native_train_tf=train_ds,
        _native_test_tf=test_ds,
    )
