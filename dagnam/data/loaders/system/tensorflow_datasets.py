"""TensorFlow Datasets-backed system dataset loaders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dagnam.data.loaders.system.common import _SYSTEM_CACHE_ROOT
from dagnam.data.loaders.system.registry import resolve_system_dataset

if TYPE_CHECKING:
    from dagnam.data.dataset import DagnamDataset

_TFDS_NAME_MAP: dict[str, str] = {
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


def _resolve_tfds_name(meta: dict) -> str | None:
    """Return the tensorflow_datasets name for a system dataset, or None."""
    name = meta.get("name", "").lower()
    if name in _TFDS_NAME_MAP:
        return _TFDS_NAME_MAP[name]
    for key, tfds_name in _TFDS_NAME_MAP.items():
        if key in name or name in key:
            return tfds_name
    return None


def resolve_system_dataset_tf(meta: dict) -> DagnamDataset:
    """Load a system dataset as a native ``tf.data.Dataset`` via ``tensorflow_datasets``.

    Falls back to the PyTorch native loader (which is then converted
    in-memory by ``DagnamDataset._native_to_tensorflow``) if ``tfds`` is not
    installed or the dataset is not recognized.
    """
    from dagnam.data.dataset import DagnamDataset

    tfds_name = _resolve_tfds_name(meta)
    if tfds_name is None:
        # Fall back to PT native + in-memory conversion.
        return resolve_system_dataset(meta)

    try:
        import tensorflow_datasets as tfds
    except ImportError:
        # Fall back — caller uses _native_to_tensorflow on PT native.
        return resolve_system_dataset(meta)

    cache = _SYSTEM_CACHE_ROOT / tfds_name
    cache.mkdir(parents=True, exist_ok=True)

    train_ds = tfds.load(tfds_name, split="train", as_supervised=True, data_dir=str(cache))
    test_ds = tfds.load(tfds_name, split="test", as_supervised=True, data_dir=str(cache))

    return DagnamDataset(
        meta,
        cache,
        _native_train_tf=train_ds,
        _native_test_tf=test_ds,
    )
