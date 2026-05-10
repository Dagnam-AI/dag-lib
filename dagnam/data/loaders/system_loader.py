"""Native loaders for system datasets (MNIST, CIFAR-10, etc.).

Each loader wraps a framework-specific library (torchvision, torchaudio)
and returns a ``DagnamDataset`` with ``_native_train`` / ``_native_test``
fields populated.  The dagnam public API (``to_pytorch_loader``, etc.)
detects these fields and uses the native datasets directly instead of
loading from a downloaded file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dagnam._core.exceptions import DatasetNotFoundError

# Cache root for system datasets downloaded by native libraries
_SYSTEM_CACHE_ROOT = Path.home() / ".dagnam" / "system_datasets"


def resolve_system_dataset(meta: dict, transform=None) -> "DagnamDataset":
    """Load a system dataset using its native library internally.

    Matches on the dataset name (case-insensitive, fuzzy).  Returns a
    ``DagnamDataset`` with ``_native_train`` and ``_native_test`` set.

    Raises:
        DatasetNotFoundError: If no native loader exists for the dataset.
    """
    from dagnam.data.dataset import DagnamDataset

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


# ------------------------------------------------------------------
# Individual loaders
# ------------------------------------------------------------------

def _load_mnist(meta: dict, transform=None) -> "DagnamDataset":
    from torchvision import datasets, transforms
    from dagnam.data.dataset import DagnamDataset

    cache = _SYSTEM_CACHE_ROOT / "mnist"
    cache.mkdir(parents=True, exist_ok=True)

    # When the caller passes a custom transform, honor it as-is (caller owns normalization).
    # Otherwise, preserve the historical bundled default (ToTensor + dataset-specific Normalize).
    base_transform = transform if transform is not None else transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST(root=str(cache), train=True, download=True, transform=base_transform)
    test_ds = datasets.MNIST(root=str(cache), train=False, download=True, transform=base_transform)

    return DagnamDataset(meta, cache, _native_train=train_ds, _native_test=test_ds)


def _load_cifar10(meta: dict, transform=None) -> "DagnamDataset":
    from torchvision import datasets, transforms
    from dagnam.data.dataset import DagnamDataset

    cache = _SYSTEM_CACHE_ROOT / "cifar10"
    cache.mkdir(parents=True, exist_ok=True)

    base_transform = transform if transform is not None else transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = datasets.CIFAR10(root=str(cache), train=True, download=True, transform=base_transform)
    test_ds = datasets.CIFAR10(root=str(cache), train=False, download=True, transform=base_transform)

    return DagnamDataset(meta, cache, _native_train=train_ds, _native_test=test_ds)


def _load_cifar100(meta: dict, transform=None) -> "DagnamDataset":
    from torchvision import datasets, transforms
    from dagnam.data.dataset import DagnamDataset

    cache = _SYSTEM_CACHE_ROOT / "cifar100"
    cache.mkdir(parents=True, exist_ok=True)

    base_transform = transform if transform is not None else transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_ds = datasets.CIFAR100(root=str(cache), train=True, download=True, transform=base_transform)
    test_ds = datasets.CIFAR100(root=str(cache), train=False, download=True, transform=base_transform)

    return DagnamDataset(meta, cache, _native_train=train_ds, _native_test=test_ds)


def _load_fashion_mnist(meta: dict, transform=None) -> "DagnamDataset":
    from torchvision import datasets, transforms
    from dagnam.data.dataset import DagnamDataset

    cache = _SYSTEM_CACHE_ROOT / "fashion_mnist"
    cache.mkdir(parents=True, exist_ok=True)

    base_transform = transform if transform is not None else transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_ds = datasets.FashionMNIST(root=str(cache), train=True, download=True, transform=base_transform)
    test_ds = datasets.FashionMNIST(root=str(cache), train=False, download=True, transform=base_transform)

    return DagnamDataset(meta, cache, _native_train=train_ds, _native_test=test_ds)


def _load_imdb(meta: dict, transform=None) -> "DagnamDataset":
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
        meta, cache,
        _native_train=(x_train, y_train),
        _native_test=(x_test, y_test),
    )


def _load_oxford_pets(meta: dict, transform=None) -> "DagnamDataset":
    from torchvision import datasets, transforms
    from dagnam.data.dataset import DagnamDataset

    cache = _SYSTEM_CACHE_ROOT / "oxford_pets"
    cache.mkdir(parents=True, exist_ok=True)

    base_transform = transform
    if base_transform is None:
        base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    try:
        train_ds = datasets.OxfordIIITPet(
            root=str(cache), split="trainval", download=True, transform=base_transform,
        )
        test_ds = datasets.OxfordIIITPet(
            root=str(cache), split="test", download=True, transform=base_transform,
        )
    except Exception:
        # Fallback: if torchvision doesn't have OxfordIIITPet, return file-based
        return DagnamDataset(meta, cache)

    return DagnamDataset(meta, cache, _native_train=train_ds, _native_test=test_ds)


def _load_speech_commands(meta: dict, transform=None) -> "DagnamDataset":
    from dagnam.data.dataset import DagnamDataset

    cache = _SYSTEM_CACHE_ROOT / "speech_commands"
    cache.mkdir(parents=True, exist_ok=True)

    try:
        import torchaudio
        train_ds = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(cache), download=True, subset="training",
        )
        test_ds = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(cache), download=True, subset="testing",
        )
        return DagnamDataset(meta, cache, _native_train=train_ds, _native_test=test_ds)
    except (ImportError, Exception):
        # torchaudio not installed or download failed — return file-based
        return DagnamDataset(meta, cache)


def _load_wikitext2(meta: dict, transform=None) -> "DagnamDataset":
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
            meta, cache,
            _native_train=list(train_iter),
            _native_test=list(test_iter),
        )
    except (ImportError, Exception):
        return DagnamDataset(meta, cache)


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Framework-specific native loaders — 16.72-bb / 16.82-bb
# ------------------------------------------------------------------

# Map canonical names → tensorflow_datasets identifiers. These are used by both
# _load_native_tf and _load_native_flax.
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


def resolve_system_dataset_tf(meta: dict) -> "DagnamDataset":
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

    train_ds = tfds.load(
        tfds_name, split="train", as_supervised=True, data_dir=str(cache)
    )
    test_ds = tfds.load(
        tfds_name, split="test", as_supervised=True, data_dir=str(cache)
    )

    return DagnamDataset(
        meta,
        cache,
        _native_train_tf=train_ds,
        _native_test_tf=test_ds,
    )


def resolve_system_dataset_flax(meta: dict) -> "DagnamDataset":
    """Load a system dataset as native FLAX batches via ``tensorflow_datasets``.

    Returns a list[FlaxBatch] for train and test splits. Falls back to the
    PyTorch native loader (converted in-memory) if ``tfds`` is not installed.

    Handles both image and text datasets: image samples are normalized to
    float32/255, text samples (bytes / strings / int sequences) are emitted
    without the image-specific normalization.
    """
    from dagnam.data.dataset import DagnamDataset

    tfds_name = _resolve_tfds_name(meta)
    if tfds_name is None:
        return resolve_system_dataset(meta)

    try:
        import tensorflow_datasets as tfds
        import numpy as np
        import jax.numpy as jnp
    except ImportError:
        return resolve_system_dataset(meta)

    from dagnam.data.loaders.flax_loader import FlaxBatch

    cache = _SYSTEM_CACHE_ROOT / tfds_name
    cache.mkdir(parents=True, exist_ok=True)

    def _encode_feature_batch(xs: list) -> "jnp.ndarray":
        """Convert a list of raw tfds samples into a JAX array.

        Images (uint8 arrays with 2+ spatial dims) are cast to float32 and
        scaled to [0, 1]. Text (bytes / strings) is encoded to integer code
        points per sample, padded to the longest sample in the batch. Other
        numeric inputs are cast to float32 without scaling.
        """
        first = xs[0]
        if isinstance(first, np.ndarray) and first.dtype == np.uint8 and first.ndim >= 2:
            return jnp.asarray(np.stack(xs).astype(np.float32) / 255.0)
        if isinstance(first, (bytes, bytearray, str)):
            # Emit raw byte sequences padded to the longest sample.
            encoded = []
            for item in xs:
                if isinstance(item, (bytes, bytearray)):
                    arr = np.frombuffer(bytes(item), dtype=np.uint8)
                else:
                    arr = np.frombuffer(item.encode("utf-8"), dtype=np.uint8)
                encoded.append(arr)
            max_len = max(len(a) for a in encoded)
            padded = np.zeros((len(encoded), max_len), dtype=np.int32)
            for i, a in enumerate(encoded):
                padded[i, : len(a)] = a
            return jnp.asarray(padded)
        if isinstance(first, np.ndarray):
            return jnp.asarray(np.stack(xs).astype(np.float32))
        # Fallback: let numpy/jax coerce.
        return jnp.asarray(np.asarray(xs))

    def _load_split(split: str, batch_size: int = 128) -> list:
        ds = tfds.load(tfds_name, split=split, as_supervised=True, data_dir=str(cache))
        batches = []
        xs, ys = [], []
        for x, lbl in tfds.as_numpy(ds):
            xs.append(x)
            ys.append(int(lbl))
            if len(xs) == batch_size:
                batches.append(
                    FlaxBatch(
                        features=_encode_feature_batch(xs),
                        labels=jnp.asarray(np.array(ys, dtype=np.int64)),
                    )
                )
                xs, ys = [], []
        if xs:
            batches.append(
                FlaxBatch(
                    features=_encode_feature_batch(xs),
                    labels=jnp.asarray(np.array(ys, dtype=np.int64)),
                )
            )
        return batches

    train_batches = _load_split("train")
    test_batches = _load_split("test")

    return DagnamDataset(
        meta,
        cache,
        _native_train_flax=train_batches,
        _native_test_flax=test_batches,
    )
