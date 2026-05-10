"""Flax/JAX system dataset loaders backed by tensorflow_datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dagnam.data.loaders.system.common import _SYSTEM_CACHE_ROOT
from dagnam.data.loaders.system.registry import resolve_system_dataset
from dagnam.data.loaders.system.tensorflow_datasets import _resolve_tfds_name

if TYPE_CHECKING:
    from dagnam.data.dataset import DagnamDataset


def resolve_system_dataset_flax(meta: dict) -> DagnamDataset:
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
        import jax.numpy as jnp
        import numpy as np
        import tensorflow_datasets as tfds
    except ImportError:
        return resolve_system_dataset(meta)

    from dagnam.data.loaders.flax import FlaxBatch

    cache = _SYSTEM_CACHE_ROOT / tfds_name
    cache.mkdir(parents=True, exist_ok=True)

    def _encode_feature_batch(xs: list) -> jnp.ndarray:
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
