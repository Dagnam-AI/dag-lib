"""Flax/JAX system dataset loaders backed by tensorflow_datasets."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, SupportsInt, cast

import numpy as np
import numpy.typing as npt

from dagnam._types import JsonObject
from dagnam.data.loaders.system.common import SYSTEM_CACHE_ROOT
from dagnam.data.loaders.system.registry import resolve_system_dataset
from dagnam.data.loaders.system.tensorflow_datasets import resolve_tfds_name

if TYPE_CHECKING:
    import jax

    from dagnam.data.dataset import DagnamDataset
    from dagnam.data.loaders.flax import FlaxBatch

TfdsSample = tuple[object, object]
FeatureSample = npt.ArrayLike | bytes | bytearray | str
JaxArrayFactory = Callable[[npt.ArrayLike], "jax.Array"]


class TfdsModule(Protocol):
    def load(self, name: str, *, split: str, as_supervised: bool, data_dir: str) -> object: ...

    def as_numpy(self, dataset: object) -> Iterable[TfdsSample]: ...


def resolve_system_dataset_flax(meta: JsonObject) -> DagnamDataset:
    """Load a system dataset as native FLAX batches via ``tensorflow_datasets``.

    Returns a list[FlaxBatch] for train and test splits. Falls back to the
    PyTorch native loader (converted in-memory) if ``tfds`` is not installed.

    Handles both image and text datasets: image samples are normalized to
    float32/255, text samples (bytes / strings / int sequences) are emitted
    without the image-specific normalization.
    """
    from dagnam.data.dataset import DagnamDataset

    tfds_name = resolve_tfds_name(meta)
    if tfds_name is None:
        return resolve_system_dataset(meta)

    try:
        import jax.numpy as jnp
        tfds = cast(TfdsModule, import_module("tensorflow_datasets"))
    except ImportError:
        return resolve_system_dataset(meta)

    from dagnam.data.loaders.flax import FlaxBatch

    cache = SYSTEM_CACHE_ROOT / tfds_name
    cache.mkdir(parents=True, exist_ok=True)

    as_jax_array = cast(JaxArrayFactory, getattr(jnp, "asarray"))

    def _encode_feature_batch(xs: list[FeatureSample]) -> "jax.Array":
        """Convert a list of raw tfds samples into a JAX array.

        Images (uint8 arrays with 2+ spatial dims) are cast to float32 and
        scaled to [0, 1]. Text (bytes / strings) is encoded to integer code
        points per sample, padded to the longest sample in the batch. Other
        numeric inputs are cast to float32 without scaling.
        """
        first = xs[0]
        if isinstance(first, np.ndarray) and first.dtype == np.uint8 and first.ndim >= 2:
            image_batch = np.stack([cast(npt.ArrayLike, item) for item in xs]).astype(np.float32)
            return as_jax_array(image_batch / 255.0)
        if isinstance(first, (bytes, bytearray, str)):
            # Emit raw byte sequences padded to the longest sample.
            encoded: list[npt.NDArray[np.uint8]] = []
            for item in xs:
                if isinstance(item, (bytes, bytearray)):
                    arr = np.frombuffer(bytes(item), dtype=np.uint8)
                elif isinstance(item, str):
                    arr = np.frombuffer(item.encode("utf-8"), dtype=np.uint8)
                else:
                    raise TypeError("Expected bytes or strings for text system dataset samples")
                encoded.append(arr)
            max_len = max(len(a) for a in encoded)
            padded = np.zeros((len(encoded), max_len), dtype=np.int32)
            for i, a in enumerate(encoded):
                padded[i, : len(a)] = a
            return as_jax_array(padded)
        if isinstance(first, np.ndarray):
            numeric_batch = np.stack([cast(npt.ArrayLike, item) for item in xs]).astype(np.float32)
            return as_jax_array(numeric_batch)
        # Fallback: let numpy/jax coerce.
        return as_jax_array(np.asarray(xs))

    def _load_split(split: str, batch_size: int = 128) -> list["FlaxBatch"]:
        ds = tfds.load(tfds_name, split=split, as_supervised=True, data_dir=str(cache))
        batches: list[FlaxBatch] = []
        xs: list[FeatureSample] = []
        ys: list[int] = []
        for x, lbl in tfds.as_numpy(ds):
            xs.append(cast(FeatureSample, x))
            if not isinstance(lbl, SupportsInt):
                raise TypeError("Expected integer-compatible tfds labels")
            ys.append(int(lbl))
            if len(xs) == batch_size:
                batches.append(
                    FlaxBatch(
                        features=_encode_feature_batch(xs),
                        labels=as_jax_array(np.array(ys, dtype=np.int64)),
                    )
                )
                xs, ys = [], []
        if xs:
            batches.append(
                FlaxBatch(
                    features=_encode_feature_batch(xs),
                    labels=as_jax_array(np.array(ys, dtype=np.int64)),
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
