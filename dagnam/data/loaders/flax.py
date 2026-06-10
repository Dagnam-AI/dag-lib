"""Flax/JAX loader — converts tabular data into JAX arrays with split batching."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import random
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np
import numpy.typing as npt

from dagnam.data._polars_utils import factorize, numeric_columns
from dagnam.data.loaders.csv import detect_label_column
from dagnam.data.loaders.media import select_split_indices

if TYPE_CHECKING:
    import jax

    from dagnam.data.dataset._typing import DatasetMixinBase


class FlaxBatch(NamedTuple):
    """A single batch of features and labels as JAX arrays."""

    features: jax.Array
    labels: jax.Array


FeatureTransform = Callable[[npt.ArrayLike, object], npt.ArrayLike | tuple[npt.ArrayLike, object]]
BatchTransform = Callable[[FlaxBatch], FlaxBatch]
JaxArrayFactory = Callable[[npt.ArrayLike], "jax.Array"]

PairBatchTransform = Callable[["jax.Array", "jax.Array"], tuple["jax.Array", "jax.Array"]]


def build_flax_batches[SampleT](
    samples: Sequence[SampleT],
    batch_size: int,
    load_sample: Callable[[SampleT], tuple[npt.NDArray[np.float32], int]],
    *,
    shuffle: bool = False,
    seed: int = 42,
    batch_transform_fn: PairBatchTransform | None = None,
) -> list[FlaxBatch]:
    """Batch ``(feature_array, label)`` samples into a list of ``FlaxBatch``.

    Shared by the image-folder and audio-folder Flax loaders. ``load_sample``
    materializes one sample to ``(feature_ndarray, label_int)``; any per-sample
    transform belongs inside that closure. Optional seeded shuffle and a
    ``(features, labels) -> (features, labels)`` batch transform mirror the
    behavior the per-loader loops previously implemented inline.
    """
    import jax.numpy as jnp

    as_jax_array = cast("JaxArrayFactory", jnp.asarray)
    ordered = list(samples)
    if shuffle:
        random.Random(seed).shuffle(ordered)

    batches: list[FlaxBatch] = []
    for start in range(0, len(ordered), batch_size):
        chunk = ordered[start : start + batch_size]
        features: list[npt.NDArray[np.float32]] = []
        labels: list[int] = []
        for sample in chunk:
            feature, label = load_sample(sample)
            features.append(feature)
            labels.append(label)
        x = as_jax_array(np.stack(features))
        y = as_jax_array(np.array(labels, dtype=np.int64))
        batch = FlaxBatch(features=x, labels=y)
        if batch_transform_fn is not None:
            feat, lbl = batch_transform_fn(batch.features, batch.labels)
            batch = FlaxBatch(features=feat, labels=lbl)
        batches.append(batch)

    return batches


def create_flax_dataset(
    dagnam_ds: DatasetMixinBase,
    split: str,
    batch_size: int,
    shuffle: bool,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    column_roles: dict[str, str] | None = None,
    transform_fn: FeatureTransform | None = None,
    batch_transform_fn: BatchTransform | None = None,
) -> list[FlaxBatch]:
    """Create a list of FlaxBatch from a tabular dataset.

    Returns a list of (features, labels) NamedTuples as JAX arrays.
    Uses the same splitting logic as the PyTorch and TF loaders.
    ``column_roles`` overrides automatic label detection.
    """
    import jax.numpy as jnp

    as_jax_array = cast("JaxArrayFactory", jnp.array)
    df = dagnam_ds.to_polars()

    label_col = detect_label_column(df, dagnam_ds.feature_schema, column_roles=column_roles)

    # Label encoding
    label_series = df[label_col]
    if dagnam_ds.class_names:
        mapping: dict[object, int] = {name: idx for idx, name in enumerate(dagnam_ds.class_names)}
        labels = np.array([mapping[v] for v in label_series.to_list()], dtype=np.int64)
    else:
        labels = factorize(label_series)

    # Feature encoding
    feature_cols = [c for c in df.columns if c != label_col]
    numeric_cols = numeric_columns(df, feature_cols)
    features = df.select(numeric_cols).to_numpy().astype(np.float32)

    # Deterministic split
    split_indices = select_split_indices(
        df.height,
        split,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    if shuffle:
        random.Random(seed + 1).shuffle(split_indices)

    split_features = features[split_indices]
    split_labels = labels[split_indices]

    if transform_fn is not None:
        transformed_features: list[npt.ArrayLike] = []
        transformed_labels: list[object] = []
        for feature, label in zip(split_features, split_labels, strict=False):
            transformed = transform_fn(feature, label)
            if isinstance(transformed, tuple) and len(transformed) == 2:
                feature, label = transformed
            else:
                feature = transformed
            transformed_features.append(feature)
            transformed_labels.append(label)
        split_features = np.asarray(transformed_features)
        split_labels = np.asarray(transformed_labels)

    # Batch into list of FlaxBatch
    batches: list[FlaxBatch] = []
    for i in range(0, len(split_indices), batch_size):
        batch_f = as_jax_array(split_features[i : i + batch_size])
        batch_l = as_jax_array(cast("npt.ArrayLike", split_labels[i : i + batch_size]))
        batch = FlaxBatch(features=batch_f, labels=batch_l)
        if batch_transform_fn is not None:
            batch = batch_transform_fn(batch)
        batches.append(batch)

    return batches
