"""Flax/JAX loader — converts tabular data into JAX arrays with split batching."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd

from dagnam.data.loaders.csv import _detect_label_column

if TYPE_CHECKING:
    import jax

    from dagnam.data.dataset import DagnamDataset


class FlaxBatch(NamedTuple):
    """A single batch of features and labels as JAX arrays."""

    features: jax.Array
    labels: jax.Array


def create_flax_dataset(
    dagnam_ds: DagnamDataset,
    split: str,
    batch_size: int,
    shuffle: bool,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    column_roles: dict[str, str] | None = None,
    transform_fn=None,
    batch_transform_fn=None,
) -> list[FlaxBatch]:
    """Create a list of FlaxBatch from a tabular dataset.

    Returns a list of (features, labels) NamedTuples as JAX arrays.
    Uses the same splitting logic as the PyTorch and TF loaders.
    ``column_roles`` overrides automatic label detection.
    """
    import jax.numpy as jnp

    df = dagnam_ds.to_pandas()

    label_col = _detect_label_column(df, dagnam_ds.feature_schema, column_roles=column_roles)

    # Label encoding
    if dagnam_ds.class_names:
        mapping = {name: idx for idx, name in enumerate(dagnam_ds.class_names)}
        labels = df[label_col].map(mapping).values.astype(np.int64)
    else:
        labels, _ = pd.factorize(df[label_col])
        labels = labels.astype(np.int64)

    # Feature encoding
    feature_cols = [c for c in df.columns if c != label_col]
    features = df[feature_cols].select_dtypes(include="number").values.astype(np.float32)

    # Deterministic split
    n = len(df)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    split_map = {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }
    split_indices = split_map[split]

    if shuffle:
        random.Random(seed + 1).shuffle(split_indices)

    split_features = features[split_indices]
    split_labels = labels[split_indices]

    if transform_fn is not None:
        transformed_features = []
        transformed_labels = []
        for feature, label in zip(split_features, split_labels):
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
    batches = []
    for i in range(0, len(split_indices), batch_size):
        batch_f = jnp.array(split_features[i : i + batch_size])
        batch_l = jnp.array(split_labels[i : i + batch_size])
        batch = FlaxBatch(features=batch_f, labels=batch_l)
        if batch_transform_fn is not None:
            batch = batch_transform_fn(batch)
        batches.append(batch)

    return batches
