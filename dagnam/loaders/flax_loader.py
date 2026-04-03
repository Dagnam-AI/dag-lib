"""Flax/JAX loader — converts tabular data into JAX arrays with split batching."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from dagnam.dataset import DagnamDataset


class FlaxBatch(NamedTuple):
    """A single batch of features and labels as JAX arrays."""

    features: "jax.Array"
    labels: "jax.Array"


def create_flax_dataset(
    dagnam_ds: "DagnamDataset",
    split: str,
    batch_size: int,
    shuffle: bool,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> list[FlaxBatch]:
    """Create a list of FlaxBatch from a tabular dataset.

    Returns a list of (features, labels) NamedTuples as JAX arrays.
    Uses the same splitting logic as the PyTorch and TF loaders.
    """
    import jax.numpy as jnp

    df = dagnam_ds.to_pandas()

    from dagnam.loaders.csv_loader import _detect_label_column

    label_col = _detect_label_column(df, dagnam_ds.feature_schema)

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

    # Batch into list of FlaxBatch
    batches = []
    for i in range(0, len(split_indices), batch_size):
        batch_f = jnp.array(split_features[i : i + batch_size])
        batch_l = jnp.array(split_labels[i : i + batch_size])
        batches.append(FlaxBatch(features=batch_f, labels=batch_l))

    return batches
