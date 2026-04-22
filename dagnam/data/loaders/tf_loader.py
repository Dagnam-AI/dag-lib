"""TensorFlow loader — converts tabular data into tf.data.Dataset."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from dagnam.data.loaders.csv_loader import _detect_label_column

if TYPE_CHECKING:
    from dagnam.data.dataset import DagnamDataset


def create_tensorflow_dataset(
    dagnam_ds: "DagnamDataset",
    split: str,
    batch_size: int,
    shuffle: bool,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> "tf.data.Dataset":
    """Create a tf.data.Dataset from a tabular dataset.

    Uses the same label detection, encoding, and splitting logic as the
    PyTorch loader.
    """
    import tensorflow as tf

    df = dagnam_ds.to_pandas()

    label_col = _detect_label_column(df, dagnam_ds.feature_schema)

    # Label encoding — get numpy arrays directly
    if dagnam_ds.class_names:
        mapping = {name: idx for idx, name in enumerate(dagnam_ds.class_names)}
        labels = df[label_col].map(mapping).values.astype(np.int64)
    else:
        labels, _ = pd.factorize(df[label_col])
        labels = labels.astype(np.int64)

    # Feature encoding
    feature_cols = [c for c in df.columns if c != label_col]
    features = df[feature_cols].select_dtypes(include="number").values.astype(np.float32)

    # Deterministic split (same logic as csv_loader)
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

    split_features = features[split_indices]
    split_labels = labels[split_indices]

    ds = tf.data.Dataset.from_tensor_slices((split_features, split_labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(split_indices), seed=seed)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
