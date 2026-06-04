"""TensorFlow loader — converts tabular data into tf.data.Dataset."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
import random
from typing import TYPE_CHECKING, cast

import numpy as np

from dagnam._types import TensorflowDataset, TensorflowModule
from dagnam.data._polars_utils import factorize, numeric_columns
from dagnam.data.loaders.csv import detect_label_column

if TYPE_CHECKING:
    from dagnam.data.dataset._typing import DatasetMixinBase


def create_tensorflow_dataset(
    dagnam_ds: DatasetMixinBase,
    split: str,
    batch_size: int,
    shuffle: bool,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    column_roles: dict[str, str] | None = None,
    map_fn: Callable[..., object] | None = None,
    batch_map_fn: Callable[..., object] | None = None,
) -> TensorflowDataset:
    """Create a tf.data.Dataset from a tabular dataset.

    Uses the same label detection, encoding, and splitting logic as the
    PyTorch loader. ``column_roles`` overrides automatic label detection.
    """
    tf_data = cast("TensorflowModule", import_module("tensorflow")).data
    df = dagnam_ds.to_polars()

    label_col = detect_label_column(df, dagnam_ds.feature_schema, column_roles=column_roles)

    # Label encoding — get numpy arrays directly
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

    # Deterministic split (same logic as csv_loader)
    n = df.height
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

    ds = tf_data.Dataset.from_tensor_slices((split_features, split_labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(split_indices), seed=seed)

    if map_fn is not None:
        ds = ds.map(map_fn)

    ds = ds.batch(batch_size)

    if batch_map_fn is not None:
        ds = ds.map(batch_map_fn)

    ds = ds.prefetch(tf_data.AUTOTUNE)

    return ds
