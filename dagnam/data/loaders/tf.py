"""TensorFlow loader — converts tabular data into tf.data.Dataset."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING, cast

from dagnam._types import TensorflowDataset, TensorflowModule
from dagnam.data._polars_utils import encode_target_series, materialize_feature_matrix
from dagnam.data.loaders.csv import detect_label_column, split_by_roles
from dagnam.data.loaders.media import select_split_indices

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
    binding: dict[str, object] | None = None,
    map_fn: Callable[..., object] | None = None,
    batch_map_fn: Callable[..., object] | None = None,
) -> TensorflowDataset:
    """Create a tf.data.Dataset from a tabular dataset.

    Uses the same label detection, encoding, and splitting logic as the
    PyTorch loader. ``column_roles`` overrides automatic label detection.
    """
    tf_data = cast("TensorflowModule", import_module("tensorflow")).data
    df = dagnam_ds.to_polars()

    if column_roles is not None:
        label_col, feature_cols = split_by_roles(df, column_roles)
    else:
        label_col = detect_label_column(df, dagnam_ds.feature_schema)
        feature_cols = [c for c in df.columns if c != label_col]

    # Label encoding — get numpy arrays directly
    label_series = df[label_col]
    labels = encode_target_series(label_series, dagnam_ds.class_names, binding)

    # Feature encoding
    features = materialize_feature_matrix(df, feature_cols, binding)

    # Deterministic split (same logic as csv_loader)
    split_indices = select_split_indices(
        df.height,
        split,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        membership=dagnam_ds.split_membership or None,
    )

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
