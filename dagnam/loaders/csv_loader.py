"""CSV/TSV loader — converts tabular data into PyTorch DataLoaders."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pandas as pd
import torch
import torch.utils.data

if TYPE_CHECKING:
    from dagnam.dataset import DagnamDataset


class _TabularDataset(torch.utils.data.Dataset):
    """Internal PyTorch Dataset wrapping feature and label tensors."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features  # float32, shape (n_samples, n_features)
        self.labels = labels  # long, shape (n_samples,)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def create_pytorch_loader(
    dagnam_ds: "DagnamDataset",
    split: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> torch.utils.data.DataLoader:
    """Create a PyTorch DataLoader from a CSV/TSV dataset.

    Label detection, encoding, deterministic splitting, and DataLoader
    configuration are all handled here.  Invalid split names are already
    validated by ``DagnamDataset.to_pytorch_loader()``.
    """
    df = dagnam_ds.to_pandas()

    # ---- label column detection ----
    label_col = _detect_label_column(df, dagnam_ds.feature_schema)

    # ---- label encoding ----
    labels = _encode_labels(df[label_col], dagnam_ds.class_names)

    # ---- feature encoding (numeric columns only, excluding label) ----
    feature_cols = [c for c in df.columns if c != label_col]
    features_df = df[feature_cols].select_dtypes(include="number")
    features = torch.tensor(features_df.values, dtype=torch.float32)

    # ---- deterministic split ----
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

    # ---- build dataset & loader ----
    ds = _TabularDataset(features[split_indices], labels[split_indices])

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _detect_label_column(df: pd.DataFrame, feature_schema: dict | None) -> str:
    """Return the label column name.

    Priority:
    1. First column with type ``"categorical"`` in *feature_schema*.
    2. Last DataFrame column as fallback.
    """
    if feature_schema and "columns" in feature_schema:
        for col_info in feature_schema["columns"]:
            if col_info.get("type") == "categorical":
                return col_info["name"]

    # Fallback: last column
    return df.columns[-1]


def _encode_labels(
    series: pd.Series, class_names: list[str] | None
) -> torch.Tensor:
    """Encode a label series into a ``long`` tensor.

    If *class_names* is provided, maps each value to its index in the list.
    Otherwise falls back to ``pd.factorize()``.
    """
    if class_names:
        mapping = {name: idx for idx, name in enumerate(class_names)}
        encoded = series.map(mapping).values
    else:
        encoded, _ = pd.factorize(series)

    return torch.tensor(encoded, dtype=torch.long)
