"""CSV/TSV loader — converts tabular data into PyTorch DataLoaders."""

from __future__ import annotations

from importlib import import_module
import random
from typing import TYPE_CHECKING, Protocol, cast

from dagnam._types import JsonObject
from dagnam.data._polars_utils import factorize, numeric_columns
from dagnam.data.loaders.torch_utils import should_pin_memory

if TYPE_CHECKING:
    import polars as pl
    from torch.utils.data import DataLoader, Dataset

    from dagnam.data.dataset._typing import DatasetMixinBase


class TorchTensor(Protocol):
    """Torch tensor surface used by the tabular loader."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: object) -> TorchTensor: ...


class TorchModule(Protocol):
    """Torch factory and dtype surface used by the tabular loader."""

    float32: object
    long: object

    def tensor(self, data: object, *, dtype: object) -> TorchTensor: ...


def _load_torch() -> TorchModule:
    return cast("TorchModule", import_module("torch"))


class _TabularDataset:
    """Internal PyTorch Dataset wrapping feature and label tensors."""

    def __init__(self, features: TorchTensor, labels: TorchTensor) -> None:
        self.features = features  # float32, shape (n_samples, n_features)
        self.labels = labels  # long, shape (n_samples,)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[TorchTensor, TorchTensor]:
        return self.features[idx], self.labels[idx]


TabularDataset = _TabularDataset


def create_pytorch_loader(
    dagnam_ds: DatasetMixinBase,
    split: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    column_roles: dict[str, str] | None = None,
) -> DataLoader[object]:
    """Create a PyTorch DataLoader from a CSV/TSV dataset.

    Label detection, encoding, deterministic splitting, and DataLoader
    configuration are all handled here.  Invalid split names are already
    validated by ``DagnamDataset.to_pytorch_loader()``.

    When *column_roles* is provided, it is used to separate feature and
    target columns instead of the heuristic-based detection.  Columns
    with role ``"ignore"`` are excluded entirely.
    """
    from torch.utils.data import DataLoader

    torch = _load_torch()
    df = dagnam_ds.to_polars()

    if column_roles is not None:
        label_col, feature_cols = split_by_roles(df, column_roles)
    else:
        # ---- legacy heuristic path ----
        label_col = detect_label_column(df, dagnam_ds.feature_schema)
        feature_cols = [c for c in df.columns if c != label_col]

    # ---- label encoding ----
    labels = _encode_labels(df[label_col], dagnam_ds.class_names)

    # ---- feature encoding (numeric columns only) ----
    numeric_cols = numeric_columns(df, feature_cols)
    features = torch.tensor(df.select(numeric_cols).to_numpy(), dtype=torch.float32)

    # ---- deterministic split ----
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

    # ---- build dataset & loader ----
    ds = _TabularDataset(features[split_indices], labels[split_indices])

    loader = DataLoader(
        cast("Dataset[object]", ds),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=should_pin_memory(),
        drop_last=(split == "train"),
    )
    return loader


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

# Role sets used by split_by_roles to classify columns.
FEATURE_ROLES = frozenset(
    {
        "feature",
        "text_input",
        "image_input",
        "audio_input",
        "prompt",
        "anchor",
    }
)
TARGET_ROLES = frozenset({"target", "text_target", "completion"})


def split_by_roles(
    df: pl.DataFrame,
    column_roles: dict[str, str],
) -> tuple[str, list[str]]:
    """Separate feature and target columns using an explicit role mapping.

    Returns ``(label_col, feature_cols)`` where *feature_cols* preserves
    the original DataFrame column order.  Columns with role ``"ignore"``
    (or object role not in the feature/target sets) are excluded from both
    lists.
    """
    feature_cols: list[str] = []
    target_cols: list[str] = []

    for col_obj in df.columns:
        col = str(col_obj)
        role = column_roles.get(col)
        if role in FEATURE_ROLES:
            feature_cols.append(col)
        elif role in TARGET_ROLES:
            target_cols.append(col)
        # else: ignore / unknown role — skip

    if not target_cols:
        raise ValueError(
            "column_roles does not specify object target column "
            "(expected a column with role 'target', 'text_target', or 'completion')"
        )

    # Use the first target column as the label column.
    label_col = target_cols[0]
    return label_col, feature_cols


def detect_label_column(
    df: pl.DataFrame,
    feature_schema: JsonObject | None,
    column_roles: dict[str, str] | None = None,
) -> str:
    """Return the label column name.

    Priority:
    1. Column explicitly marked ``"target"`` / ``"label"`` in *column_roles*.
    2. First column with type ``"categorical"`` in *feature_schema*.
    3. Last DataFrame column as fallback.
    """
    if column_roles:
        for col, role in column_roles.items():
            if role in ("target", "label") and col in df.columns:
                return col

    if feature_schema and "columns" in feature_schema:
        columns = feature_schema["columns"]
        if isinstance(columns, list):
            for col_info in columns:
                if not isinstance(col_info, dict):
                    continue
                if col_info.get("type") != "categorical":
                    continue
                name = col_info.get("name")
                if isinstance(name, str):
                    return name

    # Fallback: last column
    return df.columns[-1]


def _encode_labels(series: pl.Series, class_names: list[str] | None) -> TorchTensor:
    """Encode a label series into a ``long`` tensor.

    If *class_names* is provided, maps each value to its index in the list.
    Otherwise falls back to first-seen-order factorization.
    """
    import numpy as np

    if class_names:
        mapping: dict[object, int] = {name: idx for idx, name in enumerate(class_names)}
        encoded = np.array([mapping[v] for v in series.to_list()], dtype=np.int64)
    else:
        encoded = factorize(series)

    torch = _load_torch()
    return torch.tensor(encoded, dtype=torch.long)


def encode_labels(series: pl.Series, class_names: list[str] | None) -> TorchTensor:
    return _encode_labels(series, class_names)
