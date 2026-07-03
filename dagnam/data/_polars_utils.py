"""Polars helpers shared across dataset and loader modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import polars as pl


def factorize(series: pl.Series) -> npt.NDArray[np.int64]:
    """Encode a Series to integer codes in first-seen order.

    Equivalent to ``pandas.factorize(series)[0]``: the i-th element of the
    returned array is the position of ``series[i]`` in the ordered list of
    unique values, where order is the first occurrence in ``series``.
    """
    uniques: list[object] = series.unique(maintain_order=True).to_list()
    mapping: dict[object, int] = {v: i for i, v in enumerate(uniques)}
    return np.array([mapping[v] for v in series.to_list()], dtype=np.int64)


def numeric_columns(df: pl.DataFrame, candidates: list[str]) -> list[str]:
    """Return the subset of ``candidates`` whose columns have numeric dtype."""
    return [c for c in candidates if df.schema[c].is_numeric()]


def encode_label_series(series: pl.Series, class_names: list[str] | None) -> npt.NDArray[np.int64]:
    """Encode a label Series to ``int64`` codes — the one canonical mapping.

    With ``class_names``, each value maps to its index by STRING identity, so an
    integer label column (polars ``Int64`` holding ``0/1/2``) matches
    ``class_names`` ``["0", "1", "2"]`` exactly as ``DagnamDataset.to_arrays``
    does. Every framework loader (PyTorch/TF/Flax) and ``to_arrays`` route
    through here so they can never diverge. Without ``class_names``, falls back
    to first-seen-order factorization.

    Raises:
        ValueError: a label value is absent from ``class_names`` (message names
            the offending value and the known classes).
    """
    if not class_names:
        return factorize(series)
    mapping: dict[str, int] = {name: idx for idx, name in enumerate(class_names)}
    try:
        return np.array([mapping[str(v)] for v in series.to_list()], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(
            f"Label value {exc.args[0]!r} is not in class_names {class_names}"
        ) from exc
