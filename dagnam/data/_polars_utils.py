"""Polars helpers shared across dataset and loader modules."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
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
