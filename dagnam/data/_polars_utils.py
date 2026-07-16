"""Polars helpers shared across dataset and loader modules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import zlib

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


def _positive_int(value: object, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return default


def tokenize_text(
    texts: Sequence[object],
    maxlen: int = 200,
    num_words: int = 20000,
) -> npt.NDArray[np.int32]:
    """Hash-tokenize strings deterministically with zero reserved for padding."""
    vocab = max(2, num_words)
    result = np.zeros((len(texts), maxlen), dtype=np.int32)
    for row_index, text in enumerate(texts):
        tokens = str(text).split()[:maxlen]
        for token_index, token in enumerate(tokens):
            result[row_index, token_index] = zlib.crc32(token.encode("utf-8")) % (vocab - 1) + 1
    return result


def materialize_feature_matrix(
    df: pl.DataFrame,
    candidates: list[str],
    binding: dict[str, object] | None = None,
) -> npt.NDArray[np.generic]:
    """Materialize bound tabular inputs into one framework-neutral matrix.

    Text bindings select the declared input column and use the dataset's
    deterministic hash tokenizer. Other tabular inputs retain the legacy
    numeric-column behavior.
    """
    raw_transform = binding.get("input_transform") if binding is not None else None
    if isinstance(raw_transform, dict) and raw_transform.get("kind") == "tokenize":
        input_column = binding.get("input_column") if binding is not None else None
        if not isinstance(input_column, str) or input_column not in df.columns:
            raise ValueError("tokenize binding requires an existing string input_column")
        if input_column not in candidates:
            raise ValueError(
                f"tokenize binding input_column {input_column!r} is not assigned an input role"
            )
        raw_params = raw_transform.get("params")
        params = raw_params if isinstance(raw_params, dict) else {}
        sequence_length = _positive_int(params.get("sequence_length"), 200)
        vocab_size = _positive_int(params.get("vocab_size"), 20000)
        return tokenize_text(
            df[input_column].to_list(),
            maxlen=sequence_length,
            num_words=vocab_size,
        )

    numeric = numeric_columns(df, candidates)
    if not numeric:
        raise ValueError(
            "No numeric feature columns were selected and the binding has no supported "
            "input transform"
        )
    return np.asarray(df.select(numeric).to_numpy(), dtype=np.float32)


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


def encode_target_series(
    series: pl.Series,
    class_names: list[str] | None,
    binding: dict[str, object] | None = None,
) -> npt.NDArray[np.int64] | npt.NDArray[np.float32]:
    """Materialize a bound target according to its resolver transform.

    Classification remains the default for backward compatibility. A numeric
    target transform is the explicit regression contract and produces
    ``[samples, 1]`` float32 values to match a one-unit regression head.
    """
    raw_transform = binding.get("target_transform") if binding is not None else None
    if isinstance(raw_transform, dict) and raw_transform.get("kind") == "numeric":
        return np.asarray(series.to_numpy(), dtype=np.float32).reshape(-1, 1)
    return encode_label_series(series, class_names)
