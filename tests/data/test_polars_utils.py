"""Tests for the shared polars label/column helpers."""

from __future__ import annotations

import polars as pl
import pytest

from dagnam.data._polars_utils import encode_label_series, factorize


class TestEncodeLabelSeries:
    def test_integer_column_maps_by_string_identity(self) -> None:
        # An integer label column (polars Int64) with STRING class_names must
        # encode by string identity — the single canonical mapping shared by
        # to_arrays() and every framework loader — not raise KeyError.
        series = pl.Series([0, 1, 0, 2])
        codes = encode_label_series(series, ["0", "1", "2"])
        assert codes.tolist() == [0, 1, 0, 2]

    def test_string_column_maps_to_class_index(self) -> None:
        series = pl.Series(["dog", "cat", "dog"])
        codes = encode_label_series(series, ["cat", "dog"])
        assert codes.tolist() == [1, 0, 1]

    def test_unknown_value_raises_valueerror_naming_value(self) -> None:
        series = pl.Series(["cat", "unknown"])
        with pytest.raises(ValueError, match="not in class_names"):
            encode_label_series(series, ["cat", "dog"])

    def test_without_class_names_falls_back_to_factorize(self) -> None:
        series = pl.Series(["a", "b", "a", "c"])
        codes = encode_label_series(series, None)
        assert codes.tolist() == factorize(series).tolist()
        assert codes[0] == codes[2]
