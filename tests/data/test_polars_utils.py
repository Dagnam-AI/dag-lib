"""Tests for the shared polars label/column helpers."""

from __future__ import annotations

import polars as pl
import pytest

from dagnam.data._polars_utils import (
    encode_label_series,
    factorize,
    materialize_feature_matrix,
)


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


class TestMaterializeFeatureMatrixTokenizeBinding:
    """The tokenize branch's rejection paths.

    A tokenize binding names the column it reads. Both ways that name can be
    wrong raise rather than silently falling through to the numeric path — a
    fallthrough would train on entirely different columns than the architecture
    declared, which is exactly the "declared intent vs actual data" mismatch the
    binding exists to prevent.
    """

    @staticmethod
    def _frame() -> pl.DataFrame:
        return pl.DataFrame({"review": ["good movie", "bad movie"], "score": [1.0, 0.0]})

    @staticmethod
    def _binding(column: object) -> dict[str, object]:
        transform: dict[str, object] = {
            "kind": "tokenize",
            "params": {"sequence_length": 4},
        }
        return {"input_column": column, "input_transform": transform}

    @pytest.mark.parametrize("missing", ["absent_column", None, 7])
    def test_a_column_that_is_not_in_the_frame_is_rejected(self, missing: object) -> None:
        with pytest.raises(ValueError, match="existing string input_column"):
            materialize_feature_matrix(self._frame(), ["review"], self._binding(missing))

    def test_a_real_column_that_is_not_an_input_role_is_rejected(self) -> None:
        # The column exists, but the roles assign it to something other than the
        # model's input — tokenizing it anyway would feed the model a column the
        # architecture never bound as input.
        with pytest.raises(ValueError, match="not assigned an input role"):
            materialize_feature_matrix(self._frame(), ["score"], self._binding("review"))

    def test_a_valid_binding_tokenizes_the_named_column(self) -> None:
        matrix = materialize_feature_matrix(self._frame(), ["review"], self._binding("review"))
        assert matrix.shape == (2, 4)

    @pytest.mark.parametrize("bad_length", [0, -5, True, "200", None])
    def test_a_non_positive_sequence_length_falls_back_to_the_default(
        self, bad_length: object
    ) -> None:
        # `True` is deliberately in the list: bool is an int subclass, so a naive
        # `isinstance(value, int) and value > 0` would accept it and produce a
        # one-column matrix.
        transform: dict[str, object] = {
            "kind": "tokenize",
            "params": {"sequence_length": bad_length},
        }
        binding: dict[str, object] = {"input_column": "review", "input_transform": transform}
        matrix = materialize_feature_matrix(self._frame(), ["review"], binding)
        assert matrix.shape == (2, 200)


class TestMaterializeFeatureMatrixNumericPath:
    def test_no_numeric_columns_and_no_transform_is_rejected(self) -> None:
        frame = pl.DataFrame({"label": ["a", "b"]})
        with pytest.raises(ValueError, match="No numeric feature columns"):
            materialize_feature_matrix(frame, ["label"], None)

    def test_numeric_columns_materialize_as_float32(self) -> None:
        frame = pl.DataFrame({"a": [1, 2], "b": [3.5, 4.5]})
        matrix = materialize_feature_matrix(frame, ["a", "b"], None)
        assert matrix.shape == (2, 2)
        assert matrix.dtype.name == "float32"
