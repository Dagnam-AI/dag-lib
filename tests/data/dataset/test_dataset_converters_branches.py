"""Characterization tests for base/polars/hooks converter branches.

These pin current behavior for metadata validation, label-column detection,
to_arrays tuple handling, polars caching, and the _TransformDataset hooks
that don't require torch/tensorflow/jax to exercise.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import polars as pl
import pytest

from dagnam._types import JsonObject, NativeSplit
from dagnam.data.dataset import DagnamDataset
from dagnam.data.dataset.hooks import _TransformDataset


def _base_meta(**overrides: object) -> JsonObject:
    meta: JsonObject = {
        "id": "b1",
        "name": "branch",
        "format": "csv",
        "dataset_type": "tabular",
        "num_samples": 2,
        "num_classes": 2,
        "class_names": [],
    }
    meta.update(cast("JsonObject", overrides))
    return meta


# ---------------------------------------------------------------- metadata validation


def test_required_str_rejects_non_string(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="must be a string"):
        DagnamDataset(_base_meta(name=123), tmp_path)


def test_required_int_rejects_non_int(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        DagnamDataset(_base_meta(num_samples="lots"), tmp_path)


def test_optional_str_list_rejects_non_string_list(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="class_names"):
        DagnamDataset(_base_meta(class_names=[1, 2, 3]), tmp_path)


# ---------------------------------------------------------------- raw_data getter


def test_raw_data_getter_returns_underlying_payload(tmp_path: Path) -> None:
    ds = DagnamDataset(_base_meta(), tmp_path)
    payload: list[object] = [1, 2, 3]
    ds.raw_data = payload
    assert ds.raw_data is payload


# ---------------------------------------------------------------- to_arrays tuple sizes


def test_to_arrays_treats_non_pair_tuple_as_feature_without_label(tmp_path: Path) -> None:
    ds = DagnamDataset(_base_meta(format="custom"), tmp_path)
    # A 3-element tuple is not a (feature, label) pair → whole tuple is the feature.
    ds.raw_data = [(1, 2, 3)]
    features, labels = ds.to_arrays(split="train")
    assert labels is None
    assert features.tolist() == [[1, 2, 3]]


# ---------------------------------------------------------------- detect_label_column


def test_detect_label_column_falls_back_to_last_when_no_schema(tmp_path: Path) -> None:
    ds = DagnamDataset(_base_meta(), tmp_path)
    df = pl.DataFrame({"a": [1], "b": [2], "target": [3]})
    assert ds.detect_label_column(df) == "target"


def test_detect_label_column_columns_not_a_list_falls_back(tmp_path: Path) -> None:
    ds = DagnamDataset(
        _base_meta(feature_schema={"columns": "not-a-list"}),
        tmp_path,
    )
    df = pl.DataFrame({"a": [1], "last": [2]})
    assert ds.detect_label_column(df) == "last"


def test_detect_label_column_skips_non_dict_and_non_categorical(tmp_path: Path) -> None:
    ds = DagnamDataset(
        _base_meta(
            feature_schema={
                "columns": [
                    "not-a-dict",  # line 309: continue
                    {"name": "x", "type": "numeric"},  # 313->307: not categorical
                    {"type": "categorical"},  # categorical but name not str
                ]
            }
        ),
        tmp_path,
    )
    df = pl.DataFrame({"x": [1], "y": [2]})
    # No usable categorical column name → fall through to last column.
    assert ds.detect_label_column(df) == "y"


def test_detect_label_column_uses_categorical_schema_name(tmp_path: Path) -> None:
    ds = DagnamDataset(
        _base_meta(
            feature_schema={
                "columns": [
                    {"name": "x", "type": "numeric"},
                    {"name": "kind", "type": "categorical"},
                ]
            }
        ),
        tmp_path,
    )
    df = pl.DataFrame({"x": [1], "kind": [2], "extra": [3]})
    assert ds.detect_label_column(df) == "kind"


# ---------------------------------------------------------------- polars caching/json


def test_to_polars_cached_non_dataframe_raises(tmp_path: Path) -> None:
    ds = DagnamDataset(_base_meta(), tmp_path)
    ds._data = {"train": [1, 2]}  # cached payload that is not a polars frame
    with pytest.raises(TypeError, match="not a polars DataFrame"):
        ds.to_polars()


def test_to_polars_cached_dataframe_returned(tmp_path: Path) -> None:
    ds = DagnamDataset(_base_meta(), tmp_path)
    frame = pl.DataFrame({"a": [1]})
    ds._data = frame
    assert ds.to_polars() is frame


def test_to_polars_json_branch(tmp_path: Path) -> None:
    (tmp_path / "data.json").write_text('[{"a": 1}, {"a": 2}]', encoding="utf-8")
    ds = DagnamDataset(_base_meta(format="json"), tmp_path)
    df = ds.to_polars()
    assert df.height == 2


def test_to_polars_jsonl_branch(tmp_path: Path) -> None:
    (tmp_path / "data.jsonl").write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    ds = DagnamDataset(_base_meta(format="jsonl"), tmp_path)
    df = ds.to_polars()
    assert df.height == 2


# ---------------------------------------------------------------- _TransformDataset hooks


def test_transform_dataset_short_tuple_without_transform_returns_tuple() -> None:
    """len(tuple) < 2 and transform is None → original tuple returned (line 139)."""

    class _OneTuple:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, _idx: int) -> object:
            return (42,)

    wrapped = _TransformDataset(cast("Sequence[object]", _OneTuple()))
    assert wrapped[0] == (42,)


def test_transform_dataset_pair_with_only_target_transform() -> None:
    """transform is None (skip 143->145) but target_transform runs."""

    class _Pair:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, _idx: int) -> object:
            return ("feat", 5)

    def _plus_one(value: object) -> object:
        return cast("int", value) + 1

    wrapped = _TransformDataset(
        cast("Sequence[object]", _Pair()),
        target_transform=_plus_one,
    )
    assert wrapped[0] == ("feat", 6)


# ---------------------------------------------------------------- native_* getters


def test_native_train_getter_returns_attached_split(tmp_path: Path) -> None:
    ds = DagnamDataset(_base_meta(format="native"), tmp_path)
    payload: NativeSplit = ([1, 2], [0, 1])
    ds.native_train = payload
    assert ds.native_train == payload


def test_native_test_getter_returns_attached_split(tmp_path: Path) -> None:
    ds = DagnamDataset(_base_meta(format="native"), tmp_path)
    payload: NativeSplit = ([3, 4], [1, 0])
    ds.native_test = payload
    assert ds.native_test == payload
