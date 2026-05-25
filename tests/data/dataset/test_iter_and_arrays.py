"""Coverage for DagnamDataset.iter_samples / to_arrays / label helpers."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import pytest

from dagnam._types import JsonObject, NativeSplit
from dagnam.data.dataset import DagnamDataset


def _meta(fmt: str = "csv", **overrides: object) -> JsonObject:
    base: dict[str, object] = {
        "id": "ds-x",
        "name": "x",
        "format": fmt,
        "dataset_type": "tabular",
        "num_samples": 0,
        "num_classes": 0,
    }
    base.update(overrides)
    return cast("JsonObject", base)


def _native_split(features: object, labels: object) -> NativeSplit:
    return cast("NativeSplit", (features, labels))


class TestIterSamples:
    def test_data_dict_with_split(self, tmp_path: Path) -> None:
        ds = DagnamDataset(_meta(), tmp_path)
        ds.raw_data = {"train": [(1, 0), (2, 1)], "test": [(3, 1)]}
        assert list(ds.iter_samples(split="train")) == [(1, 0), (2, 1)]
        assert list(ds.iter_samples(split="test")) == [(3, 1)]

    def test_data_list_path(self, tmp_path: Path) -> None:
        ds = DagnamDataset(_meta(), tmp_path)
        ds.raw_data = [(1, 0), (2, 1), (3, 0)]
        assert list(ds.iter_samples()) == [(1, 0), (2, 1), (3, 0)]

    def test_native_tuple_train(self, tmp_path: Path) -> None:
        ds = DagnamDataset(
            _meta(),
            tmp_path,
            _native_train=_native_split(np.array([10, 20]), np.array([0, 1])),
        )
        out = list(ds.iter_samples(split="train"))
        assert out == [(10, 0), (20, 1)]

    def test_native_tuple_test(self, tmp_path: Path) -> None:
        ds = DagnamDataset(
            _meta(),
            tmp_path,
            _native_test=_native_split(np.array([7]), np.array([1])),
        )
        out = list(ds.iter_samples(split="test"))
        assert out == [(7, 1)]

    def test_native_indexable_non_tuple(self, tmp_path: Path) -> None:
        # Simulate a native dataset that is iterable by index but not a
        # (features, labels) tuple — e.g. a list of records.
        class IndexableDS:
            def __init__(self, items: Sequence[object]) -> None:
                self._items = items

            def __len__(self) -> int:
                return len(self._items)

            def __getitem__(self, index: int) -> object:
                return self._items[index]

        ds = DagnamDataset(
            _meta(),
            tmp_path,
            _native_train=IndexableDS([{"x": 1}, {"x": 2}]),
        )
        out = list(ds.iter_samples(split="train"))
        assert out == [{"x": 1}, {"x": 2}]

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        ds = DagnamDataset(_meta(fmt="parquet"), tmp_path)
        with pytest.raises(ValueError, match="Raw sample iteration is not available"):
            list(ds.iter_samples())

    def test_tabular_csv_path(self, tmp_path: Path) -> None:
        # Uses _iter_tabular_file_samples for csv format
        (tmp_path / "data.csv").write_text("x,label\n1,a\n2,b\n3,a\n4,b\n")
        meta = _meta(
            fmt="csv",
            feature_schema={
                "columns": [
                    {"name": "x", "type": "numeric"},
                    {"name": "label", "type": "categorical"},
                ]
            },
            class_names=["a", "b"],
        )
        ds = DagnamDataset(meta, tmp_path)
        samples = cast(
            "list[tuple[list[object], int]]",
            list(ds.iter_samples(split="train", val_ratio=0.25, test_ratio=0.25)),
        )
        assert len(samples) == 2
        for feat, lbl in samples:
            assert isinstance(feat, list)
            assert lbl in (0, 1)

    def test_tabular_semicolon_csv_path(self, tmp_path: Path) -> None:
        (tmp_path / "data.csv").write_text(
            "Username; Identifier;One-time password\nbooker12;9012;12se74\ngrey07;2070;04ap67\n"
        )
        meta = _meta(
            fmt="csv",
            feature_schema={
                "columns": [
                    {"name": "Username", "type": "categorical"},
                    {"name": " Identifier", "type": "numeric"},
                    {"name": "One-time password", "type": "categorical"},
                ]
            },
        )
        ds = DagnamDataset(meta, tmp_path)

        feats, labels = ds.to_arrays(val_ratio=0, test_ratio=0)

        assert feats.shape == (2, 1)
        assert labels is not None
        assert labels.shape == (2,)


class TestToArrays:
    def test_with_labels(self, tmp_path: Path) -> None:
        ds = DagnamDataset(_meta(), tmp_path)
        ds.raw_data = [(1, 0), (2, 1), (3, 0)]
        feats, labels = ds.to_arrays()
        assert feats.tolist() == [1, 2, 3]
        assert labels is not None
        assert labels.tolist() == [0, 1, 0]

    def test_without_labels(self, tmp_path: Path) -> None:
        ds = DagnamDataset(_meta(), tmp_path)
        ds.raw_data = [1, 2, 3]  # bare items (no label)
        feats, labels = ds.to_arrays()
        assert feats.tolist() == [1, 2, 3]
        assert labels is None


class TestLabelHelpers:
    def testdetect_label_column_from_schema(self, tmp_path: Path) -> None:
        meta = _meta(
            feature_schema={
                "columns": [
                    {"name": "x", "type": "numeric"},
                    {"name": "y", "type": "numeric"},
                    {"name": "kind", "type": "categorical"},
                ]
            }
        )
        ds = DagnamDataset(meta, tmp_path)
        df = pl.DataFrame({"x": [1], "y": [2], "kind": ["a"]})
        assert ds.detect_label_column(df) == "kind"

    def testdetect_label_column_fallback_last(self, tmp_path: Path) -> None:
        ds = DagnamDataset(_meta(), tmp_path)
        df = pl.DataFrame({"x": [1], "y": [2]})
        assert ds.detect_label_column(df) == "y"

    def test_encode_labels_factorize_when_no_class_names(self, tmp_path: Path) -> None:
        ds = DagnamDataset(_meta(), tmp_path)
        out = ds.encode_label_values(pl.Series(["x", "y", "x", "z"]))
        # factorize assigns ints in first-seen order
        assert out == [0, 1, 0, 2]

    def test_encode_labels_with_class_names(self, tmp_path: Path) -> None:
        ds = DagnamDataset(_meta(class_names=["x", "y", "z"]), tmp_path)
        out = ds.encode_label_values(pl.Series(["y", "x", "z"]))
        assert out == [1, 0, 2]


class TestSplitIndices:
    def test_split_sizes_and_disjoint(self) -> None:
        n = 100
        train = DagnamDataset.split_indices(n, "train", 0.1, 0.1, seed=0)
        val = DagnamDataset.split_indices(n, "val", 0.1, 0.1, seed=0)
        test = DagnamDataset.split_indices(n, "test", 0.1, 0.1, seed=0)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10
        assert set(train).isdisjoint(val)
        assert set(train).isdisjoint(test)
        assert set(val).isdisjoint(test)


class TestFindDataFile:
    def test_skips_meta_json(self, tmp_path: Path) -> None:
        (tmp_path / "meta.json").write_text(json.dumps({"k": "v"}))
        (tmp_path / "data.json").write_text(json.dumps([{"a": 1}]))
        ds = DagnamDataset(_meta(fmt="json"), tmp_path)
        assert ds.find_data_file().name == "data.json"
