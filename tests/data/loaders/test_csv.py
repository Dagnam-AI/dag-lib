"""Unit tests for dagnam.loaders.csv_loader."""

from __future__ import annotations

from collections.abc import Sequence, Sized
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import polars as pl
import pytest
from tests.typing_helpers import PytestMonkeyPatch
from torch.utils.data import DataLoader
from typing_extensions import override

from dagnam._types import JsonObject, JsonValue
from dagnam.data.dataset import DagnamDataset
from dagnam.data.loaders.csv import (
    TabularDataset,
    TorchTensor as CsvTorchTensor,
    create_pytorch_loader,
    detect_label_column,
    encode_labels,
    split_by_roles,
)


class TensorLike(CsvTorchTensor, Protocol):
    dtype: object

    @property
    def shape(self) -> Sequence[int]: ...

    @override
    def __len__(self) -> int: ...

    @override
    def __getitem__(self, index: object) -> CsvTorchTensor: ...

    def tolist(self) -> list[int | float]: ...


class TorchTestModule(Protocol):
    long: object
    float32: object
    accelerator: object

    def randn(self, *size: int) -> TensorLike: ...

    def zeros(self, *size: int, dtype: object | None = None) -> TensorLike: ...

    def arange(self, end: int, *, dtype: object | None = None) -> TensorLike: ...

    def equal(self, input: object, other: object) -> bool: ...


def _torch() -> TorchTestModule:
    return cast("TorchTestModule", import_module("torch"))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_csv_dataset(
    tmp_path: Path,
    class_names: list[str] | None = None,
    feature_schema: JsonObject | None = None,
    n: int = 100,
) -> DagnamDataset:
    """Create a minimal DagnamDataset backed by a CSV file."""
    data_dir = tmp_path / "ds"
    data_dir.mkdir()

    # Write a simple CSV
    rows: list[dict[str, float | str]] = []
    labels = ["cat", "dog"] if class_names is None else class_names[:2]
    for i in range(n):
        rows.append({"feat1": float(i), "feat2": float(i * 2), "label": labels[i % len(labels)]})
    df = pl.DataFrame(rows)
    df.write_csv(data_dir / "data.csv")

    class_names_json: JsonValue = list(class_names) if class_names is not None else None
    meta: JsonObject = {
        "id": "test-id",
        "name": "test-ds",
        "format": "csv",
        "dataset_type": "tabular",
        "num_samples": n,
        "num_classes": len(labels),
        "feature_schema": feature_schema,
        "class_names": class_names_json,
    }
    return DagnamDataset(meta, data_dir)


def _make_regression_csv_dataset(tmp_path: Path, n: int = 20) -> DagnamDataset:
    data_dir = tmp_path / "regression"
    data_dir.mkdir()
    pl.DataFrame(
        {
            "feature": [float(i) for i in range(n)],
            "target": [float(i) + 0.25 for i in range(n)],
        }
    ).write_csv(data_dir / "data.csv")
    return DagnamDataset(
        {
            "id": "regression-id",
            "name": "regression",
            "format": "csv",
            "dataset_type": "tabular",
            "num_samples": n,
            "num_classes": 0,
            "feature_schema": None,
            "class_names": None,
        },
        data_dir,
    )


def _make_text_csv_dataset(tmp_path: Path, n: int = 40) -> DagnamDataset:
    data_dir = tmp_path / "text"
    data_dir.mkdir()
    pl.DataFrame(
        {
            "text": [f"deterministic short review {index}" for index in range(n)],
            "label": ["positive", "negative"] * (n // 2),
        }
    ).write_csv(data_dir / "text.csv")
    return DagnamDataset(
        {
            "id": "text-id",
            "name": "text",
            "format": "csv",
            "dataset_type": "text",
            "num_samples": n,
            "num_classes": 2,
            "feature_schema": {
                "columns": [
                    {"name": "text", "type": "categorical"},
                    {"name": "label", "type": "categorical"},
                ]
            },
            "class_names": None,
        },
        data_dir,
    )


# ------------------------------------------------------------------
# _TabularDataset
# ------------------------------------------------------------------


class TestTabularDataset:
    def test_len(self) -> None:
        torch = _torch()
        ds = TabularDataset(torch.randn(10, 3), torch.zeros(10, dtype=torch.long))
        assert len(ds) == 10

    def test_getitem(self) -> None:
        torch = _torch()
        feats = torch.randn(5, 2)
        labels = torch.arange(5, dtype=torch.long)
        ds = TabularDataset(feats, labels)
        f, l = ds[2]
        assert torch.equal(f, feats[2])
        assert l == labels[2]


# ------------------------------------------------------------------
# detect_label_column
# ------------------------------------------------------------------


class TestDetectLabelColumn:
    def test_categorical_from_schema(self) -> None:
        df = pl.DataFrame({"a": [1], "b": ["x"], "c": [2]})
        schema: JsonObject = {
            "columns": [
                {"name": "a", "type": "numeric"},
                {"name": "b", "type": "categorical"},
                {"name": "c", "type": "numeric"},
            ]
        }
        assert detect_label_column(df, schema) == "b"

    def test_first_categorical_wins(self) -> None:
        df = pl.DataFrame({"a": ["x"], "b": ["y"], "c": [1]})
        schema: JsonObject = {
            "columns": [
                {"name": "a", "type": "categorical"},
                {"name": "b", "type": "categorical"},
                {"name": "c", "type": "numeric"},
            ]
        }
        assert detect_label_column(df, schema) == "a"

    def test_fallback_last_column(self) -> None:
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        assert detect_label_column(df, None) == "c"

    def test_schema_no_categorical_falls_back(self) -> None:
        df = pl.DataFrame({"a": [1], "b": [2]})
        schema: JsonObject = {
            "columns": [
                {"name": "a", "type": "numeric"},
                {"name": "b", "type": "numeric"},
            ]
        }
        assert detect_label_column(df, schema) == "b"

    def test_skips_non_dict_column_entries(self) -> None:
        # A non-dict entry in the columns list is ignored; detection continues
        # to the first valid categorical column.
        df = pl.DataFrame({"a": [1], "b": ["x"]})
        schema: JsonObject = {
            "columns": [
                "not-a-dict",
                {"name": "b", "type": "categorical"},
            ]
        }
        assert detect_label_column(df, schema) == "b"

    def test_column_roles_target_takes_priority(self) -> None:
        df = pl.DataFrame({"a": [1], "b": ["x"], "c": [2]})
        assert detect_label_column(df, None, column_roles={"a": "target"}) == "a"

    def test_column_roles_label_role_recognized(self) -> None:
        df = pl.DataFrame({"a": [1], "b": ["x"]})
        assert detect_label_column(df, None, column_roles={"b": "label"}) == "b"

    def test_column_roles_target_not_in_dataframe_is_skipped(self) -> None:
        # Role matches target/label but the column is absent from the DataFrame
        # (branch 194->193): the loop continues and the priority-1 path does not
        # return, falling through to the schema/fallback logic (branch 193->197).
        df = pl.DataFrame({"a": [1], "b": [2]})
        schema: JsonObject = {
            "columns": [
                {"name": "a", "type": "categorical"},
            ]
        }
        assert detect_label_column(df, schema, column_roles={"missing": "target"}) == "a"

    def test_column_roles_non_target_role_skipped(self) -> None:
        # Role present in df but not target/label (branch 194->193 false leg via
        # role mismatch), then exhausts the loop (193->197) to the fallback.
        df = pl.DataFrame({"a": [1], "b": [2]})
        assert detect_label_column(df, None, column_roles={"a": "feature"}) == "b"

    def test_schema_columns_not_a_list_falls_back(self) -> None:
        # feature_schema has "columns" but it is not a list (branch 199->210):
        # detection skips straight to the last-column fallback.
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        schema: JsonObject = {"columns": "not-a-list"}
        assert detect_label_column(df, schema) == "c"

    def test_categorical_column_with_non_str_name_skipped(self) -> None:
        # A categorical column whose "name" is not a str (branch 206->200):
        # the loop continues to the next valid categorical column.
        df = pl.DataFrame({"a": [1], "b": ["x"]})
        schema: JsonObject = {
            "columns": [
                {"name": 123, "type": "categorical"},
                {"name": "b", "type": "categorical"},
            ]
        }
        assert detect_label_column(df, schema) == "b"


# ------------------------------------------------------------------
# split_by_roles
# ------------------------------------------------------------------


class TestSplitByRoles:
    def test_target_then_feature_continues_loop(self) -> None:
        # A target column followed by a feature column exercises the
        # elif-then-continue leg (branch 165->160) and the feature append.
        df = pl.DataFrame({"y": ["a"], "x1": [1], "x2": [2]})
        label_col, feature_cols = split_by_roles(
            df, {"y": "target", "x1": "feature", "x2": "ignore"}
        )
        assert label_col == "y"
        assert feature_cols == ["x1"]

    def test_raises_when_no_target_role(self) -> None:
        # No column carries a target role, so split_by_roles raises (line 170,
        # branch 169->170).
        df = pl.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="does not specify object target column"):
            split_by_roles(df, {"a": "feature", "b": "feature"})


# ------------------------------------------------------------------
# _encode_labels
# ------------------------------------------------------------------


class TestEncodeLabels:
    def test_with_class_names(self) -> None:
        torch = _torch()
        series = pl.Series(["dog", "cat", "dog"])
        result = cast("TensorLike", encode_labels(series, ["cat", "dog"]))
        assert result.dtype == torch.long
        assert result.tolist() == [1, 0, 1]

    def test_without_class_names(self) -> None:
        torch = _torch()
        series = pl.Series(["a", "b", "a", "c"])
        result = cast("TensorLike", encode_labels(series, None))
        assert result.dtype == torch.long
        # factorize assigns codes in order of appearance
        assert result[0] == result[2]  # both "a"
        assert len(set(result.tolist())) == 3

    def test_integer_label_column_matches_string_class_names(self) -> None:
        # Real failing condition: an Int64 label column with string class_names.
        # Previously raised bare KeyError while to_arrays() succeeded.
        torch = _torch()
        series = pl.Series([0, 1, 0, 2])
        result = cast("TensorLike", encode_labels(series, ["0", "1", "2"]))
        assert result.dtype == torch.long
        assert result.tolist() == [0, 1, 0, 2]

    def test_unknown_label_value_raises_valueerror(self) -> None:
        series = pl.Series(["cat", "unknown"])
        with pytest.raises(ValueError, match="not in class_names"):
            encode_labels(series, ["cat", "dog"])


# ------------------------------------------------------------------
# create_pytorch_loader — integration
# ------------------------------------------------------------------


class TestCreatePytorchLoader:
    def test_text_binding_tokenizes_user_csv(self, tmp_path: Path) -> None:
        torch = _torch()
        loader = create_pytorch_loader(
            _make_text_csv_dataset(tmp_path),
            split="train",
            batch_size=8,
            num_workers=0,
            shuffle=False,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
            column_roles={"text": "text_input", "label": "target"},
            binding={
                "input_column": "text",
                "target_column": "label",
                "input_transform": {
                    "kind": "tokenize",
                    "params": {"vocab_size": 256, "sequence_length": 32},
                },
                "target_transform": {"kind": "class_index", "params": {"dtype": "long"}},
            },
        )

        features, labels = cast("tuple[TensorLike, TensorLike]", next(iter(loader)))
        assert tuple(features.shape) == (8, 32)
        assert features.dtype == torch.long
        assert labels.dtype == torch.long
        assert max(cast("TensorLike", features[0]).tolist()) < 256

    def test_numeric_binding_preserves_float_column_target(self, tmp_path: Path) -> None:
        torch = _torch()
        ds = _make_regression_csv_dataset(tmp_path)
        loader = create_pytorch_loader(
            ds,
            split="train",
            batch_size=4,
            num_workers=0,
            shuffle=False,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
            column_roles={"feature": "feature", "target": "target"},
            binding={"target_transform": {"kind": "numeric", "params": {"dtype": "float"}}},
        )

        _features, targets = cast("tuple[TensorLike, TensorLike]", next(iter(loader)))
        assert targets.dtype == torch.float32
        assert tuple(targets.shape) == (4, 1)

    def test_train_loader(self, tmp_path: Path) -> None:
        ds = _make_csv_dataset(tmp_path, class_names=["cat", "dog"])
        loader = create_pytorch_loader(
            ds,
            split="train",
            batch_size=16,
            num_workers=0,
            shuffle=True,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )
        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 16
        # drop_last=True for train
        assert loader.drop_last is True

    def test_column_roles_path(self, tmp_path: Path) -> None:
        # Passing column_roles routes through split_by_roles instead of the
        # heuristic detector (line 85).
        ds = _make_csv_dataset(tmp_path, class_names=["cat", "dog"])
        loader = create_pytorch_loader(
            ds,
            split="train",
            batch_size=8,
            num_workers=0,
            shuffle=False,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
            column_roles={"feat1": "feature", "feat2": "feature", "label": "target"},
        )
        # feat1 + feat2 are the feature columns.
        batch_feats, _labels = cast("tuple[TensorLike, TensorLike]", next(iter(loader)))
        assert batch_feats.shape[1] == 2

    def test_val_loader_no_drop_last(self, tmp_path: Path) -> None:
        ds = _make_csv_dataset(tmp_path, class_names=["cat", "dog"])
        loader = create_pytorch_loader(
            ds,
            split="val",
            batch_size=8,
            num_workers=0,
            shuffle=False,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )
        assert loader.drop_last is False

    def test_test_loader_no_drop_last(self, tmp_path: Path) -> None:
        ds = _make_csv_dataset(tmp_path, class_names=["cat", "dog"])
        loader = create_pytorch_loader(
            ds,
            split="test",
            batch_size=8,
            num_workers=0,
            shuffle=False,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )
        assert loader.drop_last is False

    def test_pin_memory_disabled_without_accelerator(
        self, monkeypatch: PytestMonkeyPatch, tmp_path: Path
    ) -> None:
        torch = _torch()

        def no_accelerator() -> bool:
            return False

        monkeypatch.setattr(torch.accelerator, "is_available", no_accelerator)
        ds = _make_csv_dataset(tmp_path, class_names=["cat", "dog"])

        loader = create_pytorch_loader(
            ds,
            split="train",
            batch_size=8,
            num_workers=0,
            shuffle=True,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )

        assert loader.pin_memory is False

    def test_deterministic_splits(self, tmp_path: Path) -> None:
        ds = _make_csv_dataset(tmp_path, class_names=["cat", "dog"])
        loader1 = create_pytorch_loader(
            ds,
            split="train",
            batch_size=8,
            num_workers=0,
            shuffle=False,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=99,
        )
        loader2 = create_pytorch_loader(
            ds,
            split="train",
            batch_size=8,
            num_workers=0,
            shuffle=False,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=99,
        )
        torch = _torch()
        for batch1, batch2 in zip(loader1, loader2, strict=False):
            f1, l1 = cast("tuple[TensorLike, TensorLike]", batch1)
            f2, l2 = cast("tuple[TensorLike, TensorLike]", batch2)
            assert torch.equal(f1, f2)
            assert torch.equal(l1, l2)

    def test_split_sizes(self, tmp_path: Path) -> None:
        n = 100
        ds = _make_csv_dataset(tmp_path, class_names=["cat", "dog"], n=n)
        val_ratio, test_ratio = 0.1, 0.1
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        n_train = n - n_val - n_test

        train_loader = create_pytorch_loader(
            ds,
            split="train",
            batch_size=n,
            num_workers=0,
            shuffle=False,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=42,
        )
        val_loader = create_pytorch_loader(
            ds,
            split="val",
            batch_size=n,
            num_workers=0,
            shuffle=False,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=42,
        )
        test_loader = create_pytorch_loader(
            ds,
            split="test",
            batch_size=n,
            num_workers=0,
            shuffle=False,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=42,
        )
        # drop_last=True for train, so use dataset length directly
        assert len(cast("Sized", train_loader.dataset)) == n_train
        assert len(cast("Sized", val_loader.dataset)) == n_val
        assert len(cast("Sized", test_loader.dataset)) == n_test

    def test_factorize_fallback(self, tmp_path: Path) -> None:
        """Labels encoded via first-seen-order factorize when class_names is None."""
        ds = _make_csv_dataset(tmp_path, class_names=None)
        loader = create_pytorch_loader(
            ds,
            split="train",
            batch_size=8,
            num_workers=0,
            shuffle=False,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )
        torch = _torch()
        _batch_feats, batch_labels = cast("tuple[TensorLike, TensorLike]", next(iter(loader)))
        assert batch_labels.dtype == torch.long

    def test_schema_label_detection(self, tmp_path: Path) -> None:
        schema: JsonObject = {
            "columns": [
                {"name": "feat1", "type": "numeric"},
                {"name": "label", "type": "categorical"},
                {"name": "feat2", "type": "numeric"},
            ]
        }
        ds = _make_csv_dataset(tmp_path, class_names=["cat", "dog"], feature_schema=schema)
        loader = create_pytorch_loader(
            ds,
            split="train",
            batch_size=8,
            num_workers=0,
            shuffle=False,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )
        # Should have 2 feature columns (feat1, feat2)
        batch_feats, _labels = cast("tuple[TensorLike, TensorLike]", next(iter(loader)))
        assert batch_feats.shape[1] == 2

    def test_feature_tensor_dtype(self, tmp_path: Path) -> None:
        ds = _make_csv_dataset(tmp_path, class_names=["cat", "dog"])
        loader = create_pytorch_loader(
            ds,
            split="train",
            batch_size=8,
            num_workers=0,
            shuffle=False,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )
        torch = _torch()
        batch_feats, _labels = cast("tuple[TensorLike, TensorLike]", next(iter(loader)))
        assert batch_feats.dtype == torch.float32
