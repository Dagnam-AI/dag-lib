"""Unit tests for DagnamDataset class."""

import json
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from tests.typing_helpers import JsonObject

from dagnam.data.dataset import DagnamDataset


@pytest.fixture
def data_dir(tmp_path: Path):
    """Create a temporary data directory."""
    d = tmp_path / "dataset_cache"
    d.mkdir()
    return d


@pytest.fixture
def meta():
    """Minimal metadata dict for constructing a DagnamDataset."""
    return {
        "id": "ds-001",
        "name": "Test Dataset",
        "format": "csv",
        "dataset_type": "tabular",
        "num_samples": 6,
        "num_classes": 3,
        "feature_schema": {
            "columns": [
                {"name": "x", "type": "numeric"},
                {"name": "label", "type": "categorical"},
            ]
        },
        "class_names": ["a", "b", "c"],
    }


# ------------------------------------------------------------------
# __init__ tests
# ------------------------------------------------------------------


class TestInit:
    def test_attributes_from_meta(self, meta: JsonObject, data_dir: Path) -> None:
        ds = DagnamDataset(meta, data_dir)
        assert ds.id == "ds-001"
        assert ds.name == "Test Dataset"
        assert ds.format == "csv"
        assert ds.dataset_type == "tabular"
        assert ds.num_samples == 6
        assert ds.num_classes == 3
        assert ds.class_names == ["a", "b", "c"]
        assert ds.feature_schema is not None
        assert ds._data_dir == data_dir
        assert ds._data is None  # lazy

    def test_optional_fields_default_to_none(self, data_dir: Path) -> None:
        minimal: JsonObject = {
            "id": "ds-002",
            "name": "Minimal",
            "format": "json",
            "dataset_type": "text",
            "num_samples": 0,
            "num_classes": 0,
        }
        ds = DagnamDataset(minimal, data_dir)
        assert ds.feature_schema is None
        assert ds.class_names is None


# ------------------------------------------------------------------
# info property tests
# ------------------------------------------------------------------


class TestInfo:
    def test_info_keys(self, meta: JsonObject, data_dir: Path) -> None:
        ds = DagnamDataset(meta, data_dir)
        info = ds.info
        expected_keys = {
            "id",
            "name",
            "format",
            "type",
            "samples",
            "classes",
            "class_names",
            "schema",
        }
        assert set(info.keys()) == expected_keys

    def test_info_values(self, meta: JsonObject, data_dir: Path) -> None:
        ds = DagnamDataset(meta, data_dir)
        info = ds.info
        assert info["id"] == "ds-001"
        assert info["name"] == "Test Dataset"
        assert info["format"] == "csv"
        assert info["type"] == "tabular"
        assert info["samples"] == 6
        assert info["classes"] == 3
        assert info["class_names"] == ["a", "b", "c"]
        assert info["schema"] is not None


# ------------------------------------------------------------------
# to_polars() tests
# ------------------------------------------------------------------


class TestToPolars:
    def test_csv_loading(self, meta: JsonObject, data_dir: Path) -> None:
        csv_file = data_dir / "data.csv"
        csv_file.write_text("x,y\n1,2\n3,4\n")
        ds = DagnamDataset(meta, data_dir)
        df = ds.to_polars()
        assert isinstance(df, pl.DataFrame)
        assert list(df.columns) == ["x", "y"]
        assert df.height == 2

    def test_tsv_loading(self, meta: JsonObject, data_dir: Path) -> None:
        meta["format"] = "tsv"
        tsv_file = data_dir / "data.tsv"
        tsv_file.write_text("x\ty\n1\t2\n3\t4\n")
        ds = DagnamDataset(meta, data_dir)
        df = ds.to_polars()
        assert list(df.columns) == ["x", "y"]
        assert df.height == 2

    def test_json_loading(self, meta: JsonObject, data_dir: Path) -> None:
        meta["format"] = "json"
        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        json_file = data_dir / "data.json"
        json_file.write_text(json.dumps(records))
        ds = DagnamDataset(meta, data_dir)
        df = ds.to_polars()
        assert df.height == 2
        assert "a" in df.columns

    def test_jsonl_loading(self, meta: JsonObject, data_dir: Path) -> None:
        meta["format"] = "jsonl"
        lines = '{"a":1,"b":2}\n{"a":3,"b":4}\n'
        jsonl_file = data_dir / "data.jsonl"
        jsonl_file.write_text(lines)
        ds = DagnamDataset(meta, data_dir)
        df = ds.to_polars()
        assert df.height == 2

    def test_caching_after_first_call(self, meta: JsonObject, data_dir: Path) -> None:
        csv_file = data_dir / "data.csv"
        csv_file.write_text("x\n1\n2\n")
        ds = DagnamDataset(meta, data_dir)
        df1 = ds.to_polars()
        df2 = ds.to_polars()
        assert df1 is df2  # same object, not re-parsed

    def test_unsupported_format_raises_valueerror(self, meta: JsonObject, data_dir: Path) -> None:
        meta["format"] = "parquet"
        ds = DagnamDataset(meta, data_dir)
        with pytest.raises(ValueError, match="Cannot load format 'parquet' as DataFrame"):
            ds.to_polars()

    def test_missing_file_raises_file_not_found(self, meta: JsonObject, data_dir: Path) -> None:
        # No CSV file in data_dir
        ds = DagnamDataset(meta, data_dir)
        with pytest.raises(FileNotFoundError, match="No data file matching"):
            ds.to_polars()

    def test_json_excludes_meta_json(self, meta: JsonObject, data_dir: Path) -> None:
        meta["format"] = "json"
        # Only meta.json present — should raise FileNotFoundError
        meta_file = data_dir / "meta.json"
        meta_file.write_text(json.dumps({"id": "test"}))
        ds = DagnamDataset(meta, data_dir)
        with pytest.raises(FileNotFoundError):
            ds.to_polars()

    def test_json_finds_non_meta_json(self, meta: JsonObject, data_dir: Path) -> None:
        meta["format"] = "json"
        # meta.json should be skipped, data.json should be found
        (data_dir / "meta.json").write_text(json.dumps({"id": "test"}))
        (data_dir / "data.json").write_text(json.dumps([{"a": 1}]))
        ds = DagnamDataset(meta, data_dir)
        df = ds.to_polars()
        assert df.height == 1


# ------------------------------------------------------------------
# to_pytorch_loader() tests
# ------------------------------------------------------------------


class TestToPytorchLoader:
    def test_importerror_when_torch_missing(self, meta: JsonObject, data_dir: Path) -> None:
        ds = DagnamDataset(meta, data_dir)
        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(ImportError, match="PyTorch is required"):
                ds.to_pytorch_loader()

    def test_invalid_split_raises_valueerror(self, meta: JsonObject, data_dir: Path) -> None:
        ds = DagnamDataset(meta, data_dir)
        with pytest.raises(ValueError, match="Unknown split: foo"):
            ds.to_pytorch_loader(split="foo")

    def test_unsupported_format_raises_valueerror(self, meta: JsonObject, data_dir: Path) -> None:
        meta["format"] = "parquet"
        ds = DagnamDataset(meta, data_dir)
        with pytest.raises(ValueError, match="Unsupported format for PyTorch loader: parquet"):
            ds.to_pytorch_loader()


# ------------------------------------------------------------------
# to_tensorflow_dataset() tests
# ------------------------------------------------------------------


class TestToTensorflowDataset:
    def test_importerror_when_tensorflow_missing(self, meta: JsonObject, data_dir: Path) -> None:
        ds = DagnamDataset(meta, data_dir)
        with patch.dict("sys.modules", {"tensorflow": None}):
            with pytest.raises(ImportError, match="TensorFlow is required"):
                ds.to_tensorflow_dataset()

    def test_invalid_split_raises_valueerror(self, meta: JsonObject, data_dir: Path) -> None:
        ds = DagnamDataset(meta, data_dir)
        with pytest.raises(ValueError, match="Unknown split: foo"):
            ds.to_tensorflow_dataset(split="foo")

    def test_unsupported_format_raises_valueerror(self, meta: JsonObject, data_dir: Path) -> None:
        meta["format"] = "parquet"
        ds = DagnamDataset(meta, data_dir)
        with pytest.raises(ValueError, match="Unsupported format for TensorFlow dataset: parquet"):
            ds.to_tensorflow_dataset()


# ------------------------------------------------------------------
# to_flax_dataset() tests
# ------------------------------------------------------------------


class TestToFlaxDataset:
    def test_importerror_when_jax_missing(self, meta: JsonObject, data_dir: Path) -> None:
        ds = DagnamDataset(meta, data_dir)
        with patch.dict("sys.modules", {"jax": None}):
            with pytest.raises(ImportError, match="JAX is required"):
                ds.to_flax_dataset()

    def test_invalid_split_raises_valueerror(self, meta: JsonObject, data_dir: Path) -> None:
        ds = DagnamDataset(meta, data_dir)
        with pytest.raises(ValueError, match="Unknown split: foo"):
            ds.to_flax_dataset(split="foo")

    def test_unsupported_format_raises_valueerror(self, meta: JsonObject, data_dir: Path) -> None:
        meta["format"] = "parquet"
        ds = DagnamDataset(meta, data_dir)
        with pytest.raises(ValueError, match="Unsupported format for Flax dataset: parquet"):
            ds.to_flax_dataset()


def test_tokenize_text_is_deterministic_and_fixed_length() -> None:
    # G078: hash-tokenization is deterministic (crc32, not Python's salted hash),
    # pads/truncates to maxlen, reserves 0 for padding, and produces integers.
    import numpy as np

    out = DagnamDataset._tokenize_text(["hello world foo", "bar"], maxlen=4, num_words=50)
    assert out.shape == (2, 4)
    assert out.dtype == np.int32
    assert (out >= 0).all()
    assert (out < 50).all()
    assert out[1, 1:].tolist() == [0, 0, 0]  # "bar" -> one token then padding
    # deterministic across calls (stable hash)
    again = DagnamDataset._tokenize_text(["hello world foo", "bar"], maxlen=4, num_words=50)
    assert out.tolist() == again.tolist()


def test_tokenize_text_truncates_to_maxlen() -> None:
    out = DagnamDataset._tokenize_text(["a b c d e f"], maxlen=3, num_words=100)
    assert out.shape == (1, 3)
    assert (out[0] > 0).all()  # all three slots filled, none padding


def test_is_text_features_classifies_arrays() -> None:
    import numpy as np

    assert DagnamDataset._is_text_features(np.array(["a", "b"]))  # unicode dtype
    assert DagnamDataset._is_text_features(np.array(["a", "b"], dtype=object))  # object-of-str
    assert not DagnamDataset._is_text_features(np.array([[1, 2], [3, 4]]))  # numeric
    assert not DagnamDataset._is_text_features(
        np.array([np.array([1, 2]), np.array([3])], dtype=object)
    )  # object-of-int-lists
    assert not DagnamDataset._is_text_features(np.array([], dtype=object))  # empty object array


def test_batches_need_padding_classifies_all_branches() -> None:
    import numpy as np

    f = DagnamDataset._batches_need_padding
    assert not f([])  # empty -> nothing to concatenate
    assert not f([np.ones((2, 3), dtype=np.int64)])  # single batch
    assert not f(
        [np.ones((2, 3), dtype=np.int64), np.ones((1, 3), dtype=np.int64)]
    )  # consistent trailing dims
    assert f([np.array([[1, 2], [1]], dtype=object)])  # ragged/object dtype
    assert f(
        [np.ones((2, 3), dtype=np.int64), np.ones((1, 5), dtype=np.int64)]
    )  # rectangular but different sequence length
    assert not f([np.ones(3, dtype=np.int64), np.ones(4, dtype=np.int64)])  # 1-D: no trailing dims
