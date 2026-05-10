"""Unit tests for dagnam.loaders.csv_loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from dagnam.data.dataset import DagnamDataset
from dagnam.data.loaders.csv import (
    _detect_label_column,
    _encode_labels,
    _TabularDataset,
    create_pytorch_loader,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_csv_dataset(tmp_path: Path, class_names=None, feature_schema=None, n=100):
    """Create a minimal DagnamDataset backed by a CSV file."""
    data_dir = tmp_path / "ds"
    data_dir.mkdir()

    # Write a simple CSV
    rows = []
    labels = ["cat", "dog"] if class_names is None else class_names[:2]
    for i in range(n):
        rows.append({"feat1": float(i), "feat2": float(i * 2), "label": labels[i % len(labels)]})
    df = pd.DataFrame(rows)
    df.to_csv(data_dir / "data.csv", index=False)

    meta = {
        "id": "test-id",
        "name": "test-ds",
        "format": "csv",
        "dataset_type": "tabular",
        "num_samples": n,
        "num_classes": len(labels),
        "feature_schema": feature_schema,
        "class_names": class_names,
    }
    return DagnamDataset(meta, data_dir)


# ------------------------------------------------------------------
# _TabularDataset
# ------------------------------------------------------------------


class TestTabularDataset:
    def test_len(self):
        ds = _TabularDataset(torch.randn(10, 3), torch.zeros(10, dtype=torch.long))
        assert len(ds) == 10

    def test_getitem(self):
        feats = torch.randn(5, 2)
        labels = torch.arange(5, dtype=torch.long)
        ds = _TabularDataset(feats, labels)
        f, l = ds[2]
        assert torch.equal(f, feats[2])
        assert l == labels[2]


# ------------------------------------------------------------------
# _detect_label_column
# ------------------------------------------------------------------


class TestDetectLabelColumn:
    def test_categorical_from_schema(self):
        df = pd.DataFrame({"a": [1], "b": ["x"], "c": [2]})
        schema = {
            "columns": [
                {"name": "a", "type": "numeric"},
                {"name": "b", "type": "categorical"},
                {"name": "c", "type": "numeric"},
            ]
        }
        assert _detect_label_column(df, schema) == "b"

    def test_first_categorical_wins(self):
        df = pd.DataFrame({"a": ["x"], "b": ["y"], "c": [1]})
        schema = {
            "columns": [
                {"name": "a", "type": "categorical"},
                {"name": "b", "type": "categorical"},
                {"name": "c", "type": "numeric"},
            ]
        }
        assert _detect_label_column(df, schema) == "a"

    def test_fallback_last_column(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        assert _detect_label_column(df, None) == "c"

    def test_schema_no_categorical_falls_back(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        schema = {
            "columns": [
                {"name": "a", "type": "numeric"},
                {"name": "b", "type": "numeric"},
            ]
        }
        assert _detect_label_column(df, schema) == "b"


# ------------------------------------------------------------------
# _encode_labels
# ------------------------------------------------------------------


class TestEncodeLabels:
    def test_with_class_names(self):
        series = pd.Series(["dog", "cat", "dog"])
        result = _encode_labels(series, ["cat", "dog"])
        assert result.dtype == torch.long
        assert result.tolist() == [1, 0, 1]

    def test_without_class_names(self):
        series = pd.Series(["a", "b", "a", "c"])
        result = _encode_labels(series, None)
        assert result.dtype == torch.long
        # factorize assigns codes in order of appearance
        assert result[0] == result[2]  # both "a"
        assert len(set(result.tolist())) == 3


# ------------------------------------------------------------------
# create_pytorch_loader — integration
# ------------------------------------------------------------------


class TestCreatePytorchLoader:
    def test_train_loader(self, tmp_path):
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
        assert isinstance(loader, torch.utils.data.DataLoader)
        assert loader.batch_size == 16
        # drop_last=True for train
        assert loader.drop_last is True

    def test_val_loader_no_drop_last(self, tmp_path):
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

    def test_test_loader_no_drop_last(self, tmp_path):
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

    def test_pin_memory(self, tmp_path):
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
        assert loader.pin_memory is True

    def test_deterministic_splits(self, tmp_path):
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
        for (f1, l1), (f2, l2) in zip(loader1, loader2):
            assert torch.equal(f1, f2)
            assert torch.equal(l1, l2)

    def test_split_sizes(self, tmp_path):
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
        assert len(train_loader.dataset) == n_train
        assert len(val_loader.dataset) == n_val
        assert len(test_loader.dataset) == n_test

    def test_factorize_fallback(self, tmp_path):
        """Labels encoded via pd.factorize when class_names is None."""
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
        batch_feats, batch_labels = next(iter(loader))
        assert batch_labels.dtype == torch.long

    def test_schema_label_detection(self, tmp_path):
        schema = {
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
        batch_feats, _ = next(iter(loader))
        assert batch_feats.shape[1] == 2

    def test_feature_tensor_dtype(self, tmp_path):
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
        batch_feats, _ = next(iter(loader))
        assert batch_feats.dtype == torch.float32
