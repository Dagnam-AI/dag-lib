"""Tests for native system dataset loading and sidecar metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dagnam import load_dataset, _load_internal
from dagnam.data.dataset import DagnamDataset
from dagnam._core.exceptions import DatasetNotFoundError


# ------------------------------------------------------------------
# DagnamDataset with native datasets
# ------------------------------------------------------------------


class TestDagnamDatasetNativeFields:
    """Verify _native_train/_native_test fields on DagnamDataset."""

    def test_default_native_fields_are_none(self, tmp_path: Path):
        meta = {
            "id": "test-id",
            "name": "Test",
            "format": "csv",
            "dataset_type": "tabular",
            "num_samples": 100,
            "num_classes": 2,
        }
        ds = DagnamDataset(meta, tmp_path)
        assert ds._native_train is None
        assert ds._native_test is None

    def test_native_fields_set_via_constructor(self, tmp_path: Path):
        meta = {
            "id": "test-id",
            "name": "Test",
            "format": "csv",
            "dataset_type": "tabular",
            "num_samples": 100,
            "num_classes": 2,
        }
        train_ds = MagicMock()
        test_ds = MagicMock()
        ds = DagnamDataset(meta, tmp_path, _native_train=train_ds, _native_test=test_ds)
        assert ds._native_train is train_ds
        assert ds._native_test is test_ds


# ------------------------------------------------------------------
# resolve_system_dataset
# ------------------------------------------------------------------


class TestResolveSystemDataset:
    """Verify system dataset name matching."""

    def test_unknown_dataset_raises(self):
        from dagnam.data.loaders.system_loader import resolve_system_dataset
        meta = {"name": "totally-unknown-dataset", "dataset_type": "tabular"}
        with pytest.raises(DatasetNotFoundError):
            resolve_system_dataset(meta)

    def test_exact_match_mnist(self):
        """MNIST should match via exact key."""
        from dagnam.data.loaders.system_loader import _NATIVE_LOADERS
        assert "mnist" in _NATIVE_LOADERS
        assert "mnist handwritten digits" in _NATIVE_LOADERS

    def test_exact_match_cifar10(self):
        from dagnam.data.loaders.system_loader import _NATIVE_LOADERS
        assert "cifar-10" in _NATIVE_LOADERS
        assert "cifar10" in _NATIVE_LOADERS

    def test_exact_match_imdb(self):
        from dagnam.data.loaders.system_loader import _NATIVE_LOADERS
        assert "imdb" in _NATIVE_LOADERS
        assert "imdb movie reviews" in _NATIVE_LOADERS


# ------------------------------------------------------------------
# load_dataset with source_type detection
# ------------------------------------------------------------------


class TestSourceTypeDetection:
    """Verify that load_dataset routes system datasets to native loaders."""

    def test_uuid_with_system_source_type_routes_to_native(self, tmp_path: Path):
        """When /meta returns source_type=system, route to native loader."""
        meta = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "name": "MNIST Handwritten Digits",
            "format": "csv",
            "dataset_type": "image",
            "source_type": "system",
            "num_samples": 60000,
            "num_classes": 10,
            "checksum": "abc123",
        }

        mock_native_ds = MagicMock(spec=DagnamDataset)

        with (
            patch("dagnam.get_api_key", return_value="key"),
            patch("dagnam.get_api_url", return_value="http://localhost"),
            patch("dagnam.DagnamClient") as MockClient,
            patch("dagnam.data.loaders.system_loader.resolve_system_dataset", return_value=mock_native_ds) as mock_resolve,
        ):
            MockClient.return_value.get_dataset_meta.return_value = meta

            ds = load_dataset(
                "550e8400-e29b-41d4-a716-446655440000",
                cache_dir=str(tmp_path),
            )

            mock_resolve.assert_called_once_with(meta)
            assert ds is mock_native_ds


# ------------------------------------------------------------------
# _load_internal (sidecar metadata)
# ------------------------------------------------------------------


class TestLoadInternal:
    """Verify sidecar metadata loading for server-side training."""

    def test_reads_sidecar_for_user_dataset(self, tmp_path: Path):
        """User dataset sidecar → direct file read."""
        dataset_id = "550e8400-e29b-41d4-a716-446655440000"
        meta_dir = tmp_path / ".dagnam_meta"
        meta_dir.mkdir()

        # Create a data file
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "data.csv").write_text("a,b\n1,2\n")

        meta = {
            "id": dataset_id,
            "name": "User Dataset",
            "format": "csv",
            "dataset_type": "tabular",
            "source_type": "uploaded",
            "num_samples": 1,
            "num_classes": 2,
            "file_path": str(data_dir / "data.csv"),
        }
        (meta_dir / f"{dataset_id}.meta.json").write_text(json.dumps(meta))

        with patch.dict(os.environ, {
            "DAGNAM_INTERNAL": "true",
            "DAGNAM_META_DIR": str(meta_dir),
        }):
            ds = _load_internal(dataset_id)

        assert ds.name == "User Dataset"
        assert ds._data_dir == data_dir

    def test_reads_sidecar_for_system_dataset(self, tmp_path: Path):
        """System dataset sidecar → native loader."""
        dataset_id = "system-mnist-id"
        meta_dir = tmp_path / ".dagnam_meta"
        meta_dir.mkdir()

        meta = {
            "id": dataset_id,
            "name": "MNIST Handwritten Digits",
            "format": "csv",
            "dataset_type": "image",
            "source_type": "system",
            "num_samples": 60000,
            "num_classes": 10,
            "file_path": None,
        }
        (meta_dir / f"{dataset_id}.meta.json").write_text(json.dumps(meta))

        mock_native_ds = MagicMock(spec=DagnamDataset)

        with (
            patch.dict(os.environ, {
                "DAGNAM_INTERNAL": "true",
                "DAGNAM_META_DIR": str(meta_dir),
            }),
            patch("dagnam.data.loaders.system_loader.resolve_system_dataset", return_value=mock_native_ds) as mock_resolve,
        ):
            ds = _load_internal(dataset_id)

        mock_resolve.assert_called_once_with(meta)
        assert ds is mock_native_ds

    def test_missing_sidecar_raises(self, tmp_path: Path):
        """Missing sidecar file raises FileNotFoundError."""
        meta_dir = tmp_path / ".dagnam_meta"
        meta_dir.mkdir()

        with (
            patch.dict(os.environ, {
                "DAGNAM_INTERNAL": "true",
                "DAGNAM_META_DIR": str(meta_dir),
            }),
        ):
            with pytest.raises(FileNotFoundError):
                _load_internal("nonexistent-id")
