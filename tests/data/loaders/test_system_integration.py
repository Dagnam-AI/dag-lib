"""Tests for native system dataset loading and sidecar metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from tests.typing_helpers import JsonObject

from dagnam import load_dataset
from dagnam._core.exceptions import DatasetNotFoundError
from dagnam.data.dataset import DagnamDataset
from dagnam.data.load import _load_internal

# ------------------------------------------------------------------
# DagnamDataset with native datasets
# ------------------------------------------------------------------


class TestDagnamDatasetNativeFields:
    """Verify _native_train/_native_test fields on DagnamDataset."""

    def test_default_native_fields_are_none(self, tmp_path: Path) -> None:
        meta: JsonObject = {
            "id": "test-id",
            "name": "Test",
            "format": "csv",
            "dataset_type": "tabular",
            "num_samples": 100,
            "num_classes": 2,
        }
        ds = DagnamDataset(meta, tmp_path)
        assert ds.native_train is None
        assert ds.native_test is None

    def test_native_fields_set_via_constructor(self, tmp_path: Path) -> None:
        meta: JsonObject = {
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
        assert ds.native_train is train_ds
        assert ds.native_test is test_ds


# ------------------------------------------------------------------
# resolve_system_dataset
# ------------------------------------------------------------------


class TestResolveSystemDataset:
    """Verify system dataset name matching."""

    def test_unknown_dataset_raises(self) -> None:
        from dagnam.data.loaders.system import resolve_system_dataset

        meta: JsonObject = {"name": "totally-unknown-dataset", "dataset_type": "tabular"}
        with pytest.raises(DatasetNotFoundError):
            resolve_system_dataset(meta)

    def test_exact_match_mnist(self) -> None:
        """MNIST should match via exact key."""
        from dagnam.data.loaders.system import NATIVE_LOADERS

        assert "mnist" in NATIVE_LOADERS
        assert "mnist handwritten digits" in NATIVE_LOADERS

    def test_exact_match_cifar10(self) -> None:
        from dagnam.data.loaders.system import NATIVE_LOADERS

        assert "cifar-10" in NATIVE_LOADERS
        assert "cifar10" in NATIVE_LOADERS

    def test_exact_match_imdb(self) -> None:
        from dagnam.data.loaders.system import NATIVE_LOADERS

        assert "imdb" in NATIVE_LOADERS
        assert "imdb movie reviews" in NATIVE_LOADERS


class TestVerifiedSystemDownloads:
    def test_download_helper_rejects_non_https_urls(self, tmp_path: Path) -> None:
        from dagnam.data.loaders.system import torchvision

        with pytest.raises(ValueError, match="HTTPS"):
            torchvision._download_verified_file(  # type: ignore[attr-defined]
                "file:///tmp/imdb.npz",
                tmp_path / "imdb.npz",
                "0" * 64,
            )

    def test_download_helper_verifies_sha256(self, tmp_path: Path) -> None:
        from dagnam.data.loaders.system import torchvision

        class Response:
            def __init__(self) -> None:
                self.headers = {"Content-Length": "3"}

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size: int):
                yield b"bad"

            def __enter__(self) -> Response:
                return self

            def __exit__(
                self, exc_type: type[BaseException] | None, exc: object, tb: object
            ) -> None:
                return None

        with patch(
            "dagnam.data.loaders.system.torchvision.requests.get",
            return_value=Response(),
        ):
            with pytest.raises(ValueError, match="checksum"):
                torchvision._download_verified_file(  # type: ignore[attr-defined]
                    "https://example.test/imdb.npz",
                    tmp_path / "imdb.npz",
                    "0" * 64,
                )

        assert not (tmp_path / "imdb.npz").exists()


# ------------------------------------------------------------------
# load_dataset with source_type detection
# ------------------------------------------------------------------


class TestSourceTypeDetection:
    """Verify that load_dataset routes system datasets to native loaders."""

    def test_uuid_with_system_source_type_routes_to_native(self, tmp_path: Path) -> None:
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
            patch("dagnam.data.load.get_api_key", return_value="key"),
            patch("dagnam.data.load.get_api_url", return_value="http://localhost"),
            patch("dagnam.data.load.DagnamClient") as MockClient,
            patch(
                "dagnam.data.loaders.system.resolve_system_dataset", return_value=mock_native_ds
            ) as mock_resolve,
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

    def test_reads_sidecar_for_user_dataset(self, tmp_path: Path) -> None:
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

        with patch.dict(
            os.environ,
            {
                "DAGNAM_INTERNAL": "true",
                "DAGNAM_META_DIR": str(meta_dir),
            },
        ):
            ds = _load_internal(dataset_id)

        assert ds.name == "User Dataset"
        assert ds._data_dir == data_dir

    def test_reads_sidecar_for_system_dataset(self, tmp_path: Path) -> None:
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
            patch.dict(
                os.environ,
                {
                    "DAGNAM_INTERNAL": "true",
                    "DAGNAM_META_DIR": str(meta_dir),
                },
            ),
            patch(
                "dagnam.data.loaders.system.resolve_system_dataset", return_value=mock_native_ds
            ) as mock_resolve,
        ):
            ds = _load_internal(dataset_id)

        mock_resolve.assert_called_once_with(meta)
        assert ds is mock_native_ds

    def test_missing_sidecar_raises(self, tmp_path: Path) -> None:
        """Missing sidecar file raises FileNotFoundError."""
        meta_dir = tmp_path / ".dagnam_meta"
        meta_dir.mkdir()

        with (
            patch.dict(
                os.environ,
                {
                    "DAGNAM_INTERNAL": "true",
                    "DAGNAM_META_DIR": str(meta_dir),
                },
            ),
            pytest.raises(FileNotFoundError),
        ):
            _load_internal("nonexistent-id")

    def test_internal_dataset_id_cannot_escape_meta_dir(self, tmp_path: Path) -> None:
        """Internal sidecar lookup must not accept path-like dataset IDs."""
        meta_dir = tmp_path / ".dagnam_meta"
        meta_dir.mkdir()
        escaped_meta = tmp_path / "escape.meta.json"
        escaped_meta.write_text(
            json.dumps(
                {
                    "id": "../escape",
                    "name": "Escaped",
                    "format": "csv",
                    "dataset_type": "tabular",
                    "source_type": "uploaded",
                    "num_samples": 1,
                    "num_classes": 1,
                    "file_path": str(escaped_meta),
                }
            ),
            encoding="utf-8",
        )

        with (
            patch.dict(
                os.environ,
                {
                    "DAGNAM_INTERNAL": "true",
                    "DAGNAM_META_DIR": str(meta_dir),
                },
            ),
            pytest.raises(ValueError, match="Unsafe dataset_id"),
        ):
            _load_internal("../escape")
