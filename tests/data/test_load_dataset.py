"""Tests for system dataset name resolution in load_dataset()."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dagnam import load_dataset
from dagnam._core.client import DagnamClient
from dagnam.data.dataset import DagnamDataset
from dagnam.data.load import _is_uuid

# ------------------------------------------------------------------
# _is_uuid tests
# ------------------------------------------------------------------


class TestIsUuid:
    """Verify the UUID detection helper."""

    @pytest.mark.parametrize(
        "value",
        [
            "550e8400-e29b-41d4-a716-446655440000",
            "A550E840-E29B-41D4-A716-446655440000",
            "00000000-0000-0000-0000-000000000000",
        ],
    )
    def test_valid_uuids(self, value: str) -> None:
        assert _is_uuid(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "mnist",
            "imdb-sentiment",
            "cifar-10",
            "my-custom-dataset",
            "not-a-uuid-at-all",
            "",
            "550e8400e29b41d4a716446655440000",  # no dashes
            "550e8400-e29b-41d4-a716",  # too short
        ],
    )
    def test_non_uuids(self, value: str) -> None:
        assert _is_uuid(value) is False


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


SYSTEM_META = {
    "id": "mnist-digits",
    "name": "MNIST Digits",
    "format": "csv",
    "dataset_type": "image",
    "num_samples": 60000,
    "num_classes": 10,
    "feature_schema": None,
    "class_names": None,
    "checksum": "placeholder",  # replaced per-test
}

USER_META = {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "My Custom Dataset",
    "format": "csv",
    "dataset_type": "tabular",
    "num_samples": 100,
    "num_classes": 2,
    "feature_schema": None,
    "class_names": None,
    "checksum": "placeholder",
}


# ------------------------------------------------------------------
# Routing tests — friendly names → system endpoints
# ------------------------------------------------------------------


class TestSystemDatasetRouting:
    """Friendly names should route through system dataset endpoints."""

    def test_friendly_name_calls_system_meta(self, tmp_path: Path) -> None:
        meta = {
            **SYSTEM_META,
            "checksum": "placeholder",
            "source_type": "system",
        }

        mock_native_ds = MagicMock(spec=DagnamDataset)

        with (
            patch("dagnam.data.load.get_api_key", return_value="key"),
            patch("dagnam.data.load.get_api_url", return_value="http://localhost"),
            patch.object(
                DagnamClient, "get_system_dataset_meta", return_value=meta
            ) as mock_sys_meta,
            patch.object(DagnamClient, "get_dataset_meta") as mock_user_meta,
            patch(
                "dagnam.data.loaders.system.load_system_dataset", return_value=mock_native_ds
            ) as mock_resolve,
        ):
            ds = load_dataset("mnist-digits", cache_dir=str(tmp_path))

            mock_sys_meta.assert_called_once_with("mnist-digits", version=None)
            mock_user_meta.assert_not_called()
            mock_resolve.assert_called_once_with(meta, binding=None)
            assert ds is mock_native_ds

    def test_friendly_name_with_dashes(self, tmp_path: Path) -> None:
        """Names like 'imdb-sentiment' are NOT UUIDs and should use system endpoints.
        With unified architecture, system datasets route to native loaders."""
        meta = {
            **SYSTEM_META,
            "id": "imdb-sentiment",
            "name": "IMDB Sentiment",
            "checksum": "placeholder",
            "source_type": "system",
        }

        mock_native_ds = MagicMock(spec=DagnamDataset)

        with (
            patch("dagnam.data.load.get_api_key", return_value="key"),
            patch("dagnam.data.load.get_api_url", return_value="http://localhost"),
            patch.object(DagnamClient, "get_system_dataset_meta", return_value=meta),
            patch(
                "dagnam.data.loaders.system.load_system_dataset", return_value=mock_native_ds
            ) as mock_resolve,
        ):
            ds = load_dataset("imdb-sentiment", cache_dir=str(tmp_path))
            assert ds is mock_native_ds
            mock_resolve.assert_called_once_with(meta, binding=None)


# ------------------------------------------------------------------
# Routing tests — UUIDs → user dataset endpoints
# ------------------------------------------------------------------


class TestUserDatasetRouting:
    """UUID IDs should route through the regular user dataset endpoints."""

    def test_uuid_calls_user_meta(self, tmp_path: Path) -> None:
        csv_content = b"a,b\n1,2\n"
        checksum = _sha256(csv_content)
        meta = {**USER_META, "checksum": checksum}
        dataset_id = "550e8400-e29b-41d4-a716-446655440000"

        with (
            patch("dagnam.data.load.get_api_key", return_value="key"),
            patch("dagnam.data.load.get_api_url", return_value="http://localhost"),
            patch.object(DagnamClient, "get_dataset_meta", return_value=meta) as mock_user_meta,
            patch.object(DagnamClient, "get_system_dataset_meta") as mock_sys_meta,
            patch.object(DagnamClient, "download_dataset") as mock_user_dl,
            patch.object(DagnamClient, "download_system_dataset") as mock_sys_dl,
        ):
            dest = tmp_path / dataset_id / "data.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(csv_content)
            mock_user_dl.return_value = dest

            ds = load_dataset(dataset_id, cache_dir=str(tmp_path))

            mock_user_meta.assert_called_once_with(dataset_id, version=None)
            mock_sys_meta.assert_not_called()
            mock_user_dl.assert_called_once()
            mock_sys_dl.assert_not_called()
            assert ds.name == "My Custom Dataset"


# ------------------------------------------------------------------
# Client method tests
# ------------------------------------------------------------------


class TestClientSystemMethods:
    """Verify DagnamClient system dataset methods hit the right URLs."""

    def test_list_system_datasets(self) -> None:
        client = DagnamClient("http://localhost:8000", "test-key")
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = [{"id": "mnist-digits", "name": "MNIST"}]

        with patch("dagnam._core.client.base.requests.get", return_value=mock_resp) as mock_get:
            result = client.list_system_datasets()

        mock_get.assert_called_once()
        call_url = mock_get.call_args[0][0]
        assert call_url == "http://localhost:8000/api/v1/datasets/system"
        assert result == [{"id": "mnist-digits", "name": "MNIST"}]

    def test_get_system_dataset_meta(self) -> None:
        client = DagnamClient("http://localhost:8000", "test-key")
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"id": "mnist-digits", "name": "MNIST"}

        with patch("dagnam._core.client.base.requests.get", return_value=mock_resp) as mock_get:
            result = client.get_system_dataset_meta("mnist-digits")

        call_url = mock_get.call_args[0][0]
        assert call_url == "http://localhost:8000/api/v1/datasets/system/mnist-digits"
        assert result["id"] == "mnist-digits"

    def test_download_system_dataset(self, tmp_path: Path) -> None:
        client = DagnamClient("http://localhost:8000", "test-key")
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.headers = {
            "Content-Disposition": 'attachment; filename="mnist.csv"',
            "Content-Length": "5",
        }
        mock_resp.iter_content.return_value = [b"hello"]

        with patch("dagnam._core.client.base.requests.get", return_value=mock_resp) as mock_get:
            result = client.download_system_dataset("mnist-digits", tmp_path)

        call_url = mock_get.call_args[0][0]
        assert call_url == "http://localhost:8000/api/v1/datasets/system/mnist-digits/download"
        assert result == tmp_path / "mnist.csv"
        assert result.read_bytes() == b"hello"
