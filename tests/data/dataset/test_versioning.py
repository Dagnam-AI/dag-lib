"""Tests for dataset version selection in load_dataset and DagnamClient."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

from dagnam import load_dataset
from dagnam._core.client import DagnamClient
from dagnam._types import JsonObject
from dagnam.data import cache

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker


class TestClientVersionQuery:
    """Tests that DagnamClient sends version query params."""

    def test_get_dataset_meta_sends_version_query(self, requests_mock: RequestsMocker) -> None:
        """version param is sent as query parameter."""
        client = DagnamClient("https://api.test", "key")
        requests_mock.get(
            "https://api.test/api/v1/datasets/ds/meta",
            json={"id": "ds", "checksum": "sha256:abc"},
        )
        client.get_dataset_meta("ds", version="v2")
        assert requests_mock.last_request.qs == {"version": ["v2"]}

    def test_get_dataset_meta_no_version_no_params(self, requests_mock: RequestsMocker) -> None:
        """No params sent when version is None."""
        client = DagnamClient("https://api.test", "key")
        requests_mock.get("https://api.test/api/v1/datasets/ds/meta", json={"id": "ds"})
        client.get_dataset_meta("ds")
        assert requests_mock.last_request.qs == {}

    def test_get_system_dataset_meta_sends_version_query(
        self, requests_mock: RequestsMocker
    ) -> None:
        """version param is sent for system dataset meta."""
        client = DagnamClient("https://api.test", "key")
        requests_mock.get(
            "https://api.test/api/v1/datasets/system/mnist",
            json={"id": "mnist", "source_type": "system"},
        )
        client.get_system_dataset_meta("mnist", version="v3")
        assert requests_mock.last_request.qs == {"version": ["v3"]}


class TestVersionedCacheKey:
    """Tests that versioned datasets use version-aware cache keys."""

    def test_load_dataset_uses_versioned_cache_key(self, tmp_path: Path) -> None:
        """Cache directory uses {dataset_id}@{version} format."""
        body = b"a,b\n1,2\n"
        checksum = "sha256:" + hashlib.sha256(body).hexdigest()
        dataset_id = "550e8400-e29b-41d4-a716-446655440000"
        meta: JsonObject = {
            "id": dataset_id,
            "name": "Versioned",
            "format": "csv",
            "dataset_type": "tabular",
            "num_samples": 1,
            "num_classes": 1,
            "checksum": checksum,
            "version": "v2",
            "source_type": "uploaded",
        }

        # Seed a valid versioned cache entry (keyed on {id}@v2) in the new
        # verify_cached format: data file + size/mtime meta + checksum.
        cache_key = f"{dataset_id}@v2"
        data_file = cache.get_cache_dir(cache_key, tmp_path) / "data.csv"
        data_file.write_bytes(body)
        cache.save_metadata(cache_key, meta, tmp_path, data_file=data_file)
        cache.save_checksum(cache_key, checksum, tmp_path)

        with (
            patch("dagnam.data.load.get_api_key", return_value="key"),
            patch("dagnam.data.load.get_api_url", return_value="https://api.test"),
            patch.object(DagnamClient, "get_dataset_meta", return_value=meta),
            patch.object(DagnamClient, "download_dataset") as mock_download,
        ):
            ds = load_dataset(dataset_id, version="v2", cache_dir=str(tmp_path))

        assert ds._data_dir.name == f"{dataset_id}@v2"
        mock_download.assert_not_called()

    def test_load_dataset_passes_version_and_filename_to_download(self, tmp_path: Path) -> None:
        """Version and metadata filename are forwarded to the download endpoint."""
        body = b"a,b\n1,2\n"
        checksum = "sha256:" + hashlib.sha256(body).hexdigest()
        dataset_id = "550e8400-e29b-41d4-a716-446655440000"
        meta: JsonObject = {
            "id": dataset_id,
            "name": "Versioned",
            "format": "csv",
            "dataset_type": "tabular",
            "num_samples": 1,
            "num_classes": 1,
            "checksum": checksum,
            "filename": "versioned.csv",
            "version": "v2",
            "source_type": "uploaded",
        }

        def _fake_download(_ds_id: str, output_dir: Path, **_kwargs: object) -> Path:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            staged = out / "versioned.csv"
            staged.write_bytes(body)
            return staged

        with (
            patch("dagnam.data.load.get_api_key", return_value="key"),
            patch("dagnam.data.load.get_api_url", return_value="https://api.test"),
            patch.object(DagnamClient, "get_dataset_meta", return_value=meta),
            patch.object(
                DagnamClient, "download_dataset", side_effect=_fake_download
            ) as mock_download,
        ):
            load_dataset(dataset_id, version="v2", cache_dir=str(tmp_path))

        assert mock_download.call_args.kwargs["version"] == "v2"
        assert mock_download.call_args.kwargs["filename"] == "versioned.csv"
        # download was handed the versioned staging dir, then promoted
        assert Path(mock_download.call_args.args[1]) == tmp_path / ".staging" / f"{dataset_id}@v2"

    def test_load_dataset_unversioned_uses_plain_id(self, tmp_path: Path) -> None:
        """Without version, cache directory uses plain dataset_id."""
        body = b"a,b\n1,2\n"
        checksum = "sha256:" + hashlib.sha256(body).hexdigest()
        dataset_id = "550e8400-e29b-41d4-a716-446655440000"
        meta: JsonObject = {
            "id": dataset_id,
            "name": "Unversioned",
            "format": "csv",
            "dataset_type": "tabular",
            "num_samples": 1,
            "num_classes": 1,
            "checksum": checksum,
            "source_type": "uploaded",
        }

        # Seed a valid plain-id cache entry in the new verify_cached format.
        data_file = cache.get_cache_dir(dataset_id, tmp_path) / "data.csv"
        data_file.write_bytes(body)
        cache.save_metadata(dataset_id, meta, tmp_path, data_file=data_file)
        cache.save_checksum(dataset_id, checksum, tmp_path)

        with (
            patch("dagnam.data.load.get_api_key", return_value="key"),
            patch("dagnam.data.load.get_api_url", return_value="https://api.test"),
            patch.object(DagnamClient, "get_dataset_meta", return_value=meta),
            patch.object(DagnamClient, "download_dataset") as mock_download,
        ):
            ds = load_dataset(dataset_id, cache_dir=str(tmp_path))

        assert ds._data_dir.name == dataset_id
        mock_download.assert_not_called()
