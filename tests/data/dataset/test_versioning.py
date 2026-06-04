"""Tests for dataset version selection in load_dataset and DagnamClient."""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

from dagnam import load_dataset
from dagnam._core.client import DagnamClient


class TestClientVersionQuery:
    """Tests that DagnamClient sends version query params."""

    def test_get_dataset_meta_sends_version_query(self) -> None:
        """version param is sent as query parameter."""
        client = DagnamClient("https://api.test", "key")
        resp = MagicMock()
        resp.ok = True
        resp.status_code = 200
        resp.json.return_value = {"id": "ds", "checksum": "sha256:abc"}

        with patch("dagnam._core.client.base.requests.get", return_value=resp) as mock_get:
            client.get_dataset_meta("ds", version="v2")

        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs["params"] == {"version": "v2"}

    def test_get_dataset_meta_no_version_no_params(self) -> None:
        """No params sent when version is None."""
        client = DagnamClient("https://api.test", "key")
        resp = MagicMock()
        resp.ok = True
        resp.status_code = 200
        resp.json.return_value = {"id": "ds"}

        with patch("dagnam._core.client.base.requests.get", return_value=resp) as mock_get:
            client.get_dataset_meta("ds")

        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs.get("params") is None

    def test_get_system_dataset_meta_sends_version_query(self) -> None:
        """version param is sent for system dataset meta."""
        client = DagnamClient("https://api.test", "key")
        resp = MagicMock()
        resp.ok = True
        resp.status_code = 200
        resp.json.return_value = {"id": "mnist", "source_type": "system"}

        with patch("dagnam._core.client.base.requests.get", return_value=resp) as mock_get:
            client.get_system_dataset_meta("mnist", version="v3")

        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs["params"] == {"version": "v3"}


class TestVersionedCacheKey:
    """Tests that versioned datasets use version-aware cache keys."""

    def test_load_dataset_uses_versioned_cache_key(self, tmp_path: Path) -> None:
        """Cache directory uses {dataset_id}@{version} format."""
        body = b"a,b\n1,2\n"
        checksum = "sha256:" + hashlib.sha256(body).hexdigest()
        dataset_id = "550e8400-e29b-41d4-a716-446655440000"
        meta = {
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

        # Pre-create the versioned cache directory with data
        versioned_dir = tmp_path / f"{dataset_id}@v2"
        versioned_dir.mkdir(parents=True)
        (versioned_dir / "data.csv").write_bytes(body)
        (versioned_dir / ".checksum").write_text(checksum, encoding="utf-8")
        (versioned_dir / ".last_access").write_text("1000.0", encoding="utf-8")

        # Write metadata
        import json

        (versioned_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        with (
            patch("dagnam.data.load.get_api_key", return_value="key"),
            patch("dagnam.data.load.get_api_url", return_value="https://api.test"),
            patch.object(DagnamClient, "get_dataset_meta", return_value=meta),
            patch.object(
                DagnamClient, "download_dataset", return_value=versioned_dir / "data.csv"
            ) as mock_download,
        ):
            ds = load_dataset(dataset_id, version="v2", cache_dir=str(tmp_path))

        assert ds._data_dir.name == f"{dataset_id}@v2"
        mock_download.assert_not_called()

    def test_load_dataset_passes_version_and_filename_to_download(self, tmp_path: Path) -> None:
        """Version and metadata filename are forwarded to the download endpoint."""
        body = b"a,b\n1,2\n"
        checksum = "sha256:" + hashlib.sha256(body).hexdigest()
        dataset_id = "550e8400-e29b-41d4-a716-446655440000"
        meta = {
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
        downloaded = tmp_path / f"{dataset_id}@v2" / "versioned.csv"
        downloaded.parent.mkdir(parents=True)
        downloaded.write_bytes(body)

        with (
            patch("dagnam.data.load.get_api_key", return_value="key"),
            patch("dagnam.data.load.get_api_url", return_value="https://api.test"),
            patch.object(DagnamClient, "get_dataset_meta", return_value=meta),
            patch.object(
                DagnamClient, "download_dataset", return_value=downloaded
            ) as mock_download,
        ):
            load_dataset(dataset_id, version="v2", cache_dir=str(tmp_path))

        assert mock_download.call_args.kwargs["version"] == "v2"
        assert mock_download.call_args.kwargs["filename"] == "versioned.csv"

    def test_load_dataset_unversioned_uses_plain_id(self, tmp_path: Path) -> None:
        """Without version, cache directory uses plain dataset_id."""
        body = b"a,b\n1,2\n"
        checksum = "sha256:" + hashlib.sha256(body).hexdigest()
        dataset_id = "550e8400-e29b-41d4-a716-446655440000"
        meta = {
            "id": dataset_id,
            "name": "Unversioned",
            "format": "csv",
            "dataset_type": "tabular",
            "num_samples": 1,
            "num_classes": 1,
            "checksum": checksum,
            "source_type": "uploaded",
        }

        # Pre-create the cache directory with data
        cache_dir = tmp_path / dataset_id
        cache_dir.mkdir(parents=True)
        (cache_dir / "data.csv").write_bytes(body)
        (cache_dir / ".checksum").write_text(checksum, encoding="utf-8")
        (cache_dir / ".last_access").write_text("1000.0", encoding="utf-8")

        import json

        (cache_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        with (
            patch("dagnam.data.load.get_api_key", return_value="key"),
            patch("dagnam.data.load.get_api_url", return_value="https://api.test"),
            patch.object(DagnamClient, "get_dataset_meta", return_value=meta),
        ):
            ds = load_dataset(dataset_id, cache_dir=str(tmp_path))

        assert ds._data_dir.name == dataset_id
