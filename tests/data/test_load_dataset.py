"""Tests for load_dataset(): name resolution, routing and cache safety."""

from __future__ import annotations

import hashlib
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from dagnam import load_dataset
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import ChecksumError, DagnamError, DatasetNotFoundError
from dagnam._types import JsonObject, JsonValue
from dagnam.data import cache
from dagnam.data.dataset import DagnamDataset
from dagnam.data.load import _is_uuid

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, RequestsMocker

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
# Routing tests: a built-in dataset's name resolves to its id
# ------------------------------------------------------------------

SYSTEM_ID = "7d0b3c1e-2f4a-4b6c-9d8e-0f1a2b3c4d5e"
BROWSE_URL = "http://localhost/api/v1/datasets/browse"


class TestSystemDatasetRouting:
    """The API addresses datasets only by id, so a name is looked up first."""

    def test_name_resolves_to_the_system_row_then_loads_by_id(
        self, tmp_path: Path, requests_mock: RequestsMocker
    ) -> None:
        requests_mock.get(
            BROWSE_URL,
            json=[
                {"id": "11111111-2222-4333-8444-555555555555", "name": "Fashion MNIST"},
                {"id": SYSTEM_ID, "name": "MNIST Handwritten Digits"},
            ],
        )
        meta = {**SYSTEM_META, "id": SYSTEM_ID, "source_type": "system"}
        requests_mock.get(f"http://localhost/api/v1/datasets/{SYSTEM_ID}/meta", json=meta)
        mock_native_ds = MagicMock(spec=DagnamDataset)

        with (
            patch("dagnam.data.load.get_api_key", return_value="key"),
            patch("dagnam.data.load.get_api_url", return_value="http://localhost"),
            patch(
                "dagnam.data.loaders.system.load_system_dataset", return_value=mock_native_ds
            ) as mock_resolve,
        ):
            ds = load_dataset("mnist handwritten digits", cache_dir=str(tmp_path))

        assert ds is mock_native_ds
        mock_resolve.assert_called_once_with(meta, binding=None)
        assert requests_mock.request_history[0].qs == {"source_type": ["system"]}

    def test_unknown_name_raises_not_found(
        self, tmp_path: Path, requests_mock: RequestsMocker
    ) -> None:
        requests_mock.get(BROWSE_URL, json=[{"id": SYSTEM_ID, "name": "MNIST Handwritten Digits"}])

        with (
            patch("dagnam.data.load.get_api_key", return_value="key"),
            patch("dagnam.data.load.get_api_url", return_value="http://localhost"),
            pytest.raises(DatasetNotFoundError) as exc_info,
        ):
            load_dataset("mnist-digits", cache_dir=str(tmp_path))
        assert requests_mock.call_count == 1
        assert exc_info.value.dataset_id == "mnist-digits"
        assert "dagnam.datasets.list_system()" in str(exc_info.value)


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

        def _fake_download(_ds_id: str, output_dir: Path, **_kwargs: object) -> Path:
            # The new locked flow hands download_dataset the *staging* dir; a
            # realistic mock writes the file there so os.replace can promote it.
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            dest = out / "data.csv"
            dest.write_bytes(csv_content)
            return dest

        with (
            patch("dagnam.data.load.get_api_key", return_value="key"),
            patch("dagnam.data.load.get_api_url", return_value="http://localhost"),
            patch.object(DagnamClient, "get_dataset_meta", return_value=meta) as mock_user_meta,
            patch.object(DagnamClient, "list_system_datasets") as mock_sys_list,
            patch.object(
                DagnamClient, "download_dataset", side_effect=_fake_download
            ) as mock_user_dl,
        ):
            ds = load_dataset(dataset_id, cache_dir=str(tmp_path))

            mock_user_meta.assert_called_once_with(dataset_id, version=None)
            mock_sys_list.assert_not_called()
            mock_user_dl.assert_called_once()
            # download was handed the deterministic staging dir, not the final dir
            staging_arg = Path(mock_user_dl.call_args.args[1])
            assert staging_arg == tmp_path / ".staging" / dataset_id
            assert ds.name == "My Custom Dataset"
            # promoted: data lives in the final cache dir, staging is consumed
            assert (tmp_path / dataset_id / "data.csv").read_bytes() == csv_content
            assert not staging_arg.exists()


# ------------------------------------------------------------------
# Client method tests
# ------------------------------------------------------------------


class TestClientSystemMethods:
    """The system catalog is the browse route filtered to built-in datasets."""

    def test_list_system_datasets(self, requests_mock: RequestsMocker) -> None:
        client = DagnamClient("http://localhost:8000", "test-key")
        url = "http://localhost:8000/api/v1/datasets/browse"
        requests_mock.get(url, json=[{"id": SYSTEM_ID, "name": "MNIST"}])

        result = client.list_system_datasets()

        assert requests_mock.call_count == 1
        assert requests_mock.last_request.qs == {"source_type": ["system"]}
        assert result == [{"id": SYSTEM_ID, "name": "MNIST"}]


# ------------------------------------------------------------------
# Concurrency-safe caching: locks + atomic staging + verify
# ------------------------------------------------------------------

USER_DATASET_ID = "550e8400-e29b-41d4-a716-446655440099"


def _full_user_meta(**extra: JsonValue) -> JsonObject:
    """A server meta dict carrying every field DagnamDataset requires."""
    meta: JsonObject = {
        **USER_META,
        "id": USER_DATASET_ID,
        "filename": "d.bin",
        "source_type": "user",
        **extra,
    }
    return meta


def _seed_valid_cache(tmp_path: Path, content: bytes = b"HELLO") -> str:
    """Populate a valid cache entry (data + _cache meta + checksum) and return the checksum."""
    data = cache.get_cache_dir(USER_DATASET_ID, tmp_path) / "d.bin"
    data.write_bytes(content)
    checksum = cache.compute_file_checksum(data)
    cache.save_metadata(USER_DATASET_ID, _full_user_meta(), tmp_path, data_file=data)
    cache.save_checksum(USER_DATASET_ID, checksum, tmp_path)
    return checksum


def test_load_dataset_valid_hit_skips_download(tmp_path: Path) -> None:
    """A cache hit outside the lock never touches download_dataset at all."""
    checksum = _seed_valid_cache(tmp_path)
    meta = _full_user_meta(checksum=checksum)
    with (
        patch("dagnam.data.load.get_api_key", return_value="key"),
        patch("dagnam.data.load.get_api_url", return_value="http://localhost"),
        patch.object(DagnamClient, "get_dataset_meta", return_value=meta),
        patch.object(DagnamClient, "download_dataset") as mock_dl,
    ):
        ds = load_dataset(USER_DATASET_ID, cache_dir=str(tmp_path))
    mock_dl.assert_not_called()
    assert ds is not None


def test_load_dataset_lock_timeout_raises_dagnam_error(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_CACHE_LOCK_TIMEOUT", "0.1")  # keep the test fast
    meta = _full_user_meta(checksum="deadbeef")
    held = cache.dataset_lock(USER_DATASET_ID, base_dir=tmp_path, timeout=0.1)
    held.acquire()
    try:
        with (
            patch("dagnam.data.load.get_api_key", return_value="key"),
            patch("dagnam.data.load.get_api_url", return_value="http://localhost"),
            patch.object(DagnamClient, "get_dataset_meta", return_value=meta),
            pytest.raises(DagnamError),
        ):
            load_dataset(USER_DATASET_ID, cache_dir=str(tmp_path))
    finally:
        held.release()


def test_load_recheck_after_lock_skips_redundant_download(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    """A peer fills the cache WHILE we block on the dataset lock; the in-lock
    recheck must observe it and skip the download entirely."""
    monkeypatch.setenv("DAGNAM_CACHE_LOCK_TIMEOUT", "5")
    peer_holds_lock = threading.Event()

    def peer() -> None:
        lock = cache.dataset_lock(USER_DATASET_ID, base_dir=tmp_path)
        with lock:
            peer_holds_lock.set()
            time.sleep(0.3)
            data = cache.get_cache_dir(USER_DATASET_ID, tmp_path) / "d.bin"
            data.write_bytes(b"HELLO")
            cache.save_metadata(USER_DATASET_ID, _full_user_meta(), tmp_path, data_file=data)
            cache.save_checksum(USER_DATASET_ID, cache.compute_file_checksum(data), tmp_path)

    # A leftover staging dir from an aborted prior attempt: the peer-fill branch
    # must reap it once a valid final entry exists.
    staging_leftover = tmp_path / ".staging" / USER_DATASET_ID
    staging_leftover.mkdir(parents=True)
    (staging_leftover / "part.bin").write_bytes(b"partial")

    t = threading.Thread(target=peer)
    t.start()
    peer_holds_lock.wait(timeout=5)  # peer holds the lock; hasn't written yet

    # verify_cached's cheap (default) path only compares recorded size/mtime,
    # never the passed server_checksum -- so a checksum unknown to us at
    # meta-fetch time doesn't matter; the in-lock recheck passes off size+mtime.
    meta = _full_user_meta(checksum="irrelevant-cheap-path-ignores-this")
    with (
        patch("dagnam.data.load.get_api_key", return_value="key"),
        patch("dagnam.data.load.get_api_url", return_value="http://localhost"),
        patch.object(DagnamClient, "get_dataset_meta", return_value=meta),
        patch.object(DagnamClient, "download_dataset") as mock_dl,
    ):
        ds = load_dataset(USER_DATASET_ID, cache_dir=str(tmp_path))
    t.join(timeout=5)
    mock_dl.assert_not_called()
    assert ds is not None
    assert not staging_leftover.exists()  # the leftover staging dir was reaped


def test_load_checksum_mismatch_leaves_no_promoted_dir(tmp_path: Path) -> None:
    """A checksum mismatch raises and leaves only the (corrupt) staging dir --
    never a promoted final_dir."""
    meta = _full_user_meta(checksum="sha256:" + hashlib.sha256(b"EXPECTED").hexdigest())

    def _fake_download(_ds_id: str, output_dir: Path, **_kwargs: object) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "d.bin"
        dest.write_bytes(b"CORRUPT")  # wrong bytes -> checksum mismatch
        return dest

    with (
        patch("dagnam.data.load.get_api_key", return_value="key"),
        patch("dagnam.data.load.get_api_url", return_value="http://localhost"),
        patch.object(DagnamClient, "get_dataset_meta", return_value=meta),
        patch.object(DagnamClient, "download_dataset", side_effect=_fake_download),
    ):
        with pytest.raises(ChecksumError):
            load_dataset(USER_DATASET_ID, cache_dir=str(tmp_path))
    # No promoted final dir; the corrupt staging dir was cleaned up.
    assert not (tmp_path / USER_DATASET_ID / "d.bin").exists()
    assert not (tmp_path / ".staging" / USER_DATASET_ID).exists()


def test_load_replaces_stale_final_dir_before_promotion(tmp_path: Path) -> None:
    """A stale/corrupt final_dir that verify_cached rejects is removed so the
    atomic os.replace has a guaranteed-absent target."""
    content = b"FRESH"
    checksum = hashlib.sha256(content).hexdigest()
    meta = _full_user_meta(checksum=checksum)
    # Pre-existing final dir with NO valid _cache meta -> verify_cached returns False.
    stale = cache.get_cache_dir(USER_DATASET_ID, tmp_path)
    (stale / "leftover.bin").write_bytes(b"OLD")

    def _fake_download(_ds_id: str, output_dir: Path, **_kwargs: object) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "d.bin"
        dest.write_bytes(content)
        return dest

    with (
        patch("dagnam.data.load.get_api_key", return_value="key"),
        patch("dagnam.data.load.get_api_url", return_value="http://localhost"),
        patch.object(DagnamClient, "get_dataset_meta", return_value=meta),
        patch.object(DagnamClient, "download_dataset", side_effect=_fake_download),
    ):
        ds = load_dataset(USER_DATASET_ID, cache_dir=str(tmp_path))
    assert ds is not None
    final = tmp_path / USER_DATASET_ID
    assert (final / "d.bin").read_bytes() == content
    assert not (final / "leftover.bin").exists()  # stale leftover was cleared


def test_load_verify_full_rehash_on_hit(tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    """verify=True forces a full re-hash of the cached file on the hit path."""
    checksum = _seed_valid_cache(tmp_path)
    meta = _full_user_meta(checksum=checksum)
    rehashed = {"n": 0}
    real_checksum = cache.compute_file_checksum

    def _counting_checksum(path: Path) -> str:
        rehashed["n"] += 1
        return real_checksum(path)

    monkeypatch.setattr("dagnam.data.cache.compute_file_checksum", _counting_checksum)
    with (
        patch("dagnam.data.load.get_api_key", return_value="key"),
        patch("dagnam.data.load.get_api_url", return_value="http://localhost"),
        patch.object(DagnamClient, "get_dataset_meta", return_value=meta),
        patch.object(DagnamClient, "download_dataset") as mock_dl,
    ):
        ds = load_dataset(USER_DATASET_ID, cache_dir=str(tmp_path), verify=True)
    mock_dl.assert_not_called()
    assert rehashed["n"] >= 1  # full re-hash actually happened
    assert ds is not None


def test_load_logs_evicted_dirs(tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    """When post-download eviction removes dirs, the count is logged."""
    content = b"FRESH"
    checksum = hashlib.sha256(content).hexdigest()
    meta = _full_user_meta(checksum=checksum)

    def _fake_download(_ds_id: str, output_dir: Path, **_kwargs: object) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "d.bin"
        dest.write_bytes(content)
        return dest

    monkeypatch.setattr("dagnam.data.load.evict_lru_locked", lambda **_kw: ["old-ds"])
    with (
        patch("dagnam.data.load.get_api_key", return_value="key"),
        patch("dagnam.data.load.get_api_url", return_value="http://localhost"),
        patch.object(DagnamClient, "get_dataset_meta", return_value=meta),
        patch.object(DagnamClient, "download_dataset", side_effect=_fake_download),
    ):
        ds = load_dataset(USER_DATASET_ID, cache_dir=str(tmp_path))
    assert ds is not None
