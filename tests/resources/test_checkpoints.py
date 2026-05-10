"""Unit tests for dagnam.checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import CheckpointNotFoundError, ChecksumError
from dagnam.resources.checkpoints import _pick_latest, download_checkpoint


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def ck_cache(tmp_path) -> Path:
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


class TestPickLatest:
    def test_prefers_best(self):
        c = _pick_latest(
            [
                {"id": "a", "epoch": 5, "step": 100, "is_best": False, "created_at": "t1"},
                {"id": "b", "epoch": 3, "step": 60, "is_best": True, "created_at": "t2"},
            ]
        )
        assert c["id"] == "b"

    def test_falls_back_to_highest_epoch(self):
        c = _pick_latest(
            [
                {"id": "a", "epoch": 1, "step": 10, "is_best": False},
                {"id": "b", "epoch": 5, "step": 50, "is_best": False},
                {"id": "c", "epoch": 3, "step": 30, "is_best": False},
            ]
        )
        assert c["id"] == "b"

    def test_empty_raises(self):
        with pytest.raises(CheckpointNotFoundError):
            _pick_latest([])


class TestDownloadCheckpoint:
    def _fake_download(self, body: bytes):
        """Return a DagnamClient.download_checkpoint_stream side_effect."""
        expected_sha = _sha256(body)

        def _side_effect(job_id, checkpoint_id, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
            return dest, expected_sha

        return _side_effect, expected_sha

    def test_downloads_explicit_checkpoint(self, ck_cache):
        body = b"weights-bytes"
        side, _ = self._fake_download(body)
        client = MagicMock(spec=DagnamClient)
        client.download_checkpoint_stream.side_effect = side

        path = download_checkpoint("job_1", "ck_1", client=client, cache_dir=ck_cache)
        assert path.exists()
        assert path.read_bytes() == body
        assert path == ck_cache / "job_1" / "ck_1.pt"

    def test_cache_hit_skips_download(self, ck_cache):
        cached = ck_cache / "job_1" / "ck_1.pt"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"already-here")
        client = MagicMock(spec=DagnamClient)

        path = download_checkpoint("job_1", "ck_1", client=client, cache_dir=ck_cache)
        assert path == cached
        client.download_checkpoint_stream.assert_not_called()

    def test_checksum_mismatch_raises_and_removes(self, ck_cache):
        def side_effect(job_id, checkpoint_id, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"actual-bytes")
            return dest, "deadbeef" * 8  # wrong sha

        client = MagicMock(spec=DagnamClient)
        client.download_checkpoint_stream.side_effect = side_effect

        with pytest.raises(ChecksumError):
            download_checkpoint("job_1", "ck_1", client=client, cache_dir=ck_cache)

        assert not (ck_cache / "job_1" / "ck_1.pt").exists()

    def test_picks_latest_when_id_omitted(self, ck_cache):
        body = b"x"
        side, _ = self._fake_download(body)
        client = MagicMock(spec=DagnamClient)
        client.list_checkpoints.return_value = [
            {"id": "old", "epoch": 1, "step": 1, "is_best": False},
            {"id": "new", "epoch": 5, "step": 50, "is_best": True},
        ]
        client.download_checkpoint_stream.side_effect = side

        path = download_checkpoint("job_1", client=client, cache_dir=ck_cache)
        assert path == ck_cache / "job_1" / "new.pt"
        client.list_checkpoints.assert_called_once_with("job_1")
