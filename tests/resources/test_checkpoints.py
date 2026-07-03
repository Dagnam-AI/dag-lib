"""Unit tests for dagnam.checkpoints."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import CheckpointNotFoundError, ChecksumError
from dagnam.resources.checkpoints import download_checkpoint, pick_best, pick_latest


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def ck_cache(tmp_path: Path) -> Path:
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


class TestPickLatest:
    def test_returns_newest_even_when_older_checkpoint_is_best(self) -> None:
        c = pick_latest(
            [
                {"id": "a", "epoch": 5, "step": 100, "is_best": False, "created_at": "t1"},
                {"id": "b", "epoch": 3, "step": 60, "is_best": True, "created_at": "t2"},
            ]
        )
        assert c["id"] == "a"

    def test_pick_best_prefers_best_checkpoint(self) -> None:
        c = pick_best(
            [
                {"id": "latest", "epoch": 5, "step": 100, "is_best": False},
                {"id": "best", "epoch": 3, "step": 60, "is_best": True},
            ]
        )
        assert c["id"] == "best"

    def test_falls_back_to_highest_epoch(self) -> None:
        c = pick_latest(
            [
                {"id": "a", "epoch": 1, "step": 10, "is_best": False},
                {"id": "b", "epoch": 5, "step": 50, "is_best": False},
                {"id": "c", "epoch": 3, "step": 30, "is_best": False},
            ]
        )
        assert c["id"] == "b"

    def test_empty_raises(self) -> None:
        with pytest.raises(CheckpointNotFoundError):
            pick_latest([])


class TestDownloadCheckpoint:
    def _fake_download(
        self, body: bytes
    ) -> tuple[Callable[[str, str, Path], tuple[Path, str]], str]:
        """Return a DagnamClient.download_checkpoint_stream side_effect."""
        expected_sha = _sha256(body)

        def _side_effect(job_id: str, checkpoint_id: str, dest: Path) -> tuple[Path, str]:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
            return dest, expected_sha

        return _side_effect, expected_sha

    def test_downloads_explicit_checkpoint(self, ck_cache: Path) -> None:
        body = b"weights-bytes"
        side, _ = self._fake_download(body)
        client = MagicMock(spec=DagnamClient)
        client.download_checkpoint_stream.side_effect = side

        path = download_checkpoint("job_1", "ck_1", client=client, cache_dir=ck_cache)
        assert path.exists()
        assert path.read_bytes() == body
        assert path == ck_cache / "job_1" / "ck_1.pt"

    def test_cache_hit_skips_download(self, ck_cache: Path) -> None:
        cached = ck_cache / "job_1" / "ck_1.pt"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"already-here")
        client = MagicMock(spec=DagnamClient)

        path = download_checkpoint("job_1", "ck_1", client=client, cache_dir=ck_cache)
        assert path == cached
        client.download_checkpoint_stream.assert_not_called()

    def test_missing_server_checksum_warns_loudly(
        self, ck_cache: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # S3/presigned downloads carry no X-Checksum-SHA256 header, so the
        # stream returns None. The file must still be accepted (S3 works), but
        # the unverified state must be LOUD, never silent.
        def side_effect(job_id: str, checkpoint_id: str, dest: Path) -> tuple[Path, None]:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"weights")
            return dest, None

        client = MagicMock(spec=DagnamClient)
        client.download_checkpoint_stream.side_effect = side_effect

        with caplog.at_level(logging.WARNING):
            path = download_checkpoint("job_1", "ck_1", client=client, cache_dir=ck_cache)

        assert path.exists()
        assert any("checksum" in record.message.lower() for record in caplog.records)

    def test_checksum_mismatch_raises_and_removes(self, ck_cache: Path) -> None:
        def side_effect(job_id: str, checkpoint_id: str, dest: Path):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"actual-bytes")
            return dest, "deadbeef" * 8  # wrong sha

        client = MagicMock(spec=DagnamClient)
        client.download_checkpoint_stream.side_effect = side_effect

        with pytest.raises(ChecksumError):
            download_checkpoint("job_1", "ck_1", client=client, cache_dir=ck_cache)

        assert not (ck_cache / "job_1" / "ck_1.pt").exists()

    def test_picks_latest_when_id_omitted(self, ck_cache: Path) -> None:
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

    def test_checkpoint_cache_paths_cannot_escape_cache_dir(
        self, ck_cache: Path, tmp_path: Path
    ) -> None:
        body = b"weights"
        side, _ = self._fake_download(body)
        client = MagicMock(spec=DagnamClient)
        client.download_checkpoint_stream.side_effect = side

        path = download_checkpoint("../job", "../ck", client=client, cache_dir=ck_cache)

        assert path.resolve().is_relative_to(ck_cache.resolve())
        assert path == ck_cache / "..%2Fjob" / "..%2Fck.pt"
        assert not (tmp_path / "job").exists()
        assert not (tmp_path / "ck.pt").exists()
