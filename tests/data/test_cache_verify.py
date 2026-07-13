"""Cheap-by-default cache-hit re-verification (``verify_cached``)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dagnam.data import cache

if TYPE_CHECKING:
    from pathlib import Path


def _seed(tmp_path: Path, content: bytes = b"DATA") -> tuple[Path, Path, str]:
    d = cache.get_cache_dir("ds1", tmp_path)
    data = d / "data.bin"
    data.write_bytes(content)
    checksum = cache.compute_file_checksum(data)
    cache.save_metadata("ds1", {"id": "ds1"}, tmp_path, data_file=data)
    cache.save_checksum("ds1", checksum, tmp_path)
    return d, data, checksum


def test_verify_cached_cheap_pass(tmp_path: Path) -> None:
    _d, _data, checksum = _seed(tmp_path)
    assert cache.verify_cached("ds1", checksum, base_dir=tmp_path) is True


def test_verify_cached_detects_size_change(tmp_path: Path) -> None:
    _d, data, checksum = _seed(tmp_path)
    data.write_bytes(b"DATA_TAMPERED")  # size + content differ from recorded
    assert cache.verify_cached("ds1", checksum, base_dir=tmp_path) is False


def test_verify_cached_full_rehash(tmp_path: Path) -> None:
    _d, data, checksum = _seed(tmp_path)
    recorded = data.stat()
    # Same size AND same mtime, but different bytes: the cheap size+mtime check
    # PASSES, so only the full sha256 rehash catches the corruption. Rewriting
    # bumps mtime, so restore it to the recorded value to isolate the rehash path.
    data.write_bytes(b"XXXX")
    os.utime(data, (recorded.st_atime, recorded.st_mtime))
    assert cache.verify_cached("ds1", checksum, base_dir=tmp_path, full=True) is False


def test_verify_cached_full_rehash_pass(tmp_path: Path) -> None:
    _d, _data, checksum = _seed(tmp_path)
    # full rehash with the SAME content passes (touches on hit).
    assert cache.verify_cached("ds1", checksum, base_dir=tmp_path, full=True) is True


def test_verify_cached_full_rehash_pass_prefixed_checksum(tmp_path: Path) -> None:
    _d, _data, checksum = _seed(tmp_path)
    # A server checksum with the ``sha256:`` prefix still matches on full rehash.
    assert cache.verify_cached("ds1", f"sha256:{checksum}", base_dir=tmp_path, full=True) is True


def test_verify_cached_missing_data_file(tmp_path: Path) -> None:
    _d, data, checksum = _seed(tmp_path)
    data.unlink()
    assert cache.verify_cached("ds1", checksum, base_dir=tmp_path) is False


def test_verify_cached_no_cache_meta(tmp_path: Path) -> None:
    # save_metadata WITHOUT data_file records no _cache block -> verify returns False.
    cache.get_cache_dir("ds1", tmp_path)
    cache.save_metadata("ds1", {"id": "ds1"}, tmp_path)
    assert cache.verify_cached("ds1", "anything", base_dir=tmp_path) is False


def test_verify_cached_touches_on_hit(tmp_path: Path) -> None:
    d, _data, checksum = _seed(tmp_path)
    access_file = d / ".last_access"
    access_file.unlink(missing_ok=True)
    assert not access_file.exists()
    assert cache.verify_cached("ds1", checksum, base_dir=tmp_path) is True
    assert access_file.exists()


def test_verify_cached_no_touch_on_miss(tmp_path: Path) -> None:
    d, data, checksum = _seed(tmp_path)
    data.write_bytes(b"DATA_TAMPERED")
    access_file = d / ".last_access"
    access_file.unlink(missing_ok=True)
    assert cache.verify_cached("ds1", checksum, base_dir=tmp_path) is False
    assert not access_file.exists()


def test_save_metadata_without_data_file_has_no_cache_block(tmp_path: Path) -> None:
    cache.get_cache_dir("ds1", tmp_path)
    cache.save_metadata("ds1", {"id": "ds1"}, tmp_path)
    meta = cache.load_metadata("ds1", tmp_path)
    assert "_cache" not in meta
