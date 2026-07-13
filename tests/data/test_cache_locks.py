"""Cross-process cache locks: dataset_lock, eviction_lock, evict_lru_locked."""

from __future__ import annotations

from typing import TYPE_CHECKING

import filelock
import pytest

from dagnam.data import cache

if TYPE_CHECKING:
    from pathlib import Path

    from tests.typing_helpers import PytestMonkeyPatch


def test_dataset_lock_path_is_sibling(tmp_path: Path) -> None:
    lock = cache.dataset_lock("ds1", base_dir=tmp_path, timeout=0.1)
    assert lock.lock_file == str(tmp_path / "ds1.lock")  # sibling, not inside ds1/


def test_dataset_lock_is_exclusive(tmp_path: Path) -> None:
    held = cache.dataset_lock("ds1", base_dir=tmp_path, timeout=0.1)
    held.acquire()
    try:
        with pytest.raises(filelock.Timeout):
            cache.dataset_lock("ds1", base_dir=tmp_path, timeout=0.1).acquire()
    finally:
        held.release()


def test_lock_timeout_env(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.delenv("DAGNAM_CACHE_LOCK_TIMEOUT", raising=False)
    assert cache.lock_timeout() == 60.0
    monkeypatch.setenv("DAGNAM_CACHE_LOCK_TIMEOUT", "5")
    assert cache.lock_timeout() == 5.0
    monkeypatch.setenv("DAGNAM_CACHE_LOCK_TIMEOUT", "bad")
    assert cache.lock_timeout() == 60.0  # unparseable -> default


def test_eviction_lock_path(tmp_path: Path) -> None:
    lock = cache.eviction_lock(base_dir=tmp_path, timeout=0.1)
    assert lock.lock_file == str(tmp_path / ".eviction.lock")


def test_evict_lru_locked_serializes(tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    # Holding the eviction lock makes a concurrent locked evict time out -> [] (no crash).
    # A short env timeout keeps the test fast (evict_lru_locked takes no timeout arg,
    # so it reads lock_timeout() internally).
    monkeypatch.setenv("DAGNAM_CACHE_LOCK_TIMEOUT", "0.1")
    held = cache.eviction_lock(base_dir=tmp_path, timeout=0.1)
    held.acquire()
    try:
        assert cache.evict_lru_locked(base_dir=tmp_path) == []
    finally:
        held.release()


def test_evict_lru_locked_evicts_when_lock_free(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    calls = {"n": 0}

    def _fake_evict(max_size_bytes: int | None = None, base_dir: Path | None = None) -> list[str]:
        calls["n"] += 1
        return ["ds-old"]

    monkeypatch.setattr(cache, "evict_lru", _fake_evict)
    assert cache.evict_lru_locked(base_dir=tmp_path) == ["ds-old"]
    assert calls["n"] == 1


def test_staging_dir_is_not_a_cache_entry(tmp_path: Path) -> None:
    # A ``.staging`` sibling holding an in-progress download must never be
    # reported as a dataset (else eviction could rmtree a peer's live download).
    (tmp_path / cache.STAGING_DIR_NAME / "ds-inflight").mkdir(parents=True)
    (tmp_path / cache.STAGING_DIR_NAME / "ds-inflight" / "part.bin").write_bytes(b"x")
    real = cache.get_cache_dir("ds-real", tmp_path)
    (real / "data.bin").write_bytes(b"y")
    ids = {e["dataset_id"] for e in cache.get_cache_info(tmp_path)}
    assert ids == {"ds-real"}
    assert cache.STAGING_DIR_NAME not in ids


def test_evict_lru_locked_passes_max_size(tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def _fake_evict(max_size_bytes: int | None = None, base_dir: Path | None = None) -> list[str]:
        seen["max"] = max_size_bytes
        return []

    monkeypatch.setattr(cache, "evict_lru", _fake_evict)
    cache.evict_lru_locked(max_size_bytes=123, base_dir=tmp_path)
    assert seen["max"] == 123
