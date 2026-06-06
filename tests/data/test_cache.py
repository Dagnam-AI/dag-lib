"""Unit tests for dagnam.cache module."""

import hashlib
import json
from pathlib import Path
import time

import pytest
from tests.typing_helpers import JsonObject, PytestMonkeyPatch

from dagnam.data import cache as cache_module
from dagnam.data.cache import (
    DEFAULT_MAX_CACHE_BYTES,
    CacheInfo,
    cache_dir_name,
    compute_file_checksum,
    evict_lru,
    get_cache_dir,
    get_cache_info,
    get_cache_size,
    is_cached,
    load_metadata,
    save_checksum,
    save_metadata,
    touch_cache,
)


class TestGetCacheDir:
    def test_default_base_dir(self, cache_dir: Path) -> None:
        result = get_cache_dir("ds-123", base_dir=cache_dir)
        assert result == cache_dir / "ds-123"
        assert result.is_dir()

    def test_creates_nested_parents(self, tmp_path: Path) -> None:
        base = tmp_path / "a" / "b" / "c"
        result = get_cache_dir("ds-456", base_dir=base)
        assert result == base / "ds-456"
        assert result.is_dir()

    def test_custom_base_dir(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom_cache"
        result = get_cache_dir("my-dataset", base_dir=custom)
        assert result == custom / "my-dataset"
        assert result.is_dir()

    def test_idempotent(self, cache_dir: Path) -> None:
        r1 = get_cache_dir("ds-1", base_dir=cache_dir)
        r2 = get_cache_dir("ds-1", base_dir=cache_dir)
        assert r1 == r2

    def test_dataset_id_cannot_escape_cache_dir(self, tmp_path: Path) -> None:
        base = tmp_path / "cache"
        result = get_cache_dir("../escape", base_dir=base)

        assert result.resolve().is_relative_to(base.resolve())
        assert result.name == "..%2Fescape"
        assert not (tmp_path / "escape").exists()


class TestIsCached:
    def test_no_checksum_file(self, cache_dir: Path) -> None:
        assert is_cached("ds-1", "abc123", base_dir=cache_dir) is False

    def test_matching_checksum(self, cache_dir: Path) -> None:
        ds_dir = cache_dir / "ds-1"
        ds_dir.mkdir(parents=True)
        (ds_dir / ".checksum").write_text("abc123")
        assert is_cached("ds-1", "abc123", base_dir=cache_dir) is True

    def test_mismatched_checksum(self, cache_dir: Path) -> None:
        ds_dir = cache_dir / "ds-1"
        ds_dir.mkdir(parents=True)
        (ds_dir / ".checksum").write_text("abc123")
        assert is_cached("ds-1", "different", base_dir=cache_dir) is False

    def test_strips_whitespace(self, cache_dir: Path) -> None:
        ds_dir = cache_dir / "ds-1"
        ds_dir.mkdir(parents=True)
        (ds_dir / ".checksum").write_text("  abc123  \n")
        assert is_cached("ds-1", "abc123", base_dir=cache_dir) is True

    def test_cache_hit_updates_last_access(self, cache_dir: Path) -> None:
        ds_dir = cache_dir / "ds-1"
        ds_dir.mkdir(parents=True)
        (ds_dir / ".checksum").write_text("abc123")
        assert is_cached("ds-1", "abc123", base_dir=cache_dir) is True
        access_file = ds_dir / ".last_access"
        assert access_file.exists()
        ts = float(access_file.read_text(encoding="utf-8").strip())
        assert abs(ts - time.time()) < 2

    def test_cache_miss_no_last_access(self, cache_dir: Path) -> None:
        ds_dir = cache_dir / "ds-1"
        ds_dir.mkdir(parents=True)
        (ds_dir / ".checksum").write_text("abc123")
        is_cached("ds-1", "wrong", base_dir=cache_dir)
        assert not (ds_dir / ".last_access").exists()


class TestSaveMetadata:
    def test_writes_json(self, cache_dir: Path, sample_metadata: JsonObject) -> None:
        save_metadata("ds-1", sample_metadata, base_dir=cache_dir)
        meta_file = cache_dir / "ds-1" / "meta.json"
        assert meta_file.exists()
        loaded = json.loads(meta_file.read_text())
        assert loaded == sample_metadata

    def test_indent_2(self, cache_dir: Path) -> None:
        save_metadata("ds-1", {"key": "value"}, base_dir=cache_dir)
        content = (cache_dir / "ds-1" / "meta.json").read_text()
        assert "  " in content  # indent=2


class TestLoadMetadata:
    def test_returns_empty_dict_if_missing(self, cache_dir: Path) -> None:
        result = load_metadata("nonexistent", base_dir=cache_dir)
        assert result == {}

    def test_loads_saved_metadata(self, cache_dir: Path, sample_metadata: JsonObject) -> None:
        save_metadata("ds-1", sample_metadata, base_dir=cache_dir)
        result = load_metadata("ds-1", base_dir=cache_dir)
        assert result == sample_metadata


class TestSaveChecksum:
    def test_writes_checksum_file(self, cache_dir: Path) -> None:
        save_checksum("ds-1", "sha256:abc123", base_dir=cache_dir)
        content = (cache_dir / "ds-1" / ".checksum").read_text()
        assert content == "sha256:abc123"


class TestComputeFileChecksum:
    def test_sha256_small_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.bin"
        data = b"hello world"
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert compute_file_checksum(f) == expected

    def test_sha256_large_file(self, tmp_path: Path) -> None:
        f = tmp_path / "large.bin"
        data = b"x" * 50_000  # larger than 8KB chunk
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert compute_file_checksum(f) == expected

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert compute_file_checksum(f) == expected


class TestTouchCache:
    def test_creates_last_access_file(self, cache_dir: Path) -> None:
        touch_cache("ds-1", base_dir=cache_dir)
        access_file = cache_dir / "ds-1" / ".last_access"
        assert access_file.exists()
        ts = float(access_file.read_text(encoding="utf-8").strip())
        assert abs(ts - time.time()) < 2

    def test_overwrites_previous_timestamp(self, cache_dir: Path) -> None:
        touch_cache("ds-1", base_dir=cache_dir)
        access_file = cache_dir / "ds-1" / ".last_access"
        first_ts = float(access_file.read_text(encoding="utf-8").strip())
        time.sleep(0.05)
        touch_cache("ds-1", base_dir=cache_dir)
        second_ts = float(access_file.read_text(encoding="utf-8").strip())
        assert second_ts > first_ts


class TestGetCacheSize:
    def test_empty_cache(self, cache_dir: Path) -> None:
        assert get_cache_size(base_dir=cache_dir) == 0

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        assert get_cache_size(base_dir=tmp_path / "nope") == 0

    def test_counts_all_files(self, cache_dir: Path) -> None:
        ds_dir = cache_dir / "ds-1"
        ds_dir.mkdir()
        (ds_dir / "data.csv").write_bytes(b"x" * 100)
        (ds_dir / "meta.json").write_bytes(b"y" * 50)
        assert get_cache_size(base_dir=cache_dir) == 150

    def test_multiple_datasets(self, cache_dir: Path) -> None:
        for name, size in [("ds-a", 200), ("ds-b", 300)]:
            d = cache_dir / name
            d.mkdir()
            (d / "data.bin").write_bytes(b"z" * size)
        assert get_cache_size(base_dir=cache_dir) == 500


class TestGetCacheInfo:
    def test_empty_cache(self, cache_dir: Path) -> None:
        assert get_cache_info(base_dir=cache_dir) == []

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        assert get_cache_info(base_dir=tmp_path / "nope") == []

    def test_returns_dataset_entries(self, cache_dir: Path) -> None:
        ds_dir = cache_dir / "ds-1"
        ds_dir.mkdir()
        (ds_dir / "data.csv").write_bytes(b"x" * 100)
        touch_cache("ds-1", base_dir=cache_dir)

        entries = get_cache_info(base_dir=cache_dir)
        assert len(entries) == 1
        assert entries[0]["dataset_id"] == "ds-1"
        assert entries[0]["size_bytes"] > 0
        assert entries[0]["last_access"] is not None

    def test_no_last_access_returns_none(self, cache_dir: Path) -> None:
        ds_dir = cache_dir / "ds-1"
        ds_dir.mkdir()
        (ds_dir / "data.csv").write_bytes(b"x" * 10)

        entries = get_cache_info(base_dir=cache_dir)
        assert entries[0]["last_access"] is None

    def test_ignores_non_directory_children(self, cache_dir: Path) -> None:
        (cache_dir / "stray_file.txt").write_text("oops")
        ds_dir = cache_dir / "ds-1"
        ds_dir.mkdir()
        (ds_dir / "data.csv").write_bytes(b"x" * 10)

        entries = get_cache_info(base_dir=cache_dir)
        assert len(entries) == 1
        assert entries[0]["dataset_id"] == "ds-1"

    def test_malformed_last_access_returns_none(self, cache_dir: Path) -> None:
        ds_dir = cache_dir / "ds-1"
        ds_dir.mkdir()
        (ds_dir / "data.csv").write_bytes(b"x" * 10)
        (ds_dir / ".last_access").write_text("not-a-number")

        entries = get_cache_info(base_dir=cache_dir)
        assert entries[0]["last_access"] is None


class TestEvictLru:
    def _make_dataset(
        self, cache_dir: Path, name: str, size: int, last_access: float | None = None
    ) -> None:
        """Helper to create a fake cached dataset."""
        ds_dir = cache_dir / name
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / "data.bin").write_bytes(b"x" * size)
        if last_access is not None:
            (ds_dir / ".last_access").write_text(str(last_access), encoding="utf-8")

    def test_no_eviction_when_under_limit(self, cache_dir: Path) -> None:
        self._make_dataset(cache_dir, "ds-1", 100, last_access=1000.0)
        evicted = evict_lru(max_size_bytes=1000, base_dir=cache_dir)
        assert evicted == []
        assert (cache_dir / "ds-1").exists()

    def test_evicts_oldest_first(self, cache_dir: Path) -> None:
        self._make_dataset(cache_dir, "old", 100, last_access=1000.0)
        self._make_dataset(cache_dir, "new", 100, last_access=2000.0)
        # Total = 200, limit = 150 → must evict one
        evicted = evict_lru(max_size_bytes=150, base_dir=cache_dir)
        assert "old" in evicted
        assert "new" not in evicted
        assert not (cache_dir / "old").exists()
        assert (cache_dir / "new").exists()

    def test_evicts_multiple_until_under_limit(self, cache_dir: Path) -> None:
        self._make_dataset(cache_dir, "a", 100, last_access=1000.0)
        self._make_dataset(cache_dir, "b", 100, last_access=2000.0)
        self._make_dataset(cache_dir, "c", 100, last_access=3000.0)
        total = get_cache_size(base_dir=cache_dir)
        # Set limit so that only 'c' can remain
        single_ds_size = total // 3 + 1
        evicted = evict_lru(max_size_bytes=single_ds_size, base_dir=cache_dir)
        assert "a" in evicted
        assert "b" in evicted
        assert "c" not in evicted

    def test_no_last_access_treated_as_oldest(self, cache_dir: Path) -> None:
        self._make_dataset(cache_dir, "no-access", 100)  # no .last_access
        self._make_dataset(cache_dir, "recent", 100, last_access=9999.0)
        evicted = evict_lru(max_size_bytes=150, base_dir=cache_dir)
        assert evicted == ["no-access"]
        assert not (cache_dir / "no-access").exists()
        assert (cache_dir / "recent").exists()

    def test_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        evicted = evict_lru(max_size_bytes=100, base_dir=tmp_path / "nope")
        assert evicted == []

    def test_empty_cache_returns_empty(self, cache_dir: Path) -> None:
        evicted = evict_lru(max_size_bytes=0, base_dir=cache_dir)
        assert evicted == []

    def test_evicts_all_if_limit_zero(self, cache_dir: Path) -> None:
        self._make_dataset(cache_dir, "ds-1", 50, last_access=1000.0)
        self._make_dataset(cache_dir, "ds-2", 50, last_access=2000.0)
        evicted = evict_lru(max_size_bytes=0, base_dir=cache_dir)
        assert len(evicted) == 2
        assert not (cache_dir / "ds-1").exists()
        assert not (cache_dir / "ds-2").exists()

    def test_stops_evicting_once_under_limit(self, cache_dir: Path) -> None:
        self._make_dataset(cache_dir, "a", 100, last_access=1000.0)
        self._make_dataset(cache_dir, "b", 100, last_access=2000.0)
        self._make_dataset(cache_dir, "c", 100, last_access=3000.0)
        # Total = 300, limit = 250 → evict only 'a'
        evicted = evict_lru(max_size_bytes=250, base_dir=cache_dir)
        assert evicted == ["a"]
        assert (cache_dir / "b").exists()
        assert (cache_dir / "c").exists()

    def test_default_max_size_uses_configured_value(
        self, cache_dir: Path, monkeypatch: PytestMonkeyPatch
    ) -> None:
        """When max_size_bytes is None, the configured value drives eviction."""
        self._make_dataset(cache_dir, "old", 100, last_access=1000.0)
        self._make_dataset(cache_dir, "new", 100, last_access=2000.0)

        # Configure a small limit so one dataset must be evicted.
        monkeypatch.setattr(
            "dagnam._core.config.get_config_value",
            lambda _key, _default: 150,
        )
        evicted = evict_lru(base_dir=cache_dir)
        assert evicted == ["old"]

    def test_default_max_size_falls_back_to_constant_when_config_non_int(
        self, cache_dir: Path, monkeypatch: PytestMonkeyPatch
    ) -> None:
        """A non-int configured value falls back to DEFAULT_MAX_CACHE_BYTES."""
        self._make_dataset(cache_dir, "ds", 100, last_access=1000.0)

        monkeypatch.setattr(
            "dagnam._core.config.get_config_value",
            lambda _key, _default: "not-an-int",
        )
        # 100 bytes is far below the 10 GB default, so nothing is evicted.
        evicted = evict_lru(base_dir=cache_dir)
        assert evicted == []
        assert (cache_dir / "ds").exists()

    def test_skips_entry_whose_directory_vanished(
        self, cache_dir: Path, monkeypatch: PytestMonkeyPatch
    ) -> None:
        """A reported cache entry whose directory no longer exists is skipped.

        Simulates a race where ``get_cache_info`` lists an entry but the
        directory is removed before eviction reaches it. The phantom entry
        must not be added to the evicted list, and the real one is evicted.
        """
        self._make_dataset(cache_dir, "real", 100, last_access=2000.0)

        phantom: CacheInfo = {
            "dataset_id": "ghost",
            "size_bytes": 1_000_000,
            "last_access": 1000.0,
        }
        real_get_cache_info = cache_module.get_cache_info

        def fake_info(base_dir: Path | None = None) -> list[CacheInfo]:
            real_entries = [e for e in real_get_cache_info(base_dir) if e["dataset_id"] == "real"]
            return [phantom, *real_entries]

        monkeypatch.setattr(cache_module, "get_cache_info", fake_info)

        evicted = evict_lru(max_size_bytes=50, base_dir=cache_dir)
        assert "ghost" not in evicted
        assert "real" in evicted
        assert not (cache_dir / "real").exists()


class TestDefaultMaxCacheBytes:
    def test_constant_is_10gb(self) -> None:
        assert DEFAULT_MAX_CACHE_BYTES == 10 * 1024 * 1024 * 1024


class TestCacheDirName:
    def test_plain_id_passthrough(self) -> None:
        assert cache_dir_name("ds-123") == "ds-123"

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            cache_dir_name("")

    def test_dot_is_percent_encoded(self) -> None:
        # quote() leaves "." untouched (it is in the safe set), so the
        # explicit "." / ".." guard must replace it to avoid a path component
        # that resolves to the current directory.
        assert cache_dir_name(".") == "%2E"

    def test_dotdot_is_percent_encoded(self) -> None:
        # "../" would escape the cache root; both dots must be encoded.
        assert cache_dir_name("..") == "%2E%2E"

    def test_slash_is_percent_encoded(self) -> None:
        # A path separator must not survive into the directory name.
        assert cache_dir_name("a/b") == "a%2Fb"
