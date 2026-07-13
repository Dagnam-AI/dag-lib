# Cache reference

Datasets and checkpoints are cached on disk so repeat runs are fast and offline-friendly.

## CLI (`dagnam cache ...`)
- `dagnam cache list` (+ `--json/--verbose/--output`) — list cached datasets and their sizes.
- `dagnam cache clear [--dataset-id <id>] [--dry-run]` — remove cached datasets (all, or one); `--dry-run` only reports what would be removed. **[guardrail: irreversible — but local-only]**

## SDK (`import dagnam`)
- `dagnam.get_cache_dir() -> Path` — the active cache root.
- `dagnam.is_cached(dataset_id) -> bool`, `dagnam.touch_cache(...)`, `dagnam.evict_lru(...)` — cache inspection / maintenance.
- `dagnam.save_metadata(...)`, `dagnam.load_metadata(...)`, `dagnam.compute_file_checksum(...)`, `dagnam.save_checksum(...)` — lower-level helpers.

## Config
- Set `DAGNAM_CACHE_DIR` to relocate the cache root (otherwise it defaults under `~/.dagnam`).
- `load_dataset` reads from the cache transparently; delete entries with `dagnam cache clear` if disk is tight.
- `max_download_bytes` (`~/.dagnam/config.json`, integer bytes, default 100 GiB) caps every on-disk download — datasets, checkpoints, and system-dataset artifacts. A download whose size exceeds the cap is refused up-front (or aborted mid-stream) with `DownloadTooLargeError`, and the partial file is deleted, so a hostile or misconfigured server cannot fill your disk. A non-integer or non-positive value falls back to the default.

## Cache trust boundary
The cache directory is **trusted**: a cache hit loads the stored file without re-hashing it (for speed). Keep the cache root private — the default under `~/.dagnam` is user-private. Do **not** point `DAGNAM_CACHE_DIR` (or a `base_dir`) at a shared/world-writable location, where another local user could swap a cached file for a same-size payload; the SDK warns once if it detects a group/world-writable cache root. If you must use a shared cache, pass `verify=True` to force a full checksum re-verification on every load.
