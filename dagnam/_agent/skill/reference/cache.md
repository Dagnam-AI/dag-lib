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
