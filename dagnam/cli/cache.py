"""Cache command handlers."""

from __future__ import annotations

import argparse
import json
import shutil

from dagnam.cli.common import _dir_size, _human_size


def _cmd_cache_list(_args: argparse.Namespace) -> None:
    from dagnam.data import cache as _cache

    if not _cache.DEFAULT_CACHE_DIR.exists():
        print("Cache is empty.")
        return

    entries: list[tuple[str, str, int]] = []
    for child in sorted(_cache.DEFAULT_CACHE_DIR.iterdir()):
        if not child.is_dir():
            continue
        dataset_id = child.name
        meta_file = child / "meta.json"
        name = "unknown"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                name = meta.get("name", "unknown")
            except (json.JSONDecodeError, OSError):
                pass
        size = _dir_size(child)
        entries.append((dataset_id, name, size))

    if not entries:
        print("Cache is empty.")
        return

    header = f"{'ID':<40} {'Name':<25} {'Size':>10}"
    print(header)
    print("-" * len(header))
    for dataset_id, name, size in entries:
        print(f"{dataset_id:<40} {name:<25} {_human_size(size):>10}")


def _cmd_cache_clear(_args: argparse.Namespace) -> None:
    from dagnam.data import cache as _cache

    if not _cache.DEFAULT_CACHE_DIR.exists():
        print("Cache is already empty.")
        return

    total = _dir_size(_cache.DEFAULT_CACHE_DIR)
    shutil.rmtree(_cache.DEFAULT_CACHE_DIR)
    print(f"Cleared cache. Freed {_human_size(total)}.")
