"""Cache command handlers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from dagnam.cli.common import dir_size, human_size
from dagnam.cli.presentation import Column, emit_result, render_table


def _render_cache(entries: object) -> str:
    rows = entries if isinstance(entries, list) else []
    if not rows:
        return "Cache is empty."
    return render_table(
        (
            Column("ID", "id", 40),
            Column("Name", "name", 25),
            Column("Size", "display_size", 10, "right"),
        ),
        rows,
    )


def cmd_cache_list(args: argparse.Namespace) -> None:
    from dagnam.data import cache as _cache

    if not _cache.DEFAULT_CACHE_DIR.exists():
        emit_result(
            [],
            output=args.output,
            json_stdout=args.json or args.verbose,
            render_human=_render_cache,
        )
        return

    entries: list[dict[str, object]] = []
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
        size = dir_size(child)
        entries.append(
            {"id": dataset_id, "name": name, "size": size, "display_size": human_size(size)}
        )

    emit_result(
        entries,
        output=args.output,
        json_stdout=args.json or args.verbose,
        render_human=_render_cache,
    )


def _cache_target(base: Path, dataset_id: str | None) -> Path:
    if not dataset_id:
        return base
    from dagnam.data.cache import cache_dir_name

    return base / cache_dir_name(dataset_id)


def cmd_cache_clear(args: argparse.Namespace) -> None:
    from dagnam.data import cache as _cache

    target = _cache_target(_cache.DEFAULT_CACHE_DIR, args.dataset_id)
    label = f"dataset cache {args.dataset_id}" if args.dataset_id else "cache"

    if not target.exists():
        print("Cache is already empty.")
        return

    total = dir_size(target)
    if args.dry_run:
        print(f"Would clear {label}. Would free {human_size(total)}.")
        return

    shutil.rmtree(target)
    print(f"Cleared {label}. Freed {human_size(total)}.")
