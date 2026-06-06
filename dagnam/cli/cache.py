"""Cache command handlers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import TYPE_CHECKING

from dagnam.cli.common import add_collection_output_args, dir_size, human_size
from dagnam.cli.presentation import Column, emit_result, render_table

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction


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


def register_cache(subparsers: SubParsersAction) -> None:
    """Register the ``cache`` command group on the top-level subparsers."""
    cache = subparsers.add_parser(
        "cache",
        help="Inspect and clear the local dataset cache.",
        description="Manage the on-disk dataset cache.",
    )
    cache_sub = cache.add_subparsers(dest="cache_command", required=True)
    cache_list = cache_sub.add_parser(
        "list", help="List cached datasets.", description="List datasets in the local cache."
    )
    add_collection_output_args(cache_list)
    cache_list.set_defaults(func=cmd_cache_list)
    cache_clear = cache_sub.add_parser(
        "clear",
        help="Delete the local cache immediately.",
        description="Permanently remove cached datasets immediately unless --dry-run is used.",
    )
    cache_clear.add_argument("--dataset-id", help="Clear only one cached dataset ID.")
    cache_clear.add_argument(
        "--dry-run", action="store_true", help="Show what would be deleted without deleting it."
    )
    cache_clear.set_defaults(func=cmd_cache_clear)
