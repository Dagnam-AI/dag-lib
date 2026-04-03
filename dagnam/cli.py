"""Command-line interface for the dagnam client library.

Usage:
    dagnam login                     Save API key to ~/.dagnam/config.json
    dagnam dataset list              List available datasets
    dagnam dataset download <id>     Download a dataset to local cache
    dagnam dataset info <id>         Show dataset metadata
    dagnam cache list                List cached datasets with sizes
    dagnam cache clear               Delete all cached datasets
"""

from __future__ import annotations

import argparse
import getpass
import json
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _human_size(nbytes: int | float) -> str:
    """Format byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def _dir_size(path: Path) -> int:
    """Recursively compute total size of a directory in bytes."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _error(msg: str) -> None:
    """Print an error message to stderr and exit."""
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _cmd_login(args: argparse.Namespace) -> None:
    from dagnam import config as _cfg
    from dagnam.client import DagnamClient
    from dagnam.exceptions import DagnamError

    api_key = getpass.getpass("API key: ")
    if not api_key.strip():
        _error("API key cannot be empty.")

    api_url = getattr(args, "api_url", None) or "https://api.dagnam.ai"

    # Validate by hitting the meta endpoint for a quick test
    client = DagnamClient(api_url, api_key)
    try:
        client.list_datasets()
    except DagnamError as exc:
        _error(f"Authentication failed: {exc}")

    # Persist
    _cfg.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config: dict = {}
    if _cfg.CONFIG_FILE.exists():
        try:
            config = json.loads(_cfg.CONFIG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            config = {}

    config["api_key"] = api_key
    if api_url != "https://api.dagnam.ai":
        config["api_url"] = api_url

    _cfg.CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Credentials saved to {_cfg.CONFIG_FILE}")


def _cmd_dataset_list(args: argparse.Namespace) -> None:
    from dagnam.auth import get_api_key, get_api_url
    from dagnam.client import DagnamClient
    from dagnam.exceptions import DagnamError

    try:
        api_key = get_api_key()
        api_url = get_api_url()
    except DagnamError as exc:
        _error(str(exc))

    client = DagnamClient(api_url, api_key)
    try:
        datasets = client.list_datasets()
    except DagnamError as exc:
        _error(str(exc))

    if not datasets:
        print("No datasets found.")
        return

    # Print formatted table
    header = f"{'ID':<40} {'Name':<25} {'Format':<8} {'Samples':>8} {'Type':<12}"
    print(header)
    print("-" * len(header))
    for ds in datasets:
        print(
            f"{ds.get('id', 'N/A'):<40} "
            f"{ds.get('name', 'N/A'):<25} "
            f"{ds.get('format', 'N/A'):<8} "
            f"{ds.get('num_samples', 'N/A'):>8} "
            f"{ds.get('dataset_type', 'N/A'):<12}"
        )


def _cmd_dataset_download(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam.cache import DEFAULT_CACHE_DIR
    from dagnam.exceptions import DagnamError

    dataset_id: str = args.dataset_id
    try:
        dagnam.load_dataset(dataset_id)
    except DagnamError as exc:
        _error(str(exc))

    print(f"Dataset '{dataset_id}' downloaded to {DEFAULT_CACHE_DIR / dataset_id}")


def _cmd_dataset_info(args: argparse.Namespace) -> None:
    from dagnam.auth import get_api_key, get_api_url
    from dagnam.client import DagnamClient
    from dagnam.exceptions import DagnamError

    dataset_id: str = args.dataset_id

    try:
        api_key = get_api_key()
        api_url = get_api_url()
    except DagnamError as exc:
        _error(str(exc))

    client = DagnamClient(api_url, api_key)
    try:
        meta = client.get_dataset_meta(dataset_id)
    except DagnamError as exc:
        _error(str(exc))

    for key, value in meta.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, list):
            print(f"{key}: {', '.join(str(v) for v in value)}")
        else:
            print(f"{key}: {value}")


def _cmd_cache_list(_args: argparse.Namespace) -> None:
    from dagnam import cache as _cache

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
    from dagnam import cache as _cache

    if not _cache.DEFAULT_CACHE_DIR.exists():
        print("Cache is already empty.")
        return

    total = _dir_size(_cache.DEFAULT_CACHE_DIR)
    shutil.rmtree(_cache.DEFAULT_CACHE_DIR)
    print(f"Cleared cache. Freed {_human_size(total)}.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dagnam",
        description="Dagnam.AI CLI — manage datasets and local cache.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- login ---
    login_parser = subparsers.add_parser("login", help="Save API key")
    login_parser.add_argument(
        "--api-url",
        default=None,
        help="Custom API URL (default: https://api.dagnam.ai)",
    )

    # --- dataset ---
    dataset_parser = subparsers.add_parser("dataset", help="Dataset operations")
    ds_sub = dataset_parser.add_subparsers(dest="dataset_command")

    ds_sub.add_parser("list", help="List available datasets")

    dl_parser = ds_sub.add_parser("download", help="Download a dataset")
    dl_parser.add_argument("dataset_id", help="Dataset ID to download")

    info_parser = ds_sub.add_parser("info", help="Show dataset metadata")
    info_parser.add_argument("dataset_id", help="Dataset ID to inspect")

    # --- cache ---
    cache_parser = subparsers.add_parser("cache", help="Local cache operations")
    cache_sub = cache_parser.add_subparsers(dest="cache_command")

    cache_sub.add_parser("list", help="List cached datasets")
    cache_sub.add_parser("clear", help="Delete all cached datasets")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(2)

    if args.command == "login":
        _cmd_login(args)
    elif args.command == "dataset":
        if getattr(args, "dataset_command", None) is None:
            parser.parse_args(["dataset", "--help"])
        elif args.dataset_command == "list":
            _cmd_dataset_list(args)
        elif args.dataset_command == "download":
            _cmd_dataset_download(args)
        elif args.dataset_command == "info":
            _cmd_dataset_info(args)
    elif args.command == "cache":
        if getattr(args, "cache_command", None) is None:
            parser.parse_args(["cache", "--help"])
        elif args.cache_command == "list":
            _cmd_cache_list(args)
        elif args.cache_command == "clear":
            _cmd_cache_clear(args)
