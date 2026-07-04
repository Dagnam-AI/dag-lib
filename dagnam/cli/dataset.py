"""Dataset command handlers."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import TYPE_CHECKING

from dagnam._types import JsonObject
from dagnam.cli.common import add_collection_output_args
from dagnam.cli.presentation import Column, emit_result, render_table

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction


def _render_datasets(result: object) -> str:
    datasets = result if isinstance(result, list) else []
    if not datasets:
        return "No datasets found."
    rows = [dataset for dataset in datasets if isinstance(dataset, dict)]
    return render_table(
        (
            Column("ID", "id", 40),
            Column("Name", "name", 25),
            Column("Format", "format", 14),
            Column("Samples", "num_samples", 10, "right"),
            Column("Type", "dataset_type", 12),
        ),
        rows,
    )


def _redact_dataset_meta(meta: JsonObject, *, show_download_url: bool) -> JsonObject:
    result = deepcopy(meta)
    if "download_url" in result and not show_download_url:
        result["download_url"] = "<redacted>"
    return result


def cmd_dataset_list(args: argparse.Namespace) -> None:
    from dagnam._core.auth import get_api_key, get_api_url
    from dagnam._core.client import DagnamClient

    api_key = args.api_key or get_api_key()
    api_url = args.api_url or get_api_url()

    client = DagnamClient(api_url, api_key)
    datasets = client.list_datasets(type=args.type, search=args.search)

    emit_result(
        datasets,
        output=args.output,
        json_stdout=args.json or args.verbose,
        render_human=_render_datasets,
    )


def cmd_dataset_download(args: argparse.Namespace) -> None:
    import dagnam

    dataset_id: str = args.dataset_id
    output_dir = Path(args.output_dir)
    show_progress = not (args.no_progress or not sys.stderr.isatty())
    dagnam.load_dataset(dataset_id, cache_dir=str(output_dir), show_progress=show_progress)

    print(f"Dataset '{dataset_id}' downloaded to {output_dir / dataset_id}")


def cmd_dataset_info(args: argparse.Namespace) -> None:
    from dagnam._core.auth import get_api_key, get_api_url
    from dagnam._core.client import DagnamClient

    dataset_id: str = args.dataset_id

    api_key = get_api_key()
    api_url = get_api_url()

    client = DagnamClient(api_url, api_key)
    meta = client.get_dataset_meta(dataset_id)

    safe_meta = _redact_dataset_meta(meta, show_download_url=args.show_download_url)
    if args.output:
        from dagnam.cli.common import write_json_file

        write_json_file(args.output, safe_meta)
    if args.json:
        from dagnam.cli.common import print_json

        print_json(safe_meta)
        return
    for key, value in safe_meta.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, list):
            print(f"{key}: {', '.join(str(v) for v in value)}")
        else:
            print(f"{key}: {value}")


def register_dataset(subparsers: SubParsersAction) -> None:
    """Register the ``dataset`` command group on the top-level subparsers."""
    dataset = subparsers.add_parser(
        "dataset",
        help="Browse and download datasets.",
        description="List, inspect, and download datasets from Dagnam.AI.",
    )
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_list = dataset_sub.add_parser(
        "list", help="List datasets.", description="List available datasets."
    )
    dataset_list.add_argument("--api-url", help="Override the API base URL.")
    dataset_list.add_argument("--api-key", help="Override the API key.")
    dataset_list.add_argument(
        "--type",
        default="all",
        help="Filter by dataset type: image, text, audio, video, tabular, custom, or all.",
    )
    dataset_list.add_argument("--search", help="Filter by name/description substring.")
    add_collection_output_args(dataset_list)
    dataset_list.set_defaults(func=cmd_dataset_list)
    dataset_info = dataset_sub.add_parser(
        "info", help="Show dataset metadata.", description="Show metadata for one dataset."
    )
    dataset_info.add_argument("dataset_id", help="ID of the dataset to inspect.")
    dataset_info.add_argument("--api-url", help="Override the API base URL.")
    dataset_info.add_argument("--api-key", help="Override the API key.")
    dataset_info.add_argument(
        "--show-download-url",
        action="store_true",
        help="Include signed download_url values in output. By default these are redacted.",
    )
    dataset_info.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    dataset_info.add_argument("--output", help="Write the redacted JSON response to this path.")
    dataset_info.set_defaults(func=cmd_dataset_info)
    dataset_download = dataset_sub.add_parser(
        "download",
        help="Download a dataset.",
        description="Download a dataset to a local directory.",
    )
    dataset_download.add_argument("dataset_id", help="ID of the dataset to download.")
    dataset_download.add_argument(
        "--output-dir", default=".", help="Destination directory (default: current dir)."
    )
    dataset_download.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable download progress output. Progress is also hidden when stderr is redirected.",
    )
    dataset_download.set_defaults(func=cmd_dataset_download)
