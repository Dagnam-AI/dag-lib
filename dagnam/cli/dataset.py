"""Dataset command handlers."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

from dagnam.cli.common import error
from dagnam.cli.presentation import Column, emit_result, render_table


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


def _redact_dataset_meta(meta: dict[str, object], *, show_download_url: bool) -> dict[str, object]:
    result = deepcopy(meta)
    if "download_url" in result and not show_download_url:
        result["download_url"] = "<redacted>"
    return result


def cmd_dataset_list(args: argparse.Namespace) -> None:
    from dagnam._core.auth import get_api_key, get_api_url
    from dagnam._core.client import DagnamClient
    from dagnam._core.exceptions import DagnamError

    try:
        api_key = args.api_key or get_api_key()
        api_url = args.api_url or get_api_url()
    except DagnamError as exc:
        error(str(exc))

    client = DagnamClient(api_url, api_key)
    try:
        datasets = client.list_datasets(type=args.type, search=args.search)
    except DagnamError as exc:
        error(str(exc))

    emit_result(
        datasets,
        output=args.output,
        json_stdout=args.json or args.verbose,
        render_human=_render_datasets,
    )


def cmd_dataset_download(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    dataset_id: str = args.dataset_id
    output_dir = Path(args.output_dir)
    show_progress = not (args.no_progress or not sys.stderr.isatty())
    try:
        dagnam.load_dataset(dataset_id, cache_dir=str(output_dir), show_progress=show_progress)
    except DagnamError as exc:
        error(str(exc))

    print(f"Dataset '{dataset_id}' downloaded to {output_dir / dataset_id}")


def cmd_dataset_info(args: argparse.Namespace) -> None:
    from dagnam._core.auth import get_api_key, get_api_url
    from dagnam._core.client import DagnamClient
    from dagnam._core.exceptions import DagnamError

    dataset_id: str = args.dataset_id

    try:
        api_key = get_api_key()
        api_url = get_api_url()
    except DagnamError as exc:
        error(str(exc))

    client = DagnamClient(api_url, api_key)
    try:
        meta = client.get_dataset_meta(dataset_id)
    except DagnamError as exc:
        error(str(exc))

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
