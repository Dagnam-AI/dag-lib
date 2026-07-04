"""Checkpoint command handlers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from dagnam.cli.common import add_collection_output_args, human_size
from dagnam.cli.presentation import Column, emit_result, render_table

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction


def _numeric_json_value(value: object, default: int = 0) -> int | float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return value
    return default


def cmd_checkpoint_list(args: argparse.Namespace) -> None:
    from dagnam._core.auth import get_api_key, get_api_url
    from dagnam._core.client import DagnamClient

    client = DagnamClient(get_api_url(), get_api_key())
    checkpoints = client.list_checkpoints(args.job_id)

    emit_result(
        checkpoints,
        output=args.output,
        json_stdout=args.json or args.verbose,
        render_human=_render_checkpoints,
    )


def _render_checkpoints(result: object) -> str:
    checkpoints = result if isinstance(result, list) else []
    if not checkpoints:
        return "No checkpoints found."
    rows = [
        {
            **checkpoint,
            "size": human_size(_numeric_json_value(checkpoint.get("file_size"))),
        }
        for checkpoint in checkpoints
        if isinstance(checkpoint, dict)
    ]
    return render_table(
        (
            Column("ID", "id", 40),
            Column("Epoch", "epoch", 6, "right"),
            Column("Step", "step", 8, "right"),
            Column("Best", "is_best", 6),
            Column("Final", "is_final", 6),
            Column("Size", "size", 10, "right"),
        ),
        rows,
    )


def cmd_checkpoint_download(args: argparse.Namespace) -> None:
    import dagnam

    checkpoint_id = None if args.checkpoint_id in (None, "latest", "best") else args.checkpoint_id
    prefer_best = args.checkpoint_id == "best"
    if args.output_dir:
        cache_dir = Path(args.output_dir)
        if prefer_best:
            path = dagnam.download_checkpoint(
                args.job_id, checkpoint_id, cache_dir=cache_dir, prefer_best=True
            )
        else:
            path = dagnam.download_checkpoint(args.job_id, checkpoint_id, cache_dir=cache_dir)
    elif prefer_best:
        path = dagnam.download_checkpoint(args.job_id, checkpoint_id, prefer_best=True)
    else:
        path = dagnam.download_checkpoint(args.job_id, checkpoint_id)
    print(str(path))


def register_checkpoint(subparsers: SubParsersAction) -> None:
    """Register the ``checkpoint`` command group on the top-level subparsers."""
    checkpoint = subparsers.add_parser(
        "checkpoint",
        help="List and download training checkpoints.",
        description="List or download checkpoints for a training job.",
    )
    checkpoint_sub = checkpoint.add_subparsers(dest="checkpoint_command", required=True)
    checkpoint_list = checkpoint_sub.add_parser(
        "list", help="List checkpoints.", description="List checkpoints for a job."
    )
    checkpoint_list.add_argument("job_id", help="ID of the training job.")
    add_collection_output_args(checkpoint_list)
    checkpoint_list.set_defaults(func=cmd_checkpoint_list)
    checkpoint_download = checkpoint_sub.add_parser(
        "download", help="Download a checkpoint.", description="Download a job checkpoint."
    )
    checkpoint_download.add_argument("job_id", help="ID of the training job.")
    checkpoint_download.add_argument(
        "checkpoint_id",
        nargs="?",
        default=None,
        help="Checkpoint ID (default: latest).",
    )
    checkpoint_download.add_argument(
        "--output-dir", help="Cache the downloaded checkpoint under this directory."
    )
    checkpoint_download.set_defaults(func=cmd_checkpoint_download)
