"""Training and checkpoint command handlers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys

from dagnam.cli.common import error, human_size


def _numeric_json_value(value: object, default: int = 0) -> int | float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return value
    return default


def cmd_checkpoint_list(args: argparse.Namespace) -> None:
    from dagnam._core.auth import get_api_key, get_api_url
    from dagnam._core.client import DagnamClient
    from dagnam._core.exceptions import DagnamError

    try:
        client = DagnamClient(get_api_url(), get_api_key())
        checkpoints = client.list_checkpoints(args.job_id)
    except DagnamError as exc:
        error(str(exc))

    if not checkpoints:
        print("No checkpoints found.")
        return
    header = f"{'ID':<40} {'Epoch':>6} {'Step':>8} {'Best':<6} {'Final':<6} {'Size':>10}"
    print(header)
    print("-" * len(header))
    for cp in checkpoints:
        print(
            f"{cp.get('id', 'N/A'):<40} "
            f"{cp.get('epoch', 0):>6} "
            f"{cp.get('step', 0):>8} "
            f"{cp.get('is_best', False)!s:<6} "
            f"{cp.get('is_final', False)!s:<6} "
            f"{human_size(_numeric_json_value(cp.get('file_size'))):>10}"
        )


def cmd_checkpoint_download(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        path = dagnam.download_checkpoint(args.job_id, args.checkpoint_id)
    except DagnamError as exc:
        error(str(exc))
    print(str(path))


def cmd_stream(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        for ev in dagnam.stream_training(
            args.job_id,
            include_heartbeats=args.heartbeats,
        ):
            if args.json:
                print(json.dumps(asdict(ev)))
            else:
                print(f"[{ev.event}] {ev.data}")
    except DagnamError as exc:
        error(str(exc))
    except KeyboardInterrupt:
        sys.exit(130)


def cmd_training_attach(args: argparse.Namespace) -> None:
    """Attach a local metrics JSONL file or child process to a Dagnam job."""
    from dagnam._core.exceptions import AuthError, DagnamError
    from dagnam.training_attach import run_training_attach

    try:
        code = run_training_attach(
            job_id=args.job_id,
            metrics_path=args.metrics_path,
            command=args.command,
            replay=args.replay,
        )
    except AuthError:
        error("Not logged in. Run 'dagnam login'.")
    except FileNotFoundError as exc:
        error(str(exc))
    except DagnamError as exc:
        error(str(exc))
    sys.exit(code)
