"""Training and checkpoint command handlers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys

from dagnam.cli.common import _error, _human_size


def _cmd_checkpoint_list(args: argparse.Namespace) -> None:
    from dagnam._core.auth import get_api_key, get_api_url
    from dagnam._core.client import DagnamClient
    from dagnam._core.exceptions import DagnamError

    try:
        client = DagnamClient(get_api_url(), get_api_key())
        checkpoints = client.list_checkpoints(args.job_id)
    except DagnamError as exc:
        _error(str(exc))

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
            f"{_human_size(cp.get('file_size') or 0):>10}"
        )


def _cmd_checkpoint_download(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        path = dagnam.download_checkpoint(args.job_id, args.checkpoint_id)
    except DagnamError as exc:
        _error(str(exc))
    print(str(path))


def _cmd_stream(args: argparse.Namespace) -> None:
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
        _error(str(exc))
    except KeyboardInterrupt:
        sys.exit(130)
