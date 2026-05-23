"""Inference command handlers."""

from __future__ import annotations

import argparse
import json

from dagnam._types import ensure_json_array, ensure_json_object
from dagnam.cli.common import error, load_json_arg


def cmd_inference_run(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        payload = ensure_json_object(load_json_arg(args.input))
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        error(f"Failed to parse --input: {exc}")

    try:
        result = dagnam.inference(args.deployment_id, payload)
    except DagnamError as exc:
        error(str(exc))
    print(json.dumps(result, indent=2))


def cmd_inference_batch(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        payload = ensure_json_array(load_json_arg(args.inputs))
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        error(f"Failed to parse --inputs: {exc}")

    try:
        result = dagnam.inference_batch(args.deployment_id, payload)
    except DagnamError as exc:
        error(str(exc))
    print(json.dumps(result, indent=2))


def cmd_inference_health(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.deployment_health(args.deployment_id)
    except DagnamError as exc:
        error(str(exc))
    print(json.dumps(result, indent=2))
