"""Inference command handlers."""

from __future__ import annotations

import argparse
import json

from dagnam._types import ensure_json_array, ensure_json_object
from dagnam.cli.common import error, load_json_arg, print_json, write_json_file


def cmd_inference_run(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        source = f"@{args.input_file}" if args.input_file else args.input
        payload = ensure_json_object(load_json_arg(source))
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        error(f"Failed to parse --input: {exc}")

    try:
        result = dagnam.inference(args.deployment_id, payload)
    except DagnamError as exc:
        error(str(exc))
    if args.output:
        write_json_file(args.output, result)
    print_json(result)


def cmd_inference_batch(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        source = f"@{args.inputs_file}" if args.inputs_file else args.inputs
        payload = ensure_json_array(load_json_arg(source))
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        error(f"Failed to parse --inputs: {exc}")

    try:
        result = dagnam.inference_batch(args.deployment_id, payload)
    except DagnamError as exc:
        error(str(exc))
    if args.output:
        write_json_file(args.output, result)
    print_json(result)


def cmd_inference_health(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.deployments.health(args.deployment_id)
    except DagnamError as exc:
        error(str(exc))
    if args.output:
        write_json_file(args.output, result)
    print_json(result)
