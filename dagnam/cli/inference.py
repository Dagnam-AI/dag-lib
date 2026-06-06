"""Inference command handlers."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

from dagnam._types import ensure_json_array, ensure_json_object
from dagnam.cli.common import error, load_json_arg, print_json, write_json_file

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction


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


def register_inference(subparsers: SubParsersAction) -> None:
    """Register the ``inference`` command group on the top-level subparsers."""
    inference = subparsers.add_parser(
        "inference",
        help="Run inference against a deployment.",
        description="Run single or batch inference and check deployment health.",
    )
    inference_sub = inference.add_subparsers(dest="inference_command", required=True)
    run = inference_sub.add_parser(
        "run", help="Run one inference.", description="Send one input to a deployment."
    )
    run.add_argument("deployment_id", help="ID of the deployment.")
    run_input = run.add_mutually_exclusive_group(required=True)
    run_input.add_argument("--input", help="JSON literal, or @path to a JSON file.")
    run_input.add_argument("--input-file", help="Path to a JSON object file.")
    run.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    run.add_argument("--output", help="Write the JSON response to this path.")
    run.set_defaults(func=cmd_inference_run)
    batch = inference_sub.add_parser(
        "batch", help="Run batch inference.", description="Send multiple inputs to a deployment."
    )
    batch.add_argument("deployment_id", help="ID of the deployment.")
    batch_input = batch.add_mutually_exclusive_group(required=True)
    batch_input.add_argument("--inputs", help="JSON array literal, or @path to a JSON file.")
    batch_input.add_argument("--inputs-file", help="Path to a JSON array file.")
    batch.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    batch.add_argument("--output", help="Write the JSON response to this path.")
    batch.set_defaults(func=cmd_inference_batch)
    health = inference_sub.add_parser(
        "health", help="Check deployment health.", description="Report deployment health status."
    )
    health.add_argument("deployment_id", help="ID of the deployment.")
    health.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    health.add_argument("--output", help="Write the JSON response to this path.")
    health.set_defaults(func=cmd_inference_health)
