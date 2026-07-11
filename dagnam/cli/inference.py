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

    try:
        source = f"@{args.input_file}" if args.input_file else args.input
        payload = ensure_json_object(load_json_arg(source))
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        error(f"Failed to parse --input: {exc}")

    result = dagnam.inference(args.deployment_id, payload)
    if args.output:
        write_json_file(args.output, result)
    print_json(result)


def cmd_inference_batch(args: argparse.Namespace) -> None:
    import dagnam

    try:
        source = f"@{args.inputs_file}" if args.inputs_file else args.inputs
        payload = ensure_json_array(load_json_arg(source))
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        error(f"Failed to parse --inputs: {exc}")

    result = dagnam.inference_batch(args.deployment_id, payload)
    if args.output:
        write_json_file(args.output, result)
    print_json(result)


def cmd_inference_health(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.deployments.health(args.deployment_id)
    if args.output:
        write_json_file(args.output, result)
    print_json(result)


def cmd_inference_schema(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.inference_schema(args.deployment_id)
    if args.output:
        write_json_file(args.output, result)
    print_json(result)


def cmd_inference_stream(args: argparse.Namespace) -> None:
    import sys

    import dagnam

    try:
        source = f"@{args.input_file}" if args.input_file else args.input
        payload = ensure_json_object(load_json_arg(source))
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        error(f"Failed to parse --input: {exc}")

    token_count = 0
    printed_tokens = False
    for event in dagnam.inference_stream(args.deployment_id, payload):
        if args.json:
            print(json.dumps({"event": event.event, "data": event.data}), flush=True)
            if event.event == "error":
                data = event.data if isinstance(event.data, dict) else {}
                error(str(data.get("message") or "streaming inference failed"))
            continue
        if event.event == "token":
            data = event.data if isinstance(event.data, dict) else {}
            text = data.get("token")
            if isinstance(text, str):
                token_count += 1
                printed_tokens = True
                print(text, end="", flush=True)
        elif event.event == "complete":
            if printed_tokens:
                print()
            print(f"Stream complete ({token_count} tokens).", file=sys.stderr)
        elif event.event == "error":
            if printed_tokens:
                print()
            data = event.data if isinstance(event.data, dict) else {}
            error(str(data.get("message") or "streaming inference failed"))


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
    schema = inference_sub.add_parser(
        "schema",
        help="Show a deployment's inference schema.",
        description="Show the input/output schema for a deployment's inference endpoint.",
    )
    schema.add_argument("deployment_id", help="ID of the deployment.")
    schema.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    schema.add_argument("--output", help="Write the JSON response to this path.")
    schema.set_defaults(func=cmd_inference_schema)
    stream = inference_sub.add_parser(
        "stream",
        help="Stream a prediction token-by-token.",
        description=(
            "Stream a prediction from a text/LLM deployment over SSE. "
            "Prints tokens incrementally; --json emits one JSON event per line (NDJSON)."
        ),
    )
    stream.add_argument("deployment_id", help="ID of the deployment.")
    stream_input = stream.add_mutually_exclusive_group(required=True)
    stream_input.add_argument("--input", help="JSON literal, or @path to a JSON file.")
    stream_input.add_argument("--input-file", help="Path to a JSON object file.")
    stream.add_argument(
        "--json", action="store_true", help="Emit NDJSON events instead of raw tokens."
    )
    stream.set_defaults(func=cmd_inference_stream)
