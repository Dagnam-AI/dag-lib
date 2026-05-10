"""Inference command handlers."""

from __future__ import annotations

import argparse
import json

from dagnam.cli.common import _error, _load_json_arg


def _cmd_inference_run(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        payload = _load_json_arg(args.input)
    except (json.JSONDecodeError, OSError) as exc:
        _error(f"Failed to parse --input: {exc}")

    try:
        result = dagnam.inference(args.deployment_id, payload)
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2))


def _cmd_inference_batch(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        payload = _load_json_arg(args.inputs)
    except (json.JSONDecodeError, OSError) as exc:
        _error(f"Failed to parse --inputs: {exc}")

    if not isinstance(payload, list):
        _error("--inputs must be a JSON array (or @path to a JSON array file)")

    try:
        result = dagnam.inference_batch(args.deployment_id, payload)
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2))


def _cmd_inference_health(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.deployment_health(args.deployment_id)
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2))
