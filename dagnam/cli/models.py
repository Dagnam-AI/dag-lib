"""`dagnam models` CLI subcommand — push, get, list, download, lineage, task-contract."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

from dagnam.cli.common import add_collection_output_args
from dagnam.cli.presentation import emit_result
from dagnam.resources import models

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction


def cmd_models_push(args: argparse.Namespace) -> None:
    result = models.push(
        name=args.name,
        slug=args.slug,
        description=args.description,
        files=args.file,
        origin=args.origin,
        license=args.license,
        visibility=args.visibility,
    )
    print(json.dumps(result, indent=2) if args.json else result)


def cmd_models_get(args: argparse.Namespace) -> None:
    from dagnam._core.resolver import resolve_client

    client = resolve_client(None, None, None)
    result = client.get_model_entry(args.model_id)
    print(json.dumps(result, indent=2) if args.json else result)


def cmd_models_list(args: argparse.Namespace) -> None:
    from dagnam._core.resolver import resolve_client

    client = resolve_client(None, None, None)
    result = client.list_model_entries(search=args.search, page=args.page, limit=args.limit)
    emit_result(
        result,
        output=args.output,
        json_stdout=args.json or args.verbose,
        render_human=lambda value: json.dumps(value, indent=2, default=str),
    )


def cmd_models_download(args: argparse.Namespace) -> None:
    path = models.download(args.version_id, args.artifact_id, cache_dir=args.output_dir)
    print(json.dumps({"path": str(path)}, indent=2) if args.json else path)


def cmd_models_lineage(args: argparse.Namespace) -> None:
    result = models.get_lineage(args.version_id)
    print(json.dumps(result, indent=2) if args.json else result)


def cmd_models_task_contract(args: argparse.Namespace) -> None:
    result = models.get_task_contract(args.key, args.version)
    print(json.dumps(result, indent=2) if args.json else result)


def register_models(subparsers: SubParsersAction) -> None:
    """Register the ``models`` command group on the top-level subparsers."""
    parser = subparsers.add_parser(
        "models",
        help="Model registry: push, get, list, download, lineage, task-contract.",
        description=(
            "Push, inspect, and download model registry entries, versions, and artifacts."
        ),
    )
    models_sub = parser.add_subparsers(dest="models_command", required=True)

    push = models_sub.add_parser("push", help="Push a new model entry+version+artifacts.")
    push.add_argument("--name", required=True, help="Model name.")
    push.add_argument("--slug", required=True, help="URL-safe unique slug.")
    push.add_argument("--description", required=True, help="Model description.")
    push.add_argument(
        "--file",
        action="append",
        required=True,
        dest="file",
        help="Artifact file to upload (repeat for multiple files).",
    )
    push.add_argument("--origin", default="imported", help="Version origin (default: imported).")
    push.add_argument("--license", default="mit", help="License (default: mit).")
    push.add_argument(
        "--visibility", default="private", help="public or private (default: private)."
    )
    push.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    push.set_defaults(func=cmd_models_push)

    get = models_sub.add_parser("get", help="Get a model entry by id.")
    get.add_argument("model_id", help="ID of the model entry.")
    get.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    get.set_defaults(func=cmd_models_get)

    list_cmd = models_sub.add_parser("list", help="List model entries.")
    add_collection_output_args(list_cmd)
    list_cmd.add_argument("--search", default=None, help="Filter by name/slug search text.")
    list_cmd.add_argument("--page", type=int, default=1, help="Page number (default: 1).")
    list_cmd.add_argument("--limit", type=int, default=20, help="Results per page (default: 20).")
    list_cmd.set_defaults(func=cmd_models_list)

    download = models_sub.add_parser("download", help="Download one artifact of a version.")
    download.add_argument("version_id", help="ID of the model version.")
    download.add_argument("artifact_id", help="ID of the artifact.")
    download.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Cache the downloaded artifact under this directory.",
    )
    download.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    download.set_defaults(func=cmd_models_download)

    lineage = models_sub.add_parser("lineage", help="Get a version's lineage graph.")
    lineage.add_argument("version_id", help="ID of the model version.")
    lineage.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    lineage.set_defaults(func=cmd_models_lineage)

    task_contract = models_sub.add_parser("task-contract", help="Get a task contract definition.")
    task_contract.add_argument("key", help="Task contract key.")
    task_contract.add_argument("version", help="Task contract version.")
    task_contract.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    task_contract.set_defaults(func=cmd_models_task_contract)
