"""Hub command handlers."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any, cast

from dagnam.cli.common import (
    add_collection_output_args,
    confirm_destructive,
    error,
    print_json,
    print_next_step,
)
from dagnam.cli.presentation import Column, emit_result, pagination_footer, render_table

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction


def _collection_items(result: object) -> list[object]:
    if isinstance(result, dict):
        items = result.get("items")
        return items if isinstance(items, list) else []
    return result if isinstance(result, list) else []


def _render_models(result: object) -> str:
    items = _collection_items(result)
    if not items:
        return "No hub models found."
    rows: list[dict[str, object]] = []
    for item in items:
        model = item if isinstance(item, dict) else {"name": item}
        rows.append(
            {
                **model,
                "name": model.get("name") or model.get("title") or "-",
                "framework": model.get("framework") or "-",
                "task": model.get("task_type") or model.get("task") or "-",
            }
        )
    table = render_table(
        (
            Column("ID", "id", 36),
            Column("Name", "name", 32),
            Column("Framework", "framework", 12),
            Column("Task", "task", 20),
        ),
        rows,
    )
    return f"{table}\n{pagination_footer(result)}" if isinstance(result, dict) else table


def _emit_collection(args: argparse.Namespace, result: object) -> None:
    emit_result(
        result,
        output=args.output,
        json_stdout=args.json or args.verbose,
        render_human=_render_models,
    )


def cmd_hub_search(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.hub.search(
        search=args.search,
        framework=args.framework,
        task_type=args.task_type,
        sort_by=args.sort_by,
        page=args.page,
        limit=args.limit,
    )
    _emit_collection(args, result)


def cmd_hub_get(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.hub.get(args.model_id)
    print_json(result)


def cmd_hub_star(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.hub.star(args.model_id)
    print_json(result)


def cmd_hub_unstar(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.hub.unstar(args.model_id)
    print_json(result)


def cmd_hub_fork(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.hub.fork(args.model_id)
    print_json(result)


def cmd_hub_featured(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.hub.featured()
    _emit_collection(args, result)


def cmd_hub_trending(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.hub.trending(days=args.days)
    _emit_collection(args, result)


def cmd_hub_upload_file(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.hub.upload_file(args.model_id, args.file_path)
    print_json(result)


def cmd_hub_publish(args: argparse.Namespace) -> None:
    from pathlib import Path
    import sys

    import dagnam

    def _progress(path: str, index: int, total: int, state: str) -> None:
        name = Path(path).name
        print(f"[{index}/{total}] {name}: {state}", file=sys.stderr)

    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
    result = dagnam.hub.publish(
        name=args.name,
        description=args.description,
        task_type=args.task_type,
        framework=args.framework,
        files=args.file,
        version=args.version,
        changelog=args.changelog,
        license=args.license,
        visibility=args.visibility,
        tags=tags,
        on_file_progress=_progress,
    )
    print_json(result)
    model = result.get("model")
    model_id = model.get("id") if isinstance(model, dict) else None
    print_next_step(f"dagnam hub get {model_id or '<model-id>'}")


def cmd_hub_update(args: argparse.Namespace) -> None:
    import dagnam

    fields: dict[str, object] = {}
    if args.name is not None:
        fields["name"] = args.name
    if args.description is not None:
        fields["description"] = args.description
    if args.visibility is not None:
        fields["visibility"] = args.visibility
    if args.tags is not None:
        fields["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
    if not fields:
        error("Nothing to update: pass at least one of --name/--description/--visibility/--tags.")
    # Cast for the dynamic-kwargs forward: pyright otherwise distributes each
    # dict value to update()'s reserved keyword params (client/api_key/api_url).
    result = dagnam.hub.update(args.model_id, **cast("dict[str, Any]", fields))
    print_json(result)


def cmd_hub_delete(args: argparse.Namespace) -> None:
    import dagnam

    confirm_destructive(
        args.model_id,
        yes=args.yes,
        prompt=(
            f"About to permanently delete hub model {args.model_id}.\n"
            f"Type '{args.model_id}' to continue: "
        ),
    )
    dagnam.hub.delete(args.model_id)
    print(f"Hub model {args.model_id} deleted.")


def cmd_hub_versions(args: argparse.Namespace) -> None:
    import dagnam

    print_json(dagnam.hub.list_versions(args.model_id))


def cmd_hub_review(args: argparse.Namespace) -> None:
    import dagnam

    print_json(dagnam.hub.add_review(args.model_id, args.rating, review_text=args.text))


def cmd_hub_starred(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.hub.starred(sort_by=args.sort_by, page=args.page, limit=args.limit)
    _emit_collection(args, result)


def cmd_hub_use_in_studio(args: argparse.Namespace) -> None:
    import dagnam

    print_json(dagnam.hub.use_in_studio(args.model_id))


def register_hub(subparsers: SubParsersAction) -> None:
    """Register the ``hub`` command group on the top-level subparsers."""
    hub = subparsers.add_parser(
        "hub",
        help="Browse the model hub.",
        description="Search, inspect, star, and fork models on the hub.",
    )
    hub_sub = hub.add_subparsers(dest="hub_command", required=True)
    hub_search = hub_sub.add_parser(
        "search", help="Search the hub.", description="Search models on the hub."
    )
    hub_search.add_argument("--search", help="Query string.")
    hub_search.add_argument("--task-type", help="Filter by task type.")
    hub_search.add_argument("--framework", help="Filter by framework.")
    hub_search.add_argument("--sort-by", default="popular", help="Sort order (default: popular).")
    hub_search.add_argument("--page", type=int, default=1, help="Page number (default: 1).")
    hub_search.add_argument("--limit", type=int, default=20, help="Results per page (default: 20).")
    add_collection_output_args(hub_search)
    hub_search.set_defaults(func=cmd_hub_search)
    hub_help = {
        "get": ("Show a hub model.", "Show details for one hub model."),
        "star": ("Star a model.", "Star a hub model."),
        "unstar": ("Unstar a model.", "Remove a star from a hub model."),
        "fork": ("Fork a model.", "Fork a hub model into your account."),
    }
    for command_name, handler in {
        "get": cmd_hub_get,
        "star": cmd_hub_star,
        "unstar": cmd_hub_unstar,
        "fork": cmd_hub_fork,
    }.items():
        short_help, long_help = hub_help[command_name]
        command = hub_sub.add_parser(command_name, help=short_help, description=long_help)
        command.add_argument("model_id", help="ID of the hub model.")
        command.set_defaults(func=handler)
    hub_featured = hub_sub.add_parser(
        "featured", help="List featured models.", description="List featured hub models."
    )
    add_collection_output_args(hub_featured)
    hub_featured.set_defaults(func=cmd_hub_featured)
    hub_trending = hub_sub.add_parser(
        "trending", help="List trending models.", description="List trending hub models."
    )
    hub_trending.add_argument(
        "--days", type=int, default=7, help="Trailing window in days (default: 7)."
    )
    add_collection_output_args(hub_trending)
    hub_trending.set_defaults(func=cmd_hub_trending)
    hub_upload = hub_sub.add_parser(
        "upload-file",
        help="Upload a file to a hub model.",
        description="Upload a file to a hub model you own.",
    )
    hub_upload.add_argument("model_id", help="ID of the hub model.")
    hub_upload.add_argument("file_path", help="Path to the file to upload.")
    hub_upload.set_defaults(func=cmd_hub_upload_file)

    hub_publish = hub_sub.add_parser(
        "publish",
        help="Publish a model to the hub.",
        description="Create a hub model, upload its files, and finalize it.",
    )
    hub_publish.add_argument("--name", required=True, help="Model name.")
    hub_publish.add_argument("--description", required=True, help="Model description.")
    hub_publish.add_argument(
        "--task-type", required=True, dest="task_type", help="Task type (e.g. text-generation)."
    )
    hub_publish.add_argument("--framework", required=True, help="Framework (e.g. pytorch).")
    hub_publish.add_argument(
        "--file",
        action="append",
        required=True,
        help="File to upload (repeat for multiple files).",
    )
    hub_publish.add_argument("--version", help="Version string to record (e.g. 1.0.0).")
    hub_publish.add_argument("--changelog", help="Changelog for --version.")
    hub_publish.add_argument("--license", default="mit", help="License (default: mit).")
    hub_publish.add_argument(
        "--visibility", default="public", help="public or private (default: public)."
    )
    hub_publish.add_argument("--tags", help="Comma-separated tags.")
    hub_publish.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    hub_publish.set_defaults(func=cmd_hub_publish)

    hub_update = hub_sub.add_parser(
        "update", help="Update a hub model.", description="Update mutable hub model fields."
    )
    hub_update.add_argument("model_id", help="ID of the hub model.")
    hub_update.add_argument("--name", help="New name.")
    hub_update.add_argument("--description", help="New description.")
    hub_update.add_argument("--visibility", help="public or private.")
    hub_update.add_argument("--tags", help="Comma-separated tags (replaces existing).")
    hub_update.set_defaults(func=cmd_hub_update)

    hub_delete = hub_sub.add_parser(
        "delete", help="Delete a hub model.", description="Permanently delete a hub model you own."
    )
    hub_delete.add_argument("model_id", help="ID of the hub model.")
    hub_delete.add_argument("--yes", action="store_true", help="Skip the typed confirmation.")
    hub_delete.set_defaults(func=cmd_hub_delete)

    hub_versions = hub_sub.add_parser(
        "versions", help="List model versions.", description="List all versions of a hub model."
    )
    hub_versions.add_argument("model_id", help="ID of the hub model.")
    hub_versions.set_defaults(func=cmd_hub_versions)

    hub_review = hub_sub.add_parser(
        "review", help="Review a model.", description="Add a rating and optional review text."
    )
    hub_review.add_argument("model_id", help="ID of the hub model.")
    hub_review.add_argument(
        "--rating", type=int, choices=range(1, 6), required=True, help="Rating from 1 to 5."
    )
    hub_review.add_argument("--text", help="Review text.")
    hub_review.set_defaults(func=cmd_hub_review)

    hub_starred = hub_sub.add_parser(
        "starred", help="List starred models.", description="List models you have starred."
    )
    hub_starred.add_argument(
        "--sort-by", default="date_starred", dest="sort_by", help="Sort order."
    )
    hub_starred.add_argument("--page", type=int, default=1, help="Page number (default: 1).")
    hub_starred.add_argument(
        "--limit", type=int, default=20, help="Results per page (default: 20)."
    )
    add_collection_output_args(hub_starred)
    hub_starred.set_defaults(func=cmd_hub_starred)

    hub_studio = hub_sub.add_parser(
        "use-in-studio",
        help="Import a model into Studio.",
        description="Import a hub model into Studio as a new project.",
    )
    hub_studio.add_argument("model_id", help="ID of the hub model.")
    hub_studio.set_defaults(func=cmd_hub_use_in_studio)
