"""Hub command handlers."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from dagnam.cli.common import add_collection_output_args, error, print_json
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
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.hub.search(
            search=args.search,
            framework=args.framework,
            task_type=args.task_type,
            sort_by=args.sort_by,
            page=args.page,
            limit=args.limit,
        )
    except DagnamError as exc:
        error(str(exc))
    _emit_collection(args, result)


def cmd_hub_get(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.hub.get(args.model_id)
    except DagnamError as exc:
        error(str(exc))
    print_json(result)


def cmd_hub_star(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.hub.star(args.model_id)
    except DagnamError as exc:
        error(str(exc))
    print_json(result)


def cmd_hub_unstar(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.hub.unstar(args.model_id)
    except DagnamError as exc:
        error(str(exc))
    print_json(result)


def cmd_hub_fork(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.hub.fork(args.model_id)
    except DagnamError as exc:
        error(str(exc))
    print_json(result)


def cmd_hub_featured(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.hub.featured()
    except DagnamError as exc:
        error(str(exc))
    _emit_collection(args, result)


def cmd_hub_trending(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.hub.trending(days=args.days)
    except DagnamError as exc:
        error(str(exc))
    _emit_collection(args, result)


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
