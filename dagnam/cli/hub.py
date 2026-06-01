"""Hub command handlers."""

from __future__ import annotations

import argparse

from dagnam.cli.common import error, print_json
from dagnam.cli.presentation import Column, emit_result, pagination_footer, render_table


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
