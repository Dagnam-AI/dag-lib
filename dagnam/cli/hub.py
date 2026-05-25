"""Hub command handlers."""

from __future__ import annotations

import argparse

from dagnam.cli.common import error, print_json, write_json_file


def _collection_items(result: object) -> list[object]:
    if isinstance(result, dict):
        items = result.get("items")
        return items if isinstance(items, list) else []
    return result if isinstance(result, list) else []


def _print_models(result: object) -> None:
    items = _collection_items(result)
    if not items:
        print("No hub models found.")
        return

    header = f"{'ID':<36} {'NAME':<32} {'FRAMEWORK':<12} {'TASK':<20}"
    print(header)
    print("-" * len(header))
    for item in items:
        model = item if isinstance(item, dict) else {"name": item}
        name = str(model.get("name") or model.get("title") or "-")[:32]
        framework = str(model.get("framework") or "-")[:12]
        task = str(model.get("task_type") or model.get("task") or "-")[:20]
        model_id = model.get("id") or "-"
        print(f"{model_id!s:<36} {name:<32} {framework:<12} {task:<20}")

    total = result.get("total") if isinstance(result, dict) else len(items)
    print(f"Total: {total} model{'s' if total != 1 else ''}")


def _emit_collection(args: argparse.Namespace, result: object) -> None:
    if args.output:
        write_json_file(args.output, result)
    if args.verbose:
        print_json(result)
    else:
        _print_models(result)


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
