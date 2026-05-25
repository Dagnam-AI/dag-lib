"""Project command handlers."""

from __future__ import annotations

import argparse

from dagnam.cli.common import error, print_json, write_json_file


def _collection_items(result: object) -> list[object]:
    if isinstance(result, dict):
        items = result.get("items")
        return items if isinstance(items, list) else []
    return result if isinstance(result, list) else []


def _date(value: object) -> str:
    return str(value or "-").split("T", maxsplit=1)[0]


def _print_projects(result: object) -> None:
    items = _collection_items(result)
    if not items:
        print("No projects found.")
        return

    header = f"{'ID':<36} {'TITLE':<32} {'STATUS':<10} {'VERSION':<10} {'UPDATED':<10}"
    print(header)
    print("-" * len(header))
    for item in items:
        project = item if isinstance(item, dict) else {}
        title = str(project.get("title") or project.get("name") or "-")[:32]
        status = str(project.get("status") or "-")[:10]
        version = str(project.get("latest_version_number") or "-")[:10]
        project_id = project.get("id") or "-"
        print(
            f"{project_id!s:<36} "
            f"{title:<32} "
            f"{status:<10} "
            f"{version:<10} "
            f"{_date(project.get('updated_at')):<10}"
        )

    total = result.get("total") if isinstance(result, dict) else len(items)
    print(f"Total: {total} project{'s' if total != 1 else ''}")


def cmd_projects_list(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.projects.list(
            framework=args.framework,
            search=args.search,
            page=args.page,
            limit=args.limit,
        )
    except DagnamError as exc:
        error(str(exc))
    if args.output:
        write_json_file(args.output, result)
    if args.verbose:
        print_json(result)
    else:
        _print_projects(result)


def cmd_projects_get(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.projects.get(args.project_id)
    except DagnamError as exc:
        error(str(exc))
    print_json(result)


def cmd_projects_create(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.projects.create(
            title=args.title,
            framework=args.framework,
            description=args.description,
            visibility=args.visibility,
        )
    except DagnamError as exc:
        error(str(exc))
    print_json(result)


def cmd_projects_delete(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        dagnam.projects.delete(args.project_id)
    except DagnamError as exc:
        error(str(exc))
    print(f"Project {args.project_id} deleted.")


def cmd_projects_duplicate(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.projects.duplicate(args.project_id, title=args.title)
    except DagnamError as exc:
        error(str(exc))
    print_json(result)
