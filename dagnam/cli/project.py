"""Project command handlers."""

from __future__ import annotations

import argparse

from dagnam.cli.common import error, load_json_arg, print_json, write_json_file
from dagnam.cli.presentation import Column, emit_result, pagination_footer, render_table


def _collection_items(result: object) -> list[object]:
    if isinstance(result, dict):
        items = result.get("items")
        return items if isinstance(items, list) else []
    return result if isinstance(result, list) else []


def _date(value: object) -> str:
    return str(value or "-").split("T", maxsplit=1)[0]


def _render_projects(result: object) -> str:
    items = _collection_items(result)
    if not items:
        return "No projects found."
    rows: list[dict[str, object]] = []
    for item in items:
        project = item if isinstance(item, dict) else {}
        rows.append(
            {
                **project,
                "title": project.get("title") or project.get("name") or "-",
                "status": project.get("status") or "-",
                "version": project.get("latest_version_number") or "-",
                "updated": _date(project.get("updated_at")),
            }
        )
    table = render_table(
        (
            Column("ID", "id", 36),
            Column("Title", "title", 32),
            Column("Status", "status", 10),
            Column("Version", "version", 10),
            Column("Updated", "updated", 10),
        ),
        rows,
    )
    return f"{table}\n{pagination_footer(result)}"


def _print_project_summary(result: object) -> None:
    project = result if isinstance(result, dict) else {}
    project_id = project.get("id") or "-"
    title = project.get("title") or project.get("name") or "-"
    print(f"Project {project_id}")
    print(f"Title: {title}")
    print(f"Status: {project.get('status') or '-'}")
    print(f"Framework: {project.get('framework') or '-'}")
    print(f"Latest version: {project.get('latest_version_number') or '-'}")
    print(f"Updated: {_date(project.get('updated_at'))}")


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
    emit_result(
        result,
        output=args.output,
        json_stdout=args.json or args.verbose,
        render_human=_render_projects,
    )


def cmd_projects_get(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.projects.get(args.project_id)
    except DagnamError as exc:
        error(str(exc))
    if args.output:
        write_json_file(args.output, result)
    if args.json or args.verbose:
        print_json(result)
    else:
        _print_project_summary(result)


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
    if args.output:
        write_json_file(args.output, result)
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
    if args.output:
        write_json_file(args.output, result)
    print_json(result)


def cmd_projects_architecture(args: argparse.Namespace) -> None:
    """Save a project's architecture (diagram state + config) from JSON inputs."""
    import dagnam
    from dagnam._core.exceptions import DagnamError
    from dagnam._types import ensure_json_value

    try:
        diagram_state = ensure_json_value(load_json_arg(args.diagram))
        architecture_config = ensure_json_value(load_json_arg(args.config))
    except (OSError, ValueError) as exc:
        error(f"Could not read JSON input: {exc}")

    try:
        result = dagnam.projects.save_architecture(
            args.project_id,
            diagram_state,
            architecture_config,
            commit_message=args.message,
        )
    except DagnamError as exc:
        error(str(exc))
    print_json(result)
