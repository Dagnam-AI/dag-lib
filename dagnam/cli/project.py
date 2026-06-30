"""Project command handlers."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from dagnam.cli.common import (
    add_collection_output_args,
    error,
    format_local,
    load_json_arg,
    print_json,
    print_next_step,
    write_json_file,
)
from dagnam.cli.presentation import Column, emit_result, pagination_footer, render_table

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction


def _collection_items(result: object) -> list[object]:
    if isinstance(result, dict):
        items = result.get("items")
        return items if isinstance(items, list) else []
    return result if isinstance(result, list) else []


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
                "updated": format_local(project.get("updated_at")),
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


def _print_project_summary_text(result: object) -> str:
    project = result if isinstance(result, dict) else {}
    project_id = project.get("id") or "-"
    title = project.get("title") or project.get("name") or "-"
    return "\n".join(
        (
            f"Project {project_id}",
            f"Title: {title}",
            f"Status: {project.get('status') or '-'}",
            f"Framework: {project.get('framework') or '-'}",
            f"Latest version: {project.get('latest_version_number') or '-'}",
            f"Updated: {format_local(project.get('updated_at'))}",
        )
    )


def _render_architecture(result: object) -> str:
    data = result if isinstance(result, dict) else {}
    version_id = data.get("version_id") or data.get("id")
    version_number = data.get("version_number") or data.get("latest_version_number")
    if version_id is not None:
        return f"Saved architecture version {version_id}."
    if version_number is not None:
        return f"Saved architecture version {version_number}."
    return "Architecture saved."


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
        print(_print_project_summary_text(result))


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
    emit_result(
        result,
        output=args.output,
        json_stdout=args.json or args.verbose,
        render_human=_print_project_summary_text,
    )
    project_id = result.get("id")
    print_next_step(f"dagnam training create {project_id or '<project-id>'} ...")


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
    emit_result(
        result,
        output=args.output,
        json_stdout=args.json or args.verbose,
        render_human=_print_project_summary_text,
    )


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
    emit_result(
        result,
        output=args.output,
        json_stdout=args.json or args.verbose,
        render_human=_render_architecture,
    )


def cmd_projects_versions_list(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.projects.list_versions(args.project_id, page=args.page, limit=args.limit)
    except DagnamError as exc:
        error(str(exc))
    print_json(result)


def cmd_projects_versions_get(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.projects.get_version(args.project_id, args.version_id)
    except DagnamError as exc:
        error(str(exc))
    print_json(result)


def cmd_projects_versions_compare(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.projects.compare_versions(args.project_id, args.version_a, args.version_b)
    except DagnamError as exc:
        error(str(exc))
    print_json(result)


def cmd_projects_versions_restore(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.projects.restore_version(args.project_id, args.version_id)
    except DagnamError as exc:
        error(str(exc))
    print_json(result)


def cmd_projects_versions_delete(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        dagnam.projects.delete_version(args.project_id, args.version_id)
    except DagnamError as exc:
        error(str(exc))
    print(f"Deleted version {args.version_id}")


def cmd_projects_versions_latest(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.projects.latest_version(args.project_id)
    except DagnamError as exc:
        error(str(exc))
    print_json(result)


def _register_project_versions(project_sub: SubParsersAction) -> None:
    """Register the nested ``projects versions …`` command group."""
    versions = project_sub.add_parser(
        "versions",
        help="Manage project architecture versions.",
        description="List, inspect, compare, restore, and delete project versions.",
    )
    versions_sub = versions.add_subparsers(dest="versions_command", required=True)

    v_list = versions_sub.add_parser(
        "list", help="List versions.", description="List a project's architecture versions."
    )
    v_list.add_argument("project_id", help="ID of the project.")
    v_list.add_argument("--page", type=int, default=1, help="Page number (default: 1).")
    v_list.add_argument("--limit", type=int, default=20, help="Results per page (default: 20).")
    v_list.set_defaults(func=cmd_projects_versions_list)

    v_get = versions_sub.add_parser(
        "get", help="Show a version.", description="Show one architecture version."
    )
    v_get.add_argument("project_id", help="ID of the project.")
    v_get.add_argument("version_id", help="ID of the version.")
    v_get.set_defaults(func=cmd_projects_versions_get)

    v_cmp = versions_sub.add_parser(
        "compare", help="Compare two versions.", description="Compare two architecture versions."
    )
    v_cmp.add_argument("project_id", help="ID of the project.")
    v_cmp.add_argument("version_a", help="First version ID.")
    v_cmp.add_argument("version_b", help="Second version ID.")
    v_cmp.set_defaults(func=cmd_projects_versions_compare)

    v_restore = versions_sub.add_parser(
        "restore", help="Restore a version.", description="Restore a project to a prior version."
    )
    v_restore.add_argument("project_id", help="ID of the project.")
    v_restore.add_argument("version_id", help="ID of the version to restore.")
    v_restore.set_defaults(func=cmd_projects_versions_restore)

    v_delete = versions_sub.add_parser(
        "delete", help="Delete a version.", description="Delete one architecture version."
    )
    v_delete.add_argument("project_id", help="ID of the project.")
    v_delete.add_argument("version_id", help="ID of the version to delete.")
    v_delete.set_defaults(func=cmd_projects_versions_delete)

    v_latest = versions_sub.add_parser(
        "latest", help="Show the latest version.", description="Show the current (latest) version."
    )
    v_latest.add_argument("project_id", help="ID of the project.")
    v_latest.set_defaults(func=cmd_projects_versions_latest)


def register_projects(subparsers: SubParsersAction) -> None:
    """Register the ``projects`` command group on the top-level subparsers."""
    projects = subparsers.add_parser(
        "projects",
        help="Manage projects.",
        description="Create, list, inspect, and delete projects.",
    )
    project_sub = projects.add_subparsers(dest="project_command", required=True)
    project_list = project_sub.add_parser(
        "list", help="List projects.", description="List your projects."
    )
    project_list.add_argument("--framework", help="Filter by framework (e.g. pytorch).")
    project_list.add_argument("--search", help="Filter by title/description substring.")
    project_list.add_argument("--page", type=int, default=1, help="Page number (default: 1).")
    project_list.add_argument(
        "--limit", type=int, default=20, help="Results per page (default: 20)."
    )
    add_collection_output_args(project_list)
    project_list.set_defaults(func=cmd_projects_list)
    project_get = project_sub.add_parser(
        "get", help="Show a project.", description="Show details for one project."
    )
    project_get.add_argument("project_id", help="ID of the project.")
    add_collection_output_args(project_get)
    project_get.set_defaults(func=cmd_projects_get)
    project_create = project_sub.add_parser(
        "create", help="Create a project.", description="Create a new project."
    )
    project_create.add_argument("--title", required=True, help="Project title (required).")
    project_create.add_argument(
        "--framework", default="pytorch", help="Framework (default: pytorch)."
    )
    project_create.add_argument("--description", help="Optional project description.")
    project_create.add_argument(
        "--visibility", default="private", help="private or public (default: private)."
    )
    add_collection_output_args(project_create)
    project_create.set_defaults(func=cmd_projects_create)
    project_delete = project_sub.add_parser(
        "delete", help="Delete a project.", description="Delete a project permanently."
    )
    project_delete.add_argument("project_id", help="ID of the project to delete.")
    project_delete.set_defaults(func=cmd_projects_delete)
    project_dup = project_sub.add_parser(
        "duplicate", help="Duplicate a project.", description="Clone an existing project."
    )
    project_dup.add_argument("project_id", help="ID of the project to duplicate.")
    project_dup.add_argument("--title", help="Title for the copy (defaults to a derived name).")
    add_collection_output_args(project_dup)
    project_dup.set_defaults(func=cmd_projects_duplicate)
    project_arch = project_sub.add_parser(
        "architecture",
        help="Save a project's architecture.",
        description=(
            "Save the architecture (diagram state + config) for a project, "
            "creating a new version. Inputs are JSON literals or @path/to/file.json."
        ),
    )
    project_arch.add_argument("project_id", help="ID of the project.")
    project_arch.add_argument(
        "--diagram",
        required=True,
        help="Diagram state as a JSON literal or @path to a JSON file (required).",
    )
    project_arch.add_argument(
        "--config",
        required=True,
        help="Architecture config as a JSON literal or @path to a JSON file (required).",
    )
    project_arch.add_argument("--message", help="Optional commit message for the version.")
    add_collection_output_args(project_arch)
    project_arch.set_defaults(func=cmd_projects_architecture)
    _register_project_versions(project_sub)
