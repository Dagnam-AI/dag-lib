"""Hub command handlers."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from dagnam._core.exceptions import HubModelNotFoundError
from dagnam.cli.common import (
    add_collection_output_args,
    confirm_destructive,
    error,
    print_json,
)
from dagnam.cli.hub_publish import (
    FILE_KINDS_HELP,
    add_design_args,
    design_from_args,
    register_hub_publish,
    sdk_usage_errors,
)
from dagnam.cli.presentation import Column, emit_result, pagination_footer, render_table
from dagnam.resources.hub_publish import base_fields

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
                "stage": model.get("stage") or "-",
                "framework": model.get("framework") or "-",
                "task": model.get("task_type") or model.get("task") or "-",
            }
        )
    table = render_table(
        (
            Column("ID", "id", 36),
            Column("Name", "name", 32),
            Column("Stage", "stage", 11),
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

    with sdk_usage_errors():
        result = dagnam.hub.search(
            search=args.search,
            framework=args.framework,
            task_type=args.task_type,
            stage=args.stage,
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

    result = dagnam.hub.upload_file(args.model_id, args.file_path, version_id=args.version_id)
    print_json(result)


def _file_line(entry: Mapping[str, object]) -> str:
    parts = [
        str(entry.get("id")),
        str(entry.get("file_name")),
        str(entry.get("scan_status") or "clean"),
    ]
    if entry.get("parameter_count") is not None:
        parts.append(f"{entry['parameter_count']} parameters")
    rejection = entry.get("rejection")
    if isinstance(rejection, dict):
        parts.append(f"{rejection.get('code')}: {rejection.get('message')}")
    return "  ".join(parts)


def cmd_hub_files(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.hub.list_files(args.model_id)
    if args.json:
        print_json(result)
        return
    entries = result.get("files")
    files = (
        [entry for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []
    )
    print("\n".join(_file_line(entry) for entry in files) if files else "No files.")


def cmd_hub_delete_file(args: argparse.Namespace) -> None:
    import dagnam

    try:
        dagnam.hub.delete_file(args.model_id, args.file_id)
    except HubModelNotFoundError:
        error(
            f"File or model not found: {args.file_id} in {args.model_id}.",
            hint=f"dagnam hub files {args.model_id}",
        )


def cmd_hub_link_design(args: argparse.Namespace) -> None:
    import dagnam

    with sdk_usage_errors():
        result = dagnam.hub.link_design(
            args.model_id,
            args.version_id,
            project_id=args.project_id,
            architecture_version_id=args.architecture_version_id,
            architecture=design_from_args(args),
        )
    print_json(result)


def cmd_hub_derivatives(args: argparse.Namespace) -> None:
    import dagnam

    with sdk_usage_errors():
        result = dagnam.hub.derivatives(
            args.model_id, stage=args.stage, page=args.page, limit=args.limit
        )
    _emit_collection(args, result)


def cmd_hub_download(args: argparse.Namespace) -> None:
    import dagnam

    for path in dagnam.hub.download(args.model_id, args.out, file_id=args.file_id):
        print(path)


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
    if args.license is not None:
        fields["license"] = args.license
    if args.license_name is not None:
        fields["license_name"] = args.license_name
    if args.license_url is not None:
        fields["license_url"] = args.license_url
    if args.base is not None:
        fields.update(base_fields(args.base))
    if not fields:
        error(
            "Nothing to update: pass at least one of --name/--description/--visibility/--tags/"
            "--base/--license/--license-name/--license-url."
        )
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
    hub_search.add_argument(
        "--stage",
        action="append",
        help="Filter by stage (repeat for several): design, pretrained, ...",
    )
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
    hub_upload.add_argument(
        "file_path",
        help=f"Path to the file to upload: {FILE_KINDS_HELP}",
    )
    hub_upload.add_argument(
        "--version-id",
        dest="version_id",
        help="The draft version to add the file to; a published version refuses new files.",
    )
    hub_upload.set_defaults(func=cmd_hub_upload_file)
    hub_files = hub_sub.add_parser(
        "files",
        help="List a model's files and their inspection state.",
        description=(
            "List the files of a hub model: pending (still being inspected), clean, or rejected "
            "with the reason. The default list is the latest published version, or your draft."
        ),
    )
    hub_files.add_argument("model_id", help="ID of the hub model.")
    hub_files.add_argument("--json", action="store_true", help="Print the full JSON listing.")
    hub_files.set_defaults(func=cmd_hub_files)
    hub_delete_file = hub_sub.add_parser(
        "delete-file",
        help="Remove a file from a hub model.",
        description="Remove a refused file, or any file of a draft version, from a hub model you manage.",
    )
    hub_delete_file.add_argument("model_id", help="ID of the hub model.")
    hub_delete_file.add_argument("file_id", help="ID of the file (see hub files).")
    hub_delete_file.set_defaults(func=cmd_hub_delete_file)
    hub_link = hub_sub.add_parser(
        "link-design",
        help="Give a weights-only version its design.",
        description=(
            "Link a design to a version that has weights and none: from one of your projects "
            "(--project) or a Studio design file (--architecture @file). The design must need "
            "between half of and all the floating point values the weights hold."
        ),
    )
    hub_link.add_argument("model_id", help="ID of the hub model.")
    hub_link.add_argument(
        "--version-id", dest="version_id", required=True, help="The version to link."
    )
    add_design_args(hub_link)
    hub_link.set_defaults(func=cmd_hub_link_design)
    hub_download = hub_sub.add_parser(
        "download",
        help="Download a model's files.",
        description="Save a hub model's files (or one file with --file-id) into a directory.",
    )
    hub_download.add_argument("model_id", help="ID of the hub model.")
    hub_download.add_argument("--file-id", dest="file_id", help="Download only this file.")
    hub_download.add_argument(
        "--out", default=".", help="Directory to save into (default: current directory)."
    )
    hub_download.set_defaults(func=cmd_hub_download)

    register_hub_publish(hub_sub)

    hub_update = hub_sub.add_parser(
        "update", help="Update a hub model.", description="Update mutable hub model fields."
    )
    hub_update.add_argument("model_id", help="ID of the hub model.")
    hub_update.add_argument("--name", help="New name.")
    hub_update.add_argument("--description", help="New description.")
    hub_update.add_argument("--visibility", help="public, private or unlisted.")
    hub_update.add_argument("--tags", help="Comma-separated tags (replaces existing).")
    hub_update.add_argument("--base", help="The model this one derives from: a Hub id or a name.")
    hub_update.add_argument("--license", help="A licence name (see hub publish --help).")
    hub_update.add_argument("--license-name", help="Display name of a custom licence.")
    hub_update.add_argument("--license-url", help="https link to a custom licence's text.")
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

    hub_derivatives = hub_sub.add_parser(
        "derivatives",
        help="List models made from a model.",
        description="List public models derived from a hub model.",
    )
    hub_derivatives.add_argument("model_id", help="ID of the hub model.")
    hub_derivatives.add_argument(
        "--stage", action="append", help="Only this stage (repeat for several)."
    )
    hub_derivatives.add_argument("--page", type=int, default=1, help="Page number (default: 1).")
    hub_derivatives.add_argument(
        "--limit", type=int, default=20, help="Results per page (default: 20)."
    )
    add_collection_output_args(hub_derivatives)
    hub_derivatives.set_defaults(func=cmd_hub_derivatives)

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
