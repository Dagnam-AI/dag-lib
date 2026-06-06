"""Deployment command handlers."""

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


def _date(value: object) -> str:
    return str(value or "-").split("T", maxsplit=1)[0]


def _redact_deployment_secrets(deployment: object) -> object:
    if not isinstance(deployment, dict):
        return deployment
    result = dict(deployment)
    if "api_key" in result:
        result["api_key"] = "<redacted>"
    return result


def _redact_deployment_collection(result: object) -> object:
    if not isinstance(result, dict):
        return result
    sanitized = dict(result)
    items = sanitized.get("items")
    if isinstance(items, list):
        sanitized["items"] = [_redact_deployment_secrets(item) for item in items]
    return sanitized


def _render_deployments(result: object) -> str:
    items = _collection_items(result)
    if not items:
        return "No deployments found."
    rows: list[dict[str, object]] = []
    for item in items:
        deployment = item if isinstance(item, dict) else {}
        rows.append(
            {
                **deployment,
                "name": deployment.get("name") or deployment.get("title") or "-",
                "status": deployment.get("status") or "-",
                "platform": deployment.get("platform") or "-",
                "updated": _date(deployment.get("updated_at")),
            }
        )
    table = render_table(
        (
            Column("ID", "id", 36),
            Column("Name", "name", 28),
            Column("Status", "status", 12),
            Column("Platform", "platform", 12),
            Column("Updated", "updated", 10),
        ),
        rows,
    )
    return f"{table}\n{pagination_footer(result)}"


def cmd_deployments_list(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.deployments.list(
            status=args.status,
            platform=args.platform,
            project_id=args.project_id,
            search=args.search,
            page=args.page,
            limit=args.limit,
        )
    except DagnamError as exc:
        error(str(exc))
    result = _redact_deployment_collection(result)
    emit_result(
        result,
        output=args.output,
        json_stdout=args.json or args.verbose,
        render_human=_render_deployments,
    )


def cmd_deployments_get(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.deployments.get(args.deployment_id)
    except DagnamError as exc:
        error(str(exc))
    print_json(_redact_deployment_secrets(result))


def cmd_deployments_create(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = (
            dagnam.deployments.create(
                name=args.name,
                project_id=args.project_id,
                checkpoint_path=args.checkpoint_path,
                platform=args.platform,
                deployment_type=args.deployment_type,
                instance_type=args.instance_type,
                num_instances=args.num_instances,
            )
            .wait()
            .result()
        )
    except DagnamError as exc:
        error(str(exc))
    print_json(result)


def cmd_deployments_pause(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        dagnam.deployments.pause(args.deployment_id).wait()
    except DagnamError as exc:
        error(str(exc))
    print(f"Deployment {args.deployment_id} paused.")


def cmd_deployments_resume(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        dagnam.deployments.resume(args.deployment_id).wait()
    except DagnamError as exc:
        error(str(exc))
    print(f"Deployment {args.deployment_id} resumed.")


def cmd_deployments_delete(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        dagnam.deployments.delete(args.deployment_id)
    except DagnamError as exc:
        error(str(exc))
    print(f"Deployment {args.deployment_id} deleted.")


def cmd_deployments_logs(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.deployments.logs(
            args.deployment_id,
            level=args.level,
            search=args.search,
            limit=args.limit,
        )
    except DagnamError as exc:
        error(str(exc))
    print_json(result)


def cmd_deployments_metrics(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.deployments.metrics(args.deployment_id, time_range=args.time_range)
    except DagnamError as exc:
        error(str(exc))
    print_json(result)


def register_deployments(subparsers: SubParsersAction) -> None:
    """Register the ``deployments`` command group on the top-level subparsers."""
    deployments = subparsers.add_parser(
        "deployments",
        help="Manage model deployments.",
        description="Create, control, and observe deployments.",
    )
    deployment_sub = deployments.add_subparsers(dest="deployment_command", required=True)
    deployment_list = deployment_sub.add_parser(
        "list", help="List deployments.", description="List your deployments."
    )
    deployment_list.add_argument("--status", help="Filter by status.")
    deployment_list.add_argument("--platform", help="Filter by platform.")
    deployment_list.add_argument("--project-id", help="Filter by project ID.")
    deployment_list.add_argument("--search", help="Filter by name substring.")
    deployment_list.add_argument("--page", type=int, default=1, help="Page number (default: 1).")
    deployment_list.add_argument(
        "--limit", type=int, default=20, help="Results per page (default: 20)."
    )
    add_collection_output_args(deployment_list)
    deployment_list.set_defaults(func=cmd_deployments_list)
    deployment_get = deployment_sub.add_parser(
        "get", help="Show a deployment.", description="Show details for one deployment."
    )
    deployment_get.add_argument("deployment_id", help="ID of the deployment.")
    deployment_get.set_defaults(func=cmd_deployments_get)
    deployment_create = deployment_sub.add_parser(
        "create", help="Create a deployment.", description="Create a new deployment."
    )
    deployment_create.add_argument("--project-id", required=True, help="Project ID (required).")
    deployment_create.add_argument("--name", required=True, help="Deployment name (required).")
    deployment_create.add_argument(
        "--checkpoint-path", required=True, help="Checkpoint path to deploy (required)."
    )
    deployment_create.add_argument(
        "--platform",
        required=True,
        help="Target platform: fastapi, torchserve, vllm, triton, or custom (required).",
    )
    deployment_create.add_argument(
        "--deployment-type", required=True, help="Deployment type (required)."
    )
    deployment_create.add_argument(
        "--instance-type", required=True, help="Compute instance type (required)."
    )
    deployment_create.add_argument(
        "--num-instances", type=int, default=1, help="Instance count (default: 1)."
    )
    deployment_create.set_defaults(func=cmd_deployments_create)
    deployment_help = {
        "pause": ("Pause a deployment.", "Pause a running deployment."),
        "resume": ("Resume a deployment.", "Resume a paused deployment."),
        "delete": ("Delete a deployment.", "Delete a deployment permanently."),
        "logs": ("Show deployment logs.", "Fetch logs for a deployment."),
        "metrics": ("Show deployment metrics.", "Fetch metrics for a deployment."),
    }
    for command_name, handler in {
        "pause": cmd_deployments_pause,
        "resume": cmd_deployments_resume,
        "delete": cmd_deployments_delete,
        "logs": cmd_deployments_logs,
        "metrics": cmd_deployments_metrics,
    }.items():
        short_help, long_help = deployment_help[command_name]
        command = deployment_sub.add_parser(command_name, help=short_help, description=long_help)
        command.add_argument("deployment_id", help="ID of the deployment.")
        if command_name == "logs":
            command.add_argument(
                "--level", help="Filter by log level: debug, info, warning, or error."
            )
            command.add_argument("--search", help="Filter by message substring.")
            command.add_argument("--limit", type=int, default=100, help="Max lines (default: 100).")
        if command_name == "metrics":
            command.add_argument(
                "--time-range", default="24h", help="Window, e.g. 24h (default: 24h)."
            )
        command.set_defaults(func=handler)
