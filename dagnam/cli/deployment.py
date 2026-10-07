"""Deployment command handlers."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from dagnam.cli.common import (
    add_collection_output_args,
    format_local,
    print_json,
    print_next_step,
)
from dagnam.cli.presentation import Column, emit_result, pagination_footer, render_table

if TYPE_CHECKING:
    from dagnam._types import JsonObject
    from dagnam.cli.common import SubParsersAction


def _collection_items(result: object) -> list[object]:
    if isinstance(result, dict):
        items = result.get("items")
        return items if isinstance(items, list) else []
    return result if isinstance(result, list) else []


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
                "updated": format_local(deployment.get("updated_at")),
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

    result = dagnam.deployments.list(
        status=args.status,
        platform=args.platform,
        project_id=args.project_id,
        search=args.search,
        page=args.page,
        limit=args.limit,
    )
    result = _redact_deployment_collection(result)
    emit_result(
        result,
        output=args.output,
        json_stdout=args.json or args.verbose,
        render_human=_render_deployments,
    )


def cmd_deployments_get(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.deployments.get(args.deployment_id)
    print_json(_redact_deployment_secrets(result))


def _render_deployed(deployment: JsonObject) -> str:
    return "\n".join(
        [
            f"Deployment {deployment.get('id')} created; it serves once its revision is active.",
            "Store this deployment key now; it will not be shown again:",
            "",
            f"  {deployment.get('api_key')}",
        ]
    )


def cmd_deployments_deploy_version(args: argparse.Namespace) -> None:
    import dagnam

    deployment = dagnam.deployments.deploy_model_version(
        args.version_id, name=args.name, project_id=args.project_id
    )
    emit_result(
        deployment,
        output=args.output,
        json_stdout=args.json,
        render_human=lambda _result: _render_deployed(deployment),
    )
    print_next_step(f"dagnam deployments revisions {deployment.get('id')}")


def cmd_deployments_pause(args: argparse.Namespace) -> None:
    import dagnam

    dagnam.deployments.pause(args.deployment_id).wait()
    print(f"Deployment {args.deployment_id} paused.")


def cmd_deployments_resume(args: argparse.Namespace) -> None:
    import dagnam

    dagnam.deployments.resume(args.deployment_id).wait()
    print(f"Deployment {args.deployment_id} resumed.")


def cmd_deployments_delete(args: argparse.Namespace) -> None:
    import dagnam

    dagnam.deployments.delete(args.deployment_id)
    print(f"Deployment {args.deployment_id} deleted.")


def cmd_deployments_logs(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.deployments.logs(
        args.deployment_id,
        level=args.level,
        search=args.search,
        limit=args.limit,
    )
    print_json(result)


def cmd_deployments_revisions(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.deployments.revisions(args.deployment_id, page=args.page, limit=args.limit)
    print_json(result)


def cmd_deployments_metrics(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.deployments.metrics(args.deployment_id, time_range=args.time_range)
    print_json(result)


def cmd_deployments_update(args: argparse.Namespace) -> None:
    import dagnam

    print_json(dagnam.deployments.update(args.deployment_id, name=args.name))


def cmd_deployments_warm(args: argparse.Namespace) -> None:
    import dagnam

    print_json(dagnam.deployments.set_warm(args.deployment_id, args.on))


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
    deploy_version = deployment_sub.add_parser(
        "deploy-version",
        help="Deploy a model version.",
        description=(
            "Deploy a servable model version from the registry in one call. "
            "Prints the deployment key once."
        ),
    )
    deploy_version.add_argument(
        "version_id", metavar="VERSION_ID", help="ID of the model version to deploy."
    )
    deploy_version.add_argument(
        "--name", help="Deployment name (default: the model name and version)."
    )
    deploy_version.add_argument(
        "--project-id", help="Project for the deployment (default: the model's project)."
    )
    deploy_version.add_argument("--json", action="store_true", help="Print raw JSON.")
    deploy_version.add_argument("--output", help="Write the raw JSON to this path.")
    deploy_version.set_defaults(func=cmd_deployments_deploy_version)
    deployment_help = {
        "pause": ("Pause a deployment.", "Pause a running deployment."),
        "resume": ("Resume a deployment.", "Resume a paused deployment."),
        "delete": ("Delete a deployment.", "Delete a deployment permanently."),
        "logs": ("Show deployment logs.", "Fetch logs for a deployment."),
        "metrics": ("Show deployment metrics.", "Fetch metrics for a deployment."),
        "revisions": ("Show revision history.", "List a deployment's revisions, newest first."),
    }
    for command_name, handler in {
        "pause": cmd_deployments_pause,
        "resume": cmd_deployments_resume,
        "delete": cmd_deployments_delete,
        "logs": cmd_deployments_logs,
        "metrics": cmd_deployments_metrics,
        "revisions": cmd_deployments_revisions,
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
        if command_name == "revisions":
            command.add_argument("--page", type=int, default=1, help="Page number (default: 1).")
            command.add_argument(
                "--limit", type=int, default=50, help="Results per page (default: 50)."
            )
        command.set_defaults(func=handler)

    warm = deployment_sub.add_parser(
        "warm",
        help="Keep a deployment warm, or release it.",
        description=(
            "--on pins one GPU container continuously so the first call after idle is fast; "
            "this costs for as long as it is on and needs a plan that includes remote GPU. "
            "--off lets the deployment scale to zero when idle."
        ),
    )
    warm.add_argument("deployment_id", help="ID of the deployment.")
    warm_state = warm.add_mutually_exclusive_group(required=True)
    warm_state.add_argument("--on", action="store_true", help="Keep one container warm.")
    warm_state.add_argument(
        "--off", dest="on", action="store_false", help="Scale to zero when idle."
    )
    warm.set_defaults(func=cmd_deployments_warm)

    dep_update = deployment_sub.add_parser(
        "update", help="Rename a deployment.", description="Rename a deployment."
    )
    dep_update.add_argument("deployment_id", help="ID of the deployment.")
    dep_update.add_argument("--name", required=True, help="New name.")
    dep_update.set_defaults(func=cmd_deployments_update)
