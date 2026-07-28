"""Deployment command handlers."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any, cast

from dagnam.cli.common import (
    add_collection_output_args,
    error,
    format_local,
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


def cmd_deployments_create(args: argparse.Namespace) -> None:
    import dagnam

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
    print_json(result)
    deployment_id = result.get("id") if isinstance(result, dict) else None
    print_next_step(f"dagnam inference {deployment_id or '<deployment-id>'} run ...")


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


def cmd_deployments_metrics(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.deployments.metrics(args.deployment_id, time_range=args.time_range)
    print_json(result)


def cmd_deployments_collect_metrics(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.deployments.collect_metrics(
        args.deployment_id, backfill_minutes=args.backfill_minutes
    )
    print_json(result)


def cmd_deployments_platforms(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.deployments.platforms()
    print_json(result)


def cmd_deployments_retry(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.deployments.retry(args.deployment_id)
    print_json(result)


def cmd_deployments_estimate_cost(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.deployments.estimate_cost(
        platform=args.platform,
        instance_type=args.instance_type,
        num_instances=args.num_instances,
        auto_scaling_enabled=args.auto_scaling,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        region=args.region,
    )
    print_json(result)


def cmd_deployments_validate(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.deployments.validate(
        name=args.name,
        project_id=args.project_id,
        checkpoint_path=args.checkpoint_path,
        platform=args.platform,
        deployment_type=args.deployment_type,
        instance_type=args.instance_type,
        num_instances=args.num_instances,
        auto_scaling_enabled=args.auto_scaling,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        region=args.region,
    )
    print_json(result)


def cmd_deployments_update(args: argparse.Namespace) -> None:
    import dagnam

    fields: dict[str, object] = {}
    if args.name is not None:
        fields["name"] = args.name
    if args.instance_type is not None:
        fields["instance_type"] = args.instance_type
    if args.num_instances is not None:
        fields["num_instances"] = args.num_instances
    if args.min_instances is not None:
        fields["min_instances"] = args.min_instances
    if args.max_instances is not None:
        fields["max_instances"] = args.max_instances
    if args.auto_scaling is not None:
        fields["auto_scaling_enabled"] = args.auto_scaling
    if not fields:
        error(
            "Nothing to update: pass at least one of --name/--instance-type/--num-instances/"
            "--min-instances/--max-instances/--auto-scaling/--no-auto-scaling."
        )
    result = dagnam.deployments.update(args.deployment_id, **cast("dict[str, Any]", fields))
    print_json(result)


def cmd_deployments_scale(args: argparse.Namespace) -> None:
    import dagnam

    op = dagnam.deployments.scale(args.deployment_id, args.num_instances)
    print_json(op.initial() if args.no_wait else op.wait().result())


def cmd_deployments_rollback(args: argparse.Namespace) -> None:
    import dagnam

    op = dagnam.deployments.rollback(args.deployment_id, args.checkpoint_id)
    print_json(op.initial() if args.no_wait else op.wait().result())


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
        "retry": ("Retry a deployment.", "Retry a failed or stuck deployment."),
        "delete": ("Delete a deployment.", "Delete a deployment permanently."),
        "logs": ("Show deployment logs.", "Fetch logs for a deployment."),
        "metrics": ("Show deployment metrics.", "Fetch metrics for a deployment."),
    }
    for command_name, handler in {
        "pause": cmd_deployments_pause,
        "resume": cmd_deployments_resume,
        "retry": cmd_deployments_retry,
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

    collect = deployment_sub.add_parser(
        "collect-metrics",
        help="Collect deployment metrics now.",
        description="Trigger an immediate metrics collection (backfills on first run).",
    )
    collect.add_argument("deployment_id", help="ID of the deployment.")
    collect.add_argument(
        "--backfill-minutes",
        type=int,
        default=60,
        help="Minutes of history to backfill when no metrics exist yet (default: 60).",
    )
    collect.set_defaults(func=cmd_deployments_collect_metrics)

    platforms = deployment_sub.add_parser(
        "platforms",
        help="List deployment platforms.",
        description="List available serving platforms and their capabilities.",
    )
    platforms.set_defaults(func=cmd_deployments_platforms)

    estimate = deployment_sub.add_parser(
        "estimate-cost",
        help="Estimate deployment cost.",
        description="Estimate hourly/daily/monthly cost for a deployment shape.",
    )
    estimate.add_argument("--platform", required=True, help="Serving platform.")
    estimate.add_argument(
        "--instance-type", required=True, dest="instance_type", help="Compute instance type."
    )
    estimate.add_argument(
        "--num-instances", type=int, default=1, dest="num_instances", help="Instance count."
    )
    estimate.add_argument(
        "--auto-scaling", action="store_true", dest="auto_scaling", help="Enable auto-scaling."
    )
    estimate.add_argument("--min-instances", type=int, default=None, dest="min_instances")
    estimate.add_argument("--max-instances", type=int, default=None, dest="max_instances")
    estimate.add_argument("--region", default=None, help="Deployment region.")
    estimate.set_defaults(func=cmd_deployments_estimate_cost)

    validate = deployment_sub.add_parser(
        "validate",
        help="Validate a deployment config.",
        description="Validate a deployment configuration without creating it.",
    )
    validate.add_argument("--name", required=True, help="Deployment name.")
    validate.add_argument("--project-id", required=True, dest="project_id", help="Project ID.")
    validate.add_argument(
        "--checkpoint-path", required=True, dest="checkpoint_path", help="Checkpoint path."
    )
    validate.add_argument("--platform", required=True, help="Serving platform.")
    validate.add_argument(
        "--deployment-type", required=True, dest="deployment_type", help="Deployment type."
    )
    validate.add_argument(
        "--instance-type", required=True, dest="instance_type", help="Compute instance type."
    )
    validate.add_argument(
        "--num-instances", type=int, default=1, dest="num_instances", help="Instance count."
    )
    validate.add_argument(
        "--auto-scaling", action="store_true", dest="auto_scaling", help="Enable auto-scaling."
    )
    validate.add_argument("--min-instances", type=int, default=None, dest="min_instances")
    validate.add_argument("--max-instances", type=int, default=None, dest="max_instances")
    validate.add_argument("--region", default=None, help="Deployment region.")
    validate.set_defaults(func=cmd_deployments_validate)

    dep_update = deployment_sub.add_parser(
        "update", help="Update a deployment.", description="Update mutable deployment fields."
    )
    dep_update.add_argument("deployment_id", help="ID of the deployment.")
    dep_update.add_argument("--name", help="New name.")
    dep_update.add_argument("--instance-type", dest="instance_type", help="New instance type.")
    dep_update.add_argument(
        "--num-instances", type=int, default=None, dest="num_instances", help="Instance count."
    )
    dep_update.add_argument("--min-instances", type=int, default=None, dest="min_instances")
    dep_update.add_argument("--max-instances", type=int, default=None, dest="max_instances")
    auto = dep_update.add_mutually_exclusive_group()
    auto.add_argument(
        "--auto-scaling",
        action="store_true",
        dest="auto_scaling",
        default=None,
        help="Enable auto-scaling.",
    )
    auto.add_argument(
        "--no-auto-scaling",
        action="store_false",
        dest="auto_scaling",
        help="Disable auto-scaling.",
    )
    dep_update.set_defaults(func=cmd_deployments_update)

    dep_scale = deployment_sub.add_parser(
        "scale", help="Scale a deployment.", description="Change a deployment's instance count."
    )
    dep_scale.add_argument("deployment_id", help="ID of the deployment.")
    dep_scale.add_argument(
        "--num-instances", type=int, required=True, dest="num_instances", help="Target count."
    )
    dep_scale.add_argument(
        "--no-wait", action="store_true", help="Return immediately without polling."
    )
    dep_scale.set_defaults(func=cmd_deployments_scale)

    dep_rollback = deployment_sub.add_parser(
        "rollback",
        help="Roll back a deployment.",
        description="Redeploy a previous checkpoint.",
    )
    dep_rollback.add_argument("deployment_id", help="ID of the deployment.")
    dep_rollback.add_argument(
        "--checkpoint-id",
        required=True,
        dest="checkpoint_id",
        help="ID of the checkpoint to redeploy.",
    )
    dep_rollback.add_argument(
        "--no-wait", action="store_true", help="Return immediately without polling."
    )
    dep_rollback.set_defaults(func=cmd_deployments_rollback)
