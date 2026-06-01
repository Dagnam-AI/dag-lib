"""Deployment command handlers."""

from __future__ import annotations

import argparse

from dagnam.cli.common import error, print_json
from dagnam.cli.presentation import Column, emit_result, pagination_footer, render_table


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
