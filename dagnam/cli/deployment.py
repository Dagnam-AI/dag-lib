"""Deployment command handlers."""

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


def _print_deployments(result: object) -> None:
    items = _collection_items(result)
    if not items:
        print("No deployments found.")
        return

    header = f"{'ID':<36} {'NAME':<28} {'STATUS':<12} {'PLATFORM':<12} {'UPDATED':<10}"
    print(header)
    print("-" * len(header))
    for item in items:
        deployment = item if isinstance(item, dict) else {}
        name = str(deployment.get("name") or deployment.get("title") or "-")[:28]
        status = str(deployment.get("status") or "-")[:12]
        platform = str(deployment.get("platform") or "-")[:12]
        deployment_id = deployment.get("id") or "-"
        print(
            f"{deployment_id!s:<36} "
            f"{name:<28} "
            f"{status:<12} "
            f"{platform:<12} "
            f"{_date(deployment.get('updated_at')):<10}"
        )

    total = result.get("total") if isinstance(result, dict) else len(items)
    print(f"Total: {total} deployment{'s' if total != 1 else ''}")


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
    if args.output:
        write_json_file(args.output, result)
    if args.verbose:
        print_json(result)
    else:
        _print_deployments(result)


def cmd_deployments_get(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.deployments.get(args.deployment_id)
    except DagnamError as exc:
        error(str(exc))
    print_json(result)


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
