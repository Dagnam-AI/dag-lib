"""Deployment command handlers."""

from __future__ import annotations

import argparse
import json

from dagnam.cli.common import error


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
    print(json.dumps(result, indent=2, default=str))


def cmd_deployments_get(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.deployments.get(args.deployment_id)
    except DagnamError as exc:
        error(str(exc))
    print(json.dumps(result, indent=2, default=str))


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
    print(json.dumps(result, indent=2, default=str))


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
    print(json.dumps(result, indent=2, default=str))


def cmd_deployments_metrics(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.deployments.metrics(args.deployment_id, time_range=args.time_range)
    except DagnamError as exc:
        error(str(exc))
    print(json.dumps(result, indent=2, default=str))
