"""Hub command handlers."""

from __future__ import annotations

import argparse
import json

from dagnam.cli.common import error


def cmd_hub_search(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.hub.search(
            search=args.search,
            framework=args.framework,
            task_type=args.task_type,
            sort_by=args.sort_by,
            page=args.page,
            limit=args.limit,
        )
    except DagnamError as exc:
        error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def cmd_hub_get(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.hub.get(args.model_id)
    except DagnamError as exc:
        error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def cmd_hub_star(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.hub.star(args.model_id)
    except DagnamError as exc:
        error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def cmd_hub_unstar(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.hub.unstar(args.model_id)
    except DagnamError as exc:
        error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def cmd_hub_fork(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.hub.fork(args.model_id)
    except DagnamError as exc:
        error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def cmd_hub_featured(_args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.hub.featured()
    except DagnamError as exc:
        error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def cmd_hub_trending(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.hub.trending(days=args.days)
    except DagnamError as exc:
        error(str(exc))
    print(json.dumps(result, indent=2, default=str))
