"""Project command handlers."""

from __future__ import annotations

import argparse
import json

from dagnam.cli.common import error


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
    print(json.dumps(result, indent=2, default=str))


def cmd_projects_get(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.projects.get(args.project_id)
    except DagnamError as exc:
        error(str(exc))
    print(json.dumps(result, indent=2, default=str))


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
    print(json.dumps(result, indent=2, default=str))


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
    print(json.dumps(result, indent=2, default=str))
