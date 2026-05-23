"""Codegen command handlers."""

from __future__ import annotations

import argparse
import json

from dagnam.cli.common import error


def cmd_codegen_generate(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.codegen.generate(
            args.project_id,
            framework=args.framework,
            version_id=args.version_id,
            async_mode=getattr(args, "async"),
        )
    except DagnamError as exc:
        error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def cmd_codegen_preview(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.codegen.preview(
            args.project_id,
            framework=args.framework,
            version_id=args.version_id,
        )
    except DagnamError as exc:
        error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def cmd_codegen_validate(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.codegen.validate(args.project_id, version_id=args.version_id)
    except DagnamError as exc:
        error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def cmd_codegen_download(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.codegen.download(
            args.project_id,
            framework=args.framework,
            version_id=args.version_id,
            dest=args.output,
        )
    except DagnamError as exc:
        error(str(exc))
    print(json.dumps(result, indent=2, default=str))
