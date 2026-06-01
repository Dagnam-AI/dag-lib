"""Codegen command handlers."""

from __future__ import annotations

import argparse
import json
import sys

from dagnam.cli.common import error, write_json_file


def _emit_json_result(result: object, output: str | None, *, label: str) -> None:
    if output:
        write_json_file(output, result)
        print(f"Wrote {label} to {output}")
        return
    print(json.dumps(result, indent=2, default=str))


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
    _emit_json_result(result, args.output, label="code generation response")


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
    _emit_json_result(result, args.output, label="code preview")


def cmd_codegen_validate(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.codegen.validate(args.project_id, version_id=args.version_id)
    except DagnamError as exc:
        error(str(exc))
    _emit_json_result(result, args.output, label="validation response")


def cmd_codegen_download(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    show_progress = not (getattr(args, "no_progress", False) or not sys.stderr.isatty())
    try:
        result = dagnam.codegen.download(
            args.project_id,
            framework=args.framework,
            version_id=args.version_id,
            dest=args.output,
            show_progress=show_progress,
        )
    except DagnamError as exc:
        error(str(exc))
    print(str(result))
