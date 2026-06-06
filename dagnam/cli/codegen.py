"""Codegen command handlers."""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING

from dagnam.cli.common import error, write_json_file

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction


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


def register_codegen(subparsers: SubParsersAction) -> None:
    """Register the ``codegen`` command group on the top-level subparsers."""
    codegen = subparsers.add_parser(
        "codegen",
        help="Generate model code from a project.",
        description="Generate, preview, validate, or download model code.",
    )
    codegen_sub = codegen.add_subparsers(dest="codegen_command", required=True)
    codegen_help = {
        "generate": ("Generate code.", "Generate model code for a project."),
        "preview": ("Preview code.", "Preview generated code without saving."),
        "validate": ("Validate code.", "Validate that a project can generate code."),
        "download": ("Download code.", "Download generated code to a file."),
    }
    for command_name, handler in {
        "generate": cmd_codegen_generate,
        "preview": cmd_codegen_preview,
        "validate": cmd_codegen_validate,
        "download": cmd_codegen_download,
    }.items():
        short_help, long_help = codegen_help[command_name]
        command = codegen_sub.add_parser(command_name, help=short_help, description=long_help)
        command.add_argument("project_id", help="ID of the project.")
        command.add_argument("--framework", default="pytorch", help="Framework (default: pytorch).")
        command.add_argument("--version-id", help="Specific project version ID.")
        command.add_argument(
            "--async", action="store_true", help="Run asynchronously and return a job."
        )
        output_help = (
            "Write the downloaded ZIP to this path."
            if command_name == "download"
            else "Write the JSON response to this path."
        )
        command.add_argument("--output", help=output_help)
        if command_name == "download":
            command.add_argument(
                "--no-progress",
                action="store_true",
                help=(
                    "Disable download progress output. Progress is also hidden "
                    "when stderr is redirected."
                ),
            )
        command.set_defaults(func=handler)
