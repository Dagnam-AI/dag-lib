"""Codegen command handlers."""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from dagnam.cli.common import add_collection_output_args
from dagnam.cli.presentation import emit_result

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction


def _render_validate(result: object) -> str:
    data = result if isinstance(result, dict) else {}
    valid = bool(data.get("valid"))
    status = "valid" if valid else "invalid"
    errors = data.get("errors")
    if isinstance(errors, list) and errors:
        joined = "; ".join(str(item) for item in errors)
        return f"Project is {status}: {joined}"
    return f"Project is {status}."


def _render_preview(result: object) -> str:
    data = result if isinstance(result, dict) else {}
    files = data.get("files")
    if isinstance(files, list) and files:
        names = [str(f.get("name", "?")) for f in files if isinstance(f, dict)]
        return f"Preview: {len(files)} file(s): {', '.join(names)}"
    return "Preview ready."


def _render_generate(result: object) -> str:
    data = result if isinstance(result, dict) else {}
    task_id = data.get("task_id")
    if task_id is not None:
        return f"Generation started (task {task_id})."
    files = data.get("files")
    if isinstance(files, list):
        return f"Generated {len(files)} file(s)."
    return "Code generated."


def cmd_codegen_generate(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.codegen.generate(
        args.project_id,
        framework=args.framework,
        version_id=args.version_id,
        async_mode=getattr(args, "async"),
    )
    emit_result(
        result,
        output=args.output,
        json_stdout=args.json or args.verbose,
        render_human=_render_generate,
    )


def cmd_codegen_preview(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.codegen.preview(
        args.project_id,
        framework=args.framework,
        version_id=args.version_id,
    )
    emit_result(
        result,
        output=args.output,
        json_stdout=args.json or args.verbose,
        render_human=_render_preview,
    )


def cmd_codegen_validate(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.codegen.validate(args.project_id, version_id=args.version_id)
    emit_result(
        result,
        output=args.output,
        json_stdout=args.json or args.verbose,
        render_human=_render_validate,
    )


def cmd_codegen_download(args: argparse.Namespace) -> None:
    import dagnam

    show_progress = not (getattr(args, "no_progress", False) or not sys.stderr.isatty())
    result = dagnam.codegen.download(
        args.project_id,
        framework=args.framework,
        version_id=args.version_id,
        dest=args.dest,
        show_progress=show_progress,
    )
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
        if command_name == "download":
            command.add_argument(
                "--dest",
                help="Write the downloaded ZIP to this path.",
            )
            command.add_argument(
                "--no-progress",
                action="store_true",
                help=(
                    "Disable download progress output. Progress is also hidden "
                    "when stderr is redirected."
                ),
            )
        else:
            command.add_argument(
                "--async", action="store_true", help="Run asynchronously and return a job."
            )
            add_collection_output_args(command)
        command.set_defaults(func=handler)
