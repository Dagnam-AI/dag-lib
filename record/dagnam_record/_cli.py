"""``python -m dagnam_record status|purge``: look at, or delete, what the recorder wrote.

Exit codes: 0 done; 1 nothing to show or a purge that was refused or could not finish (the
reason is on stderr); 2 a usage error (argparse's own code). Trace content is never printed: only
the directory, file names, counts and two timestamps.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

from dagnam_record import __version__
from dagnam_record._files import (
    DEFAULT_DIR,
    NotRecorderDirectoryError,
    Summary,
    purge,
    resolve,
    summarize,
)
from dagnam_record._writer import COUNT_KEYS

PROG = "python -m dagnam_record"


def _out(text: str) -> None:
    sys.stdout.write(text + "\n")


def _err(text: str) -> None:
    sys.stderr.write(text + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Show or delete the calls that dagnam-record has written on this machine.",
    )
    parser.add_argument("--version", action="version", version=f"dagnam-record {__version__}")
    parser.add_argument(
        "--dir", default=DEFAULT_DIR, help=f"trace directory (default {DEFAULT_DIR})"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (
        (
            "status",
            "print the directory, file and line counts, first and last time, and drop counts",
        ),
        ("purge", "delete the recorder's files (dated *.jsonl and .status.json) in the directory"),
    ):
        command = commands.add_parser(name, help=text, description=text)
        # SUPPRESS: ``--dir`` may come before or after the command, and the later one wins.
        command.add_argument("--dir", default=argparse.SUPPRESS, help="trace directory")
    return parser


def _render(summary: Summary) -> List[str]:
    lines = [
        f"directory: {summary.directory}",
        f"files: {summary.files}",
        f"lines: {summary.lines}",
        f"first: {summary.first or 'n/a'}",
        f"last: {summary.last or 'n/a'}",
        f"started: {summary.started or 'n/a'}",
        f"expired: {'n/a' if summary.expired is None else ('yes' if summary.expired else 'no')}",
    ]
    if summary.counts is None:
        lines.append("counts: n/a (no readable .status.json)")
    else:
        lines += [f"{key}: {summary.counts[key]}" for key in COUNT_KEYS]
    return lines


def _status(directory: str) -> int:
    try:
        summary = summarize(directory)
    except (OSError, NotRecorderDirectoryError) as exc:
        _err(f"{PROG}: {_why(exc, directory)}")
        return 1
    _out("\n".join(_render(summary)))
    return 0


def _purge(directory: str) -> int:
    try:
        result = purge(directory)
    except FileNotFoundError:
        _out(f"nothing to delete: {directory} does not exist")
        return 0
    except (OSError, NotRecorderDirectoryError) as exc:
        _err(f"{PROG}: refused: {_why(exc, directory)}")
        return 1
    for name in result.deleted:
        _out(f"deleted {name}")
    for name in result.failed:
        _err(f"{PROG}: could not delete {name}")
    _out(f"deleted {len(result.deleted)} file(s) from {directory}")
    return 1 if result.failed else 0


def _why(exc: Exception, directory: str) -> str:
    if isinstance(exc, NotRecorderDirectoryError):
        return f"{directory} holds no recorder file"
    return f"cannot use {directory}: {exc.strerror if isinstance(exc, OSError) and exc.strerror else type(exc).__name__}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command line and return its exit code."""
    arguments = _parser().parse_args(argv)
    directory = resolve(arguments.dir)
    return _status(directory) if arguments.command == "status" else _purge(directory)
