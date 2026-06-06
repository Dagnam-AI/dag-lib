"""Shared CLI helper functions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    # argparse exposes no public type for the object returned by
    # ``ArgumentParser.add_subparsers()``; ``_SubParsersAction`` is the canonical
    # (underscore-prefixed) type. Alias it once here so every command module
    # shares a single clean annotation instead of repeating the private reference.
    type SubParsersAction = argparse._SubParsersAction[argparse.ArgumentParser]  # pyright: ignore[reportPrivateUsage]

# dos rebel font: <https://patorjk.com/software/taag/#p=display&f=DOS+Rebel&t=DAGNAM.AI&x=none&v=4&h=3&w=80&we=false>
DAGNAM_ASCII_ART = r"""
 ██████████   █████████   █████████ ██████   █████ █████████ ██████   ██████      █████████ █████
░░███░░░░███ ███░░░░░███ ███░░░░░██░░██████ ░░███ ███░░░░░██░░██████ ██████      ███░░░░░██░░███
 ░███   ░░██░███    ░██████     ░░░ ░███░███ ░███░███    ░███░███░█████░███     ░███    ░███░███
 ░███    ░██░██████████░███         ░███░░███░███░███████████░███░░███ ░███     ░███████████░███
 ░███    ░██░███░░░░░██░███    █████░███ ░░██████░███░░░░░███░███ ░░░  ░███     ░███░░░░░███░███
 ░███    ███░███    ░██░░███  ░░███ ░███  ░░█████░███    ░███░███      ░███     ░███    ░███░███
 ██████████ █████   ████░░█████████ █████  ░░█████████   █████████     █████ ██ █████   █████████
░░░░░░░░░░ ░░░░░   ░░░░░ ░░░░░░░░░ ░░░░░    ░░░░░░░░░   ░░░░░░░░░     ░░░░░ ░░ ░░░░░   ░░░░░░░░░
"""


def _terminal_width(fallback: int = 80) -> int:
    """Return the current terminal width, tracking live window resizes.

    Queries the attached terminal device (stdout, then stderr) directly so the
    width reflects the window's current size. This is preferred over
    ``shutil.get_terminal_size``, which honors a possibly-stale ``COLUMNS``
    environment variable and would otherwise pin the banner to an old width
    after the user resizes the window. Falls back to ``shutil`` (which reads
    ``COLUMNS`` then the default) only when no terminal is attached, e.g. when
    output is piped or redirected.
    """
    for stream in (sys.__stdout__, sys.__stderr__):
        if stream is None:
            continue
        try:
            return os.get_terminal_size(stream.fileno()).columns
        except (OSError, ValueError, AttributeError):
            continue
    return shutil.get_terminal_size(fallback=(fallback, 24)).columns


def format_ascii_art(columns: int | None = None) -> str:
    """Stem banner lines to the available terminal width so they do not wrap.

    Recomputes the width on every call so the banner stays responsive to live
    terminal resizes rather than freezing at the width seen on first render.
    """
    width = columns if columns is not None else _terminal_width()
    return "\n".join(line[:width].rstrip() for line in DAGNAM_ASCII_ART.strip("\n").splitlines())


def add_collection_output_args(command: argparse.ArgumentParser) -> None:
    """Add the shared ``--json``/``--verbose``/``--output`` options to a command."""
    command.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON response instead of the concise table.",
    )
    command.add_argument(
        "--verbose",
        action="store_true",
        help="Compatibility alias for --json.",
    )
    command.add_argument("--output", help="Save the full JSON response to this path.")


def human_size(nbytes: int | float) -> str:
    """Format byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def dir_size(path: Path) -> int:
    """Recursively compute total size of a directory in bytes."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def error(msg: str) -> NoReturn:
    """Print an error message to stderr and exit."""
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def load_json_arg(value: str) -> object:
    """Parse --input/--inputs as JSON literal or @path/to/file.json."""
    if value.startswith("@"):
        path = Path(value[1:])
        return json.loads(path.read_text(encoding="utf-8-sig"))
    return json.loads(value)


def print_json(value: object) -> None:
    """Print a JSON value with stable CLI formatting."""
    print(json.dumps(value, indent=2, default=str))


def write_json_file(path: str | Path, value: object) -> None:
    """Write a JSON value to disk with the same formatting used by verbose output."""
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def resolve_version() -> str:
    """Return the installed dagnam version, falling back to the in-tree __version__.

    Uses installed package metadata as the source of truth (so it reflects the
    actually-installed wheel and never drifts from pyproject), and only imports
    the heavy top-level package on the fallback path used for source checkouts.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("dagnam")
    except PackageNotFoundError:
        from dagnam import __version__

        return __version__


def format_version_banner() -> str:
    """Return the branded human-readable CLI version string."""
    return f"{format_ascii_art()}\n\ndagnam {resolve_version()}"


def mask_key(key: str) -> str:
    """Mask a secret, revealing only a short prefix and suffix.

    Short secrets (<= 10 chars) are fully masked so nothing meaningful leaks.
    Uses an ASCII ellipsis to stay safe on legacy Windows code pages.
    """
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]}"
