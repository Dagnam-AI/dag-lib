"""Shared CLI helper functions."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import NoReturn


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
        return json.loads(path.read_text(encoding="utf-8"))
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


def mask_key(key: str) -> str:
    """Mask a secret, revealing only a short prefix and suffix.

    Short secrets (<= 10 chars) are fully masked so nothing meaningful leaks.
    Uses an ASCII ellipsis to stay safe on legacy Windows code pages.
    """
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]}"
