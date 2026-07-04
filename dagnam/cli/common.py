"""Shared CLI helper functions."""

from __future__ import annotations

import argparse
from contextlib import suppress
from datetime import UTC, datetime
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

DOCS_URL = "https://dagnam.ai/docs"

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

# Plain-ASCII fallback banner for consoles whose encoding cannot represent the
# box-drawing glyphs above. ``dagnam -v`` / ``dagnam -h`` print the banner to
# stdout via argparse; on a legacy code page (cp1252 — the default Windows
# console, many CI shells, Git Bash) writing those glyphs raises
# ``UnicodeEncodeError`` and crashes a command that must never fail. When stdout
# cannot be upgraded to UTF-8, the banner degrades to this ASCII form (G019).
DAGNAM_ASCII_FALLBACK_ART = r"""
==================================================
                    DAGNAM.AI
==================================================
"""


def parse_api_datetime(value: str) -> datetime:
    """Parse an ISO-8601 API timestamp, treating a tz-naive value as UTC.

    Backend timestamps are UTC; some are serialized without an explicit offset.
    A naive result is anchored to UTC so callers never misread it as local time.
    Raises ``ValueError`` on an unparseable string (same as ``fromisoformat``).
    """
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def format_local(value: object) -> str:
    """Render a UTC API timestamp as a local-time date string for display.

    Falsy or non-string inputs render as ``"-"``. An unparseable string falls
    back to its leading date portion so display never crashes on malformed data.
    """
    if not value or not isinstance(value, str):
        return "-"
    try:
        parsed = parse_api_datetime(value)
    except ValueError:
        return value.split("T", maxsplit=1)[0]
    return parsed.astimezone().strftime("%Y-%m-%d")


def configure_console_encoding() -> None:
    """Best-effort upgrade of the console streams to UTF-8.

    Modern terminals (including Windows Terminal and a cp1252 console) accept
    being switched to UTF-8 in place via ``TextIOWrapper.reconfigure`` (Python
    3.7+), which lets the branded banner render with its box-drawing glyphs
    instead of degrading to ASCII. Any stream that cannot be reconfigured (a
    plain pipe, a redirected file, a substituted test stream) is silently
    skipped; those are covered by the ASCII fallback in ``format_ascii_art`` so
    the command still never crashes on a legacy code page (G019).
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            continue


def _stream_can_encode(text: str, stream: object) -> bool:
    """Return whether ``stream``'s text encoding can represent ``text``.

    Reads the stream's declared ``encoding`` (treating a missing or empty value
    as the safest assumption, ``ascii``) and tests a strict round-trip, so we
    can decide whether the box-drawing banner is safe to emit or must degrade to
    the ASCII fallback. An unknown codec name (``LookupError``) is treated as
    not-encodable rather than propagating.
    """
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        text.encode(encoding, errors="strict")
    except (UnicodeEncodeError, LookupError):
        return False
    return True


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
    Degrades to a plain-ASCII banner when stdout's encoding cannot represent the
    box-drawing glyphs, so ``dagnam -v``/``-h`` never crash on a cp1252 console
    (G019).
    """
    width = columns if columns is not None else _terminal_width()
    art = (
        DAGNAM_ASCII_ART
        if _stream_can_encode(DAGNAM_ASCII_ART, sys.stdout)
        else DAGNAM_ASCII_FALLBACK_ART
    )
    return "\n".join(line[:width].rstrip() for line in art.strip("\n").splitlines())


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


def error(msg: str, *, hint: str | None = None) -> NoReturn:
    """Print a contextual error in the unified error frame and exit.

    ``hint`` renders as a ``Try:`` suggestion under the message. Uses the same
    renderer as the ``run_command`` backstop so every CLI failure looks alike.
    """
    from dagnam.cli.errors import render_message

    print(render_message(msg, hint=hint), file=sys.stderr)
    sys.exit(1)


def print_next_step(command: str) -> None:
    """Print a suggested follow-up command to stderr.

    Written to stderr so it never pollutes stdout, which several create handlers
    use for raw JSON that must stay pipe-parseable.
    """
    print(f"\nNext: {command}", file=sys.stderr)


def run_command(args: argparse.Namespace) -> int:
    """Dispatch ``args.func(args)`` with a clean top-level error backstop.

    The single funnel for CLI failures: every exception that escapes a handler
    is rendered by ``dagnam.cli.errors.render_error`` into the unified error
    block (title, details, ``Try:`` suggestions, docs link). ``--debug`` or
    ``DAGNAM_DEBUG`` re-raises the real traceback instead.
    """
    debug = bool(getattr(args, "debug", False)) or bool(os.environ.get("DAGNAM_DEBUG"))
    try:
        args.func(args)
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        # Downstream pipe closed (e.g. `dagnam ... | head`): exit quietly, and
        # point stdout at devnull so interpreter shutdown does not spew a
        # second BrokenPipeError while flushing the dead pipe.
        with suppress(OSError):
            stdout_fd = sys.stdout.fileno()
            os.dup2(os.open(os.devnull, os.O_WRONLY), stdout_fd)
        return 1
    except Exception as exc:  # BLE001 - intentional top-level CLI backstop
        if debug:
            raise
        from dagnam.cli.errors import render_error

        print(render_error(exc), file=sys.stderr)
        return 1
    return 0


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
