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
import time
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from collections.abc import Callable

    # argparse exposes no public type for the object returned by
    # ``ArgumentParser.add_subparsers()``; ``_SubParsersAction`` is the canonical
    # (underscore-prefixed) type. Alias it once here so every command module
    # shares a single clean annotation instead of repeating the private reference.
    type SubParsersAction = argparse._SubParsersAction[argparse.ArgumentParser]  # pyright: ignore[reportPrivateUsage]

DOCS_URL = "https://dagnam.ai/docs"

# dos rebel font: <https://patorjk.com/software/taag/#p=display&f=DOS+Rebel&t=DAGNAM.AI&x=none&v=4&h=3&w=80&we=false>
DAGNAM_ASCII_ART = r"""
 ██████████   █████████    █████████ ██████   █████ █████████ ██████   ██████      █████████ █████
░░███░░░░███ ███░░░░░███  ███░░░░░██░░██████ ░░███ ███░░░░░██░░██████ ██████      ███░░░░░██░░███
 ░███   ░░██░███    ░███░███    ░░░  ░███░███ ░███░███    ░███░███░█████░███     ░███    ░███░███
 ░███    ░██░███████████░███         ░███░░███░███░███████████░███░░███ ░███     ░███████████░███
 ░███    ░██░███░░░░░███░███    █████░███ ░░██████░███░░░░░███░███ ░░░  ░███     ░███░░░░░███░███
 ░███    ███░███    ░███░░███  ░░███ ░███  ░░█████░███    ░███░███      ░███     ░███    ░███░███
 ██████████ █████   █████░░█████████ █████  ░░█████████   █████████     █████ ██ █████   █████████
░░░░░░░░░░ ░░░░░   ░░░░░  ░░░░░░░░ ░░░░░    ░░░░░░░░░   ░░░░░░░░░     ░░░░░ ░░ ░░░░░   ░░░░░░░░░
"""

# Plain-ASCII fallback banner for consoles whose encoding cannot represent the
# box-drawing glyphs above. ``dagnam -v`` / ``dagnam -h`` print the banner to
# stdout via argparse; on a legacy code page (cp1252 — the default Windows
# console, many CI shells, Git Bash) writing those glyphs raises
# ``UnicodeEncodeError`` and crashes a command that must never fail. When stdout
# cannot be upgraded to UTF-8, the banner degrades to this ASCII form.
# JS Stick Letters font
DAGNAM_ASCII_FALLBACK_ART = r"""
 __        __
|  \  /\  / _` |\ |  /\   |\/|   /\  |
|__/ /--\ \__> | \| /--\  |  | ./--\ |
"""

# Brand palette matching the product's brand theme (light/dark brand red):
# letter bodies use the light-mode brand red and the shading uses the dark-mode
# brand red, so the terminal wordmark matches the product logo.
_BANNER_BODY_RGB = (255, 79, 79)  # oklch(67.517% 0.21256 24.87)
_BANNER_SHADE_RGB = (202, 3, 3)  # oklch(52.768% 0.21534 29.097)
_BANNER_BODY_COLOR = "\x1b[38;2;{};{};{}m".format(*_BANNER_BODY_RGB)
_BANNER_RESET = "\x1b[0m"

# The ``dagnam -v`` banner animation: a highlight band sweeps left-to-right
# through the artwork in a seamless loop, redrawing the banner in place.
# 28 frames at 25 ms per cycle (~0.7 s), 3 cycles, then it settles static.
_BANNER_SWEEP_FRAMES = 28
_BANNER_SWEEP_SECONDS_PER_FRAME = 0.025
_BANNER_SWEEP_HALF_WIDTH = 18.0
_BANNER_SWEEP_LOOPS = 3


def _blend_color(
    base: tuple[int, int, int],
    target: tuple[int, int, int],
    col: int,
    band: float | None,
) -> str:
    """Truecolor escape for a glyph at ``col``, blended toward the sweep band.

    Without a band the glyph keeps its flat ``base`` color. With one, the
    color blends linearly toward ``target`` as the band center approaches,
    reaching the full ``target`` color at the center.
    """
    weight = 0.0 if band is None else max(0.0, 1.0 - abs(col - band) / _BANNER_SWEEP_HALF_WIDTH)
    channels = (round(b + (t - b) * weight) for b, t in zip(base, target, strict=True))
    return "\x1b[38;2;{};{};{}m".format(*channels)


def _colorize_banner(art: str, band: float | None = None) -> str:
    """Paint runs of banner glyphs in the brand palette.

    Escape codes wrap each run of same-colored glyphs, applied after width
    trimming so line-length math always happens on plain text. Solid blocks
    stay body red; shade blocks sit in the dark red and brighten toward the
    body red as the sweep ``band`` passes; the plain-ASCII fallback glyphs get
    the inverse treatment (body red, dipping toward the dark red at the band).
    """
    lines: list[str] = []
    for line in art.splitlines():
        pieces: list[str] = []
        active: str | None = None
        for col, char in enumerate(line):
            if char == "█":
                color: str | None = _BANNER_BODY_COLOR
            elif char == "░":
                color = _blend_color(_BANNER_SHADE_RGB, _BANNER_BODY_RGB, col, band)
            elif not char.isspace():
                color = _blend_color(_BANNER_BODY_RGB, _BANNER_SHADE_RGB, col, band)
            else:
                color = None
            if color != active:
                if active is not None:
                    pieces.append(_BANNER_RESET)
                if color is not None:
                    pieces.append(color)
                active = color
            pieces.append(char)
        if active is not None:
            pieces.append(_BANNER_RESET)
        lines.append("".join(pieces))
    return "\n".join(lines)


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
    the command still never crashes on a legacy code page.
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


def format_ascii_art(columns: int | None = None, *, color: bool | None = None) -> str:
    """Stem banner lines to the available terminal width so they do not wrap.

    Recomputes the width on every call so the banner stays responsive to live
    terminal resizes rather than freezing at the width seen on first render.
    Degrades to a plain-ASCII banner when stdout's encoding cannot represent the
    box-drawing glyphs, so ``dagnam -v``/``-h`` never crash on a cp1252 console
    When ``color`` is ``None`` the banner is painted in the brand
    palette only if stdout supports ANSI styling (same gate as error output).
    """
    width = columns if columns is not None else _terminal_width()
    art = (
        DAGNAM_ASCII_ART
        if _stream_can_encode(DAGNAM_ASCII_ART, sys.stdout)
        else DAGNAM_ASCII_FALLBACK_ART
    )
    plain = "\n".join(line[:width].rstrip() for line in art.strip("\n").splitlines())
    if color is None:
        # Lazy import: errors.py imports DOCS_URL from this module at load time.
        from dagnam.cli.errors import color_enabled

        color = color_enabled(sys.stdout)
    return _colorize_banner(plain) if color else plain


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


def confirm_or_abort(prompt: str, *, assume_yes: bool) -> None:
    """Gate a destructive CLI action behind a typed ``yes`` confirmation.

    Prints ``prompt`` followed by an instruction to type ``yes``, then reads a
    line from stdin. Anything other than an exact (case-sensitive) ``yes``
    aborts with exit code 1 and a clear message, so a stray keypress or blank
    Enter can never confirm a destructive action by accident. Passing
    ``assume_yes=True`` (typically a CLI ``--yes`` flag) skips the prompt
    entirely for scripted/non-interactive use. Generic and reusable across any
    destructive command (e.g. revoking every session at once, deleting the
    account) that needs the same confirm-or-abort behavior.
    """
    if assume_yes:
        return
    print(prompt)
    typed = input("Type 'yes' to confirm: ").strip()
    if typed != "yes":
        error("Aborted: confirmation not received.")


def confirm_destructive(expected: str, *, yes: bool, prompt: str) -> None:
    """Gate a destructive command behind a typed confirmation (or ``--yes``).

    Prints ``prompt``, reads one line, and aborts with exit code 1 unless the
    reply matches ``expected`` exactly. Never default-destructive.
    """
    if yes:
        return
    reply = input(prompt)
    if reply != expected:
        error("Confirmation did not match; aborting.")


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


def print_version_banner(
    *,
    animate: bool | None = None,
    sleep: Callable[[float], object] = time.sleep,
) -> None:
    """Print the branded version banner, animated on interactive terminals.

    A highlight band sweeps left-to-right through the artwork in a seamless
    loop (``_BANNER_SWEEP_LOOPS`` cycles), then settles on the static banner.
    Frames redraw in place with cursor-movement escapes, so the animation only
    runs when stdout is a real TTY that passes the ANSI color gate; anywhere
    else (pipes, redirects, ``NO_COLOR``, dumb or legacy consoles) the static
    banner from ``format_version_banner`` is printed instead. ``animate`` and
    ``sleep`` are injectable for tests.
    """
    if animate is None:
        # Lazy import: errors.py imports DOCS_URL from this module at load time.
        from dagnam.cli.errors import color_enabled

        isatty = getattr(sys.stdout, "isatty", None)
        animate = color_enabled(sys.stdout) and callable(isatty) and bool(isatty())
    if not animate:
        print(format_version_banner())
        return
    art = format_ascii_art(color=False)
    line_count = art.count("\n") + 1
    width = max(len(line) for line in art.splitlines())
    start = -_BANNER_SWEEP_HALF_WIDTH
    span = width + 2 * _BANNER_SWEEP_HALF_WIDTH
    out = sys.stdout
    out.write("\x1b[?25l")  # hide the cursor while frames overwrite each other
    try:
        for frame in range(_BANNER_SWEEP_LOOPS * _BANNER_SWEEP_FRAMES):
            band = start + span * (frame % _BANNER_SWEEP_FRAMES) / _BANNER_SWEEP_FRAMES
            out.write(_colorize_banner(art, band=band) + "\n")
            out.flush()
            sleep(_BANNER_SWEEP_SECONDS_PER_FRAME)
            out.write(f"\x1b[{line_count}F")  # back to the banner's first line
        out.write(_colorize_banner(art) + "\n")  # settle on the static banner
        out.flush()
    finally:
        out.write("\x1b[?25h")
    print(f"\ndagnam {resolve_version()}")


def mask_key(key: str) -> str:
    """Mask a secret, revealing only a short prefix and suffix.

    Short secrets (<= 10 chars) are fully masked so nothing meaningful leaks.
    Uses an ASCII ellipsis to stay safe on legacy Windows code pages.
    """
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]}"
