"""Consistent rendering and JSON emission for CLI commands."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Literal

# C0 controls (except tab/newline), DEL, and C1 controls. This range includes
# ESC (0x1b) and CSI (0x9b), so stripping it neutralises every ANSI/OSC escape
# sequence — the mechanism behind terminal-title spoofing, OSC 8 hyperlink
# forgery, and OSC 52 clipboard writes from a hostile server-supplied string.
_TERMINAL_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def sanitize_terminal_text(value: object) -> str:
    """Strip terminal control/escape characters from server-controlled text.

    Every human-mode renderer prints server-returned strings (dataset/hub names
    and descriptions, project titles, error bodies) straight to the TTY. Left
    raw, an embedded escape sequence could spoof output, retitle the terminal,
    forge a hyperlink, or write the user's clipboard. Removing the control chars
    (which include ESC and CSI) makes the text inert while leaving all printable
    content — and tabs/newlines — intact.
    """
    return _TERMINAL_CONTROL_CHARS.sub("", str(value))


@dataclass(frozen=True)
class Column:
    """One bounded-width ASCII table column."""

    label: str
    key: str
    max_width: int
    align: Literal["left", "right"] = "left"


def truncate(value: object, width: int) -> str:
    """Render a value within ``width`` using an ASCII ellipsis marker.

    Sanitises terminal control/escape sequences first so a hostile
    server-supplied cell value cannot reach the TTY as a live escape.
    """
    text = sanitize_terminal_text(value)
    marker = "..."
    if len(text) <= width:
        return text
    return text[: max(width - len(marker), 0)] + marker[:width]


def _cell(value: object, width: int, align: Literal["left", "right"]) -> str:
    text = truncate(value, width)
    return text.rjust(width) if align == "right" else text.ljust(width)


def render_table(columns: Sequence[Column], rows: Sequence[Mapping[str, object]]) -> str:
    """Render rows as a stable ASCII table with bounded columns."""
    widths = [
        min(
            column.max_width,
            max(
                [len(column.label)]
                + [len(truncate(row.get(column.key, "-"), column.max_width)) for row in rows]
            ),
        )
        for column in columns
    ]
    header = " ".join(
        _cell(column.label, width, column.align)
        for column, width in zip(columns, widths, strict=True)
    )
    lines = [header, "-" * len(header)]
    lines.extend(
        " ".join(
            _cell(row.get(column.key, "-"), width, column.align)
            for column, width in zip(columns, widths, strict=True)
        )
        for row in rows
    )
    return "\n".join(lines)


def pagination_footer(result: object) -> str:
    """Render page context without exposing backend empty-page quirks."""
    data: Mapping[str, object] = result if isinstance(result, dict) else {}
    items = data.get("items")
    shown = len(items) if isinstance(items, list) else 0
    page = data.get("page")
    pages = data.get("pages")
    total = data.get("total")
    page = page if isinstance(page, int) and page > 0 else 1
    pages = pages if isinstance(pages, int) and pages > 0 else 1
    total = total if isinstance(total, int) and total >= 0 else shown
    return f"Page {page} of {pages} - showing {shown} of {total}"


def emit_result(
    result: object,
    *,
    output: str | Path | None,
    json_stdout: bool,
    render_human: Callable[[object], str],
) -> None:
    """Write full JSON if requested and emit JSON or concise human stdout."""
    rendered_json = json.dumps(result, indent=2, default=str)
    if output is not None:
        Path(output).write_text(rendered_json + "\n", encoding="utf-8")
    print(rendered_json if json_stdout else render_human(result))
