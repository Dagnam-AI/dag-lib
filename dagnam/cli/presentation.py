"""Consistent rendering and JSON emission for CLI commands."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Column:
    """One bounded-width ASCII table column."""

    label: str
    key: str
    max_width: int
    align: Literal["left", "right"] = "left"


def truncate(value: object, width: int) -> str:
    """Render a value within ``width`` using an ASCII ellipsis marker."""
    text = str(value)
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


def pagination_footer(result: Mapping[str, object]) -> str:
    """Render page context without exposing backend empty-page quirks."""
    items = result.get("items")
    shown = len(items) if isinstance(items, list) else 0
    page = result.get("page")
    pages = result.get("pages")
    total = result.get("total")
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
