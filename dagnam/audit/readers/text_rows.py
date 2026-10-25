"""Text exports row by row: JSONL and ``.json`` documents, gzipped or not.

Every row a source reader sees from a text file comes through here, so the two hostile-file
checks that cannot be left to each reader live here once: a row whose strings cannot be encoded
as UTF-8 (a lone surrogate escape) is malformed, and a leading byte order mark is not part of
the first row.
"""

from __future__ import annotations

from collections.abc import Iterator
import gzip
import json
from pathlib import Path
import re
from typing import Any, TextIO

Row = dict[str, Any]

_LONE_SURROGATE_ESCAPE = re.compile(r"\\u[dD][89a-fA-F]")
"""A JSON ``\\uD800``-``\\uDFFF`` escape: a lone one decodes to a string no UTF-8 can hold."""


def object_row(value: object, suspect: bool) -> Row | None:
    """``value``, a parsed row, if it is an object whose strings can all be encoded as UTF-8.

    JSON allows a lone surrogate escape, which Python decodes to a ``str`` that raises
    ``UnicodeEncodeError`` at the first hash or write. Such a row is malformed like any other
    the export cannot be read from, so it is counted here, once, for every source. A paired
    surrogate escape is one ordinary character and passes. ``suspect`` is whether the text the
    value came from holds a surrogate escape at all.
    """
    if not isinstance(value, dict):
        return None
    if suspect:  # a cheap pre-check on the text; the encode is the real test
        try:
            json.dumps(value, ensure_ascii=False).encode("utf-8")
        except UnicodeEncodeError:
            return None
    return value


def parse_line(line: str) -> Row | None:
    try:
        return object_row(json.loads(line), _LONE_SURROGATE_ESCAPE.search(line) is not None)
    except (ValueError, RecursionError):  # nested too deeply to decode: one bad row, not the read
        return None


def open_text(path: Path) -> TextIO:
    """The file as text, gunzipped on the fly for ``.gz``; undecodable bytes never stop a read.

    A byte order mark (what Windows tools put first in a UTF-8 file) is dropped, not read as
    part of the first row.
    """
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", errors="replace")
    return path.open(encoding="utf-8-sig", errors="replace")


def jsonl_rows(path: Path) -> Iterator[Row | None]:
    with open_text(path) as handle:
        for line in handle:
            if line.strip():  # a blank line is not a row
                yield parse_line(line)


def json_rows(path: Path) -> Iterator[Row | None]:
    """A ``.json`` export: one array of rows (a UI "Export JSON"), an API page's ``data``, or JSONL."""
    # ponytail: one JSON document is parsed whole; a UI export is small, and a
    # large export comes as JSONL, which streams. Add a streaming parser if not.
    with open_text(path) as handle:
        text = handle.read()
    try:
        document = json.loads(text)
    except (
        ValueError,
        RecursionError,
    ):  # JSON Lines under a .json name (U+2028 may sit in a string), or nested too deep
        yield from (parse_line(line) for line in text.split("\n") if line.strip())
        return
    suspect = _LONE_SURROGATE_ESCAPE.search(text) is not None
    items = document.get("data") if isinstance(document, dict) else document
    for item in items if isinstance(items, list) else [document]:
        yield object_row(item, suspect)
