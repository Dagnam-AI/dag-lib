"""Shared reader machinery: streaming rows through polars, malformed accounting, field helpers.

A reader never loads a whole export into memory: rows stream through polars in
batches of :data:`BATCH_ROWS`. JSONL is scanned line-by-line (each line parsed
here rather than by ``pl.scan_ndjson``, which aborts the whole file on one bad
line and cannot represent a column whose JSON type varies between rows, both
of which real vendor exports contain); Parquet and CSV go through
``pl.scan_parquet`` / ``pl.scan_csv``. A ``.gz`` file is gunzipped to a
temporary file first.

Malformed rows are counted and skipped. The share is judged once the file has
been read: above :data:`MALFORMED_FATAL_SHARE` the read ends with
:class:`MalformedExportError` carrying the first three offending row indices;
below it the count is simply reported in :class:`ReadStats`.

Token usage is every vendor's own spelling of the same two numbers, so the
source readers all go through :func:`prompt_tokens` / :func:`completion_tokens`
rather than each knowing one shape.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import gzip
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any

import polars as pl

from dagnam._core.exceptions import DagnamError
from dagnam._core.text import sanitize_terminal_text
from dagnam.audit.record import Message, TraceRecord

MALFORMED_FATAL_SHARE = 0.05
BATCH_ROWS = 1_000
UNKNOWN_MODEL = "unknown"
_FIRST_MALFORMED_KEPT = 3
_SYSTEM_ROLES = frozenset({"system", "developer"})

Row = dict[str, Any]


class UnsupportedExportError(DagnamError):
    """The export's first row lacks fields the chosen source reader requires."""

    def __init__(self, source: str, missing: tuple[str, ...]) -> None:
        self.source = source
        self.missing = missing
        super().__init__(
            f"{source} export is missing required field(s) {', '.join(missing)}; "
            "describe the columns with --source jsonl --map target=column"
        )


@dataclass(slots=True)
class ReadStats:
    """Row accounting for one read; final once the record iterator is exhausted."""

    rows_seen: int = 0
    rows_kept: int = 0
    rows_malformed: int = 0
    first_malformed: tuple[int, ...] = ()

    def note_malformed(self, index: int) -> None:
        """Count a malformed row, remembering the first few indices for the error message."""
        self.rows_malformed += 1
        if len(self.first_malformed) < _FIRST_MALFORMED_KEPT:
            self.first_malformed = (*self.first_malformed, index)


class MalformedExportError(DagnamError):
    """More than :data:`MALFORMED_FATAL_SHARE` of the rows could not be read."""

    def __init__(self, source: str, stats: ReadStats) -> None:
        self.source = source
        self.stats = stats
        super().__init__(
            f"{source} export: {stats.rows_malformed} of {stats.rows_seen} rows are malformed "
            f"(more than {MALFORMED_FATAL_SHARE:.0%}); first offending row indices: "
            f"{list(stats.first_malformed)}"
        )


class MalformedRowError(ValueError):
    """Raised by a source reader for a row it cannot turn into a record."""


@dataclass(frozen=True, slots=True)
class Reader:
    """One export format: the top-level fields it needs and its row converter."""

    required_fields: tuple[str, ...]
    to_record: Callable[[Row], TraceRecord | None]


# ---- streaming rows ------------------------------------------------------------


def _parse_line(line: str) -> Row | None:
    try:
        value = json.loads(line)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _jsonl_rows(path: Path) -> Iterator[Row | None]:
    # One string column holding each raw line: the separator never occurs in
    # valid JSON (control characters must be escaped) and quotes stay literal.
    frame = pl.scan_csv(
        path,
        has_header=False,
        separator="\x01",
        quote_char=None,
        new_columns=["line"],
        infer_schema=False,
        encoding="utf8-lossy",
    )
    for batch in frame.collect_batches(chunk_size=BATCH_ROWS):
        for line in batch.get_column("line"):
            if line is not None:  # polars reads a blank line as null; it is not a row
                yield _parse_line(line)


def _frame_rows(scan: Callable[[Path], pl.LazyFrame]) -> Callable[[Path], Iterator[Row | None]]:
    def rows(path: Path) -> Iterator[Row | None]:
        for batch in scan(path).collect_batches(chunk_size=BATCH_ROWS):
            yield from batch.iter_rows(named=True)

    return rows


_SCANNERS: dict[str, Callable[[Path], Iterator[Row | None]]] = {
    ".jsonl": _jsonl_rows,
    ".ndjson": _jsonl_rows,
    ".parquet": _frame_rows(pl.scan_parquet),
    # Every CSV cell arrives as a string; the field helpers coerce numbers.
    ".csv": _frame_rows(lambda path: pl.scan_csv(path, infer_schema=False)),
}


def iter_rows(path: Path) -> Iterator[Row | None]:
    """Yield each row of a JSONL/Parquet/CSV (optionally gzipped) export as a dict.

    A JSONL line that is not a JSON object yields ``None`` so the caller can
    count it as malformed without losing its index.
    """
    if path.suffix == ".gz":
        with tempfile.TemporaryDirectory() as tmp:
            inner = Path(tmp) / path.name[: -len(".gz")]
            with gzip.open(path, "rb") as src, inner.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            yield from iter_rows(inner)
        return
    scan = _SCANNERS.get(path.suffix)
    if scan is None:
        raise ValueError(
            f"unsupported export file type {path.suffix!r}: expected .jsonl, .parquet or .csv "
            "(optionally gzipped)"
        )
    try:
        yield from scan(path)
    except pl.exceptions.NoDataError:  # an empty file has no rows, which is not an error
        return


def read_records(
    path: Path, source: str, reader: Reader
) -> tuple[Iterator[TraceRecord], ReadStats]:
    """Stream ``path`` through ``reader``; the stats fill in as the iterator advances."""
    stats = ReadStats()
    return _records(path, source, reader, stats), stats


def _records(path: Path, source: str, reader: Reader, stats: ReadStats) -> Iterator[TraceRecord]:
    shape_checked = False
    for index, row in enumerate(iter_rows(path)):
        stats.rows_seen += 1
        if row is None:
            stats.note_malformed(index)
            continue
        if not shape_checked:
            missing = tuple(name for name in reader.required_fields if name not in row)
            if missing:
                raise UnsupportedExportError(source, missing)
            shape_checked = True
        try:
            record = reader.to_record(row)
        except MalformedRowError:
            stats.note_malformed(index)
            continue
        if record is None:
            continue
        stats.rows_kept += 1
        yield record
    if stats.rows_seen and stats.rows_malformed / stats.rows_seen > MALFORMED_FATAL_SHARE:
        raise MalformedExportError(source, stats)


# ---- field helpers shared by the source readers ----------------------------------


def get(row: Mapping[str, Any], *paths: str) -> Any:
    """First non-``None`` value found at any of the dotted ``paths`` (aliases in priority order)."""
    for path in paths:
        value: Any = row
        for key in path.split("."):
            value = value.get(key) if isinstance(value, Mapping) else None
        if value is not None:
            return value
    return None


def require(row: Mapping[str, Any], *paths: str) -> Any:
    """Like :func:`get`, but a missing value makes the row malformed."""
    value = get(row, *paths)
    if value is None:
        raise MalformedRowError(f"missing {paths[0]}")
    return value


# The two token counts as OpenAI, Anthropic, Google, Langfuse and LangSmith each
# spell them. ``input``/``output`` are the Langfuse usage-block names and are
# tried only inside a usage block: at the top level of a row they are the prompt
# and the reply, not counts.
_PROMPT_TOKEN_KEYS = (
    "prompt_tokens",
    "promptTokens",
    "input_tokens",
    "inputTokens",
    "promptTokenCount",
    "prompt_token_count",
)
_COMPLETION_TOKEN_KEYS = (
    "completion_tokens",
    "completionTokens",
    "output_tokens",
    "outputTokens",
    "candidatesTokenCount",
    "candidates_token_count",
)
# Anthropic reports cached prompt tokens separately from ``input_tokens``, so
# they are added on rather than chosen between.
_CACHED_PROMPT_TOKEN_KEYS = ("cache_read_input_tokens", "cache_creation_input_tokens")


def _token_paths(
    roots: tuple[str, ...], keys: tuple[str, ...], nested: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(
        f"{root}.{key}" if root else key
        for root in roots
        for key in ((*keys, *nested) if root else keys)
    )


def _tokens(row: Mapping[str, Any], paths: tuple[str, ...]) -> int:
    return as_int(get(row, *paths) or 0)


def prompt_tokens(row: Mapping[str, Any], *roots: str) -> int:
    """Prompt tokens from a usage block at any of ``roots`` (``""`` = the row itself).

    Every vendor spelling is tried per root, in the order the roots are given:
    OpenAI's ``prompt_tokens``, Anthropic's ``input_tokens``, Gemini's
    ``promptTokenCount``, Langfuse's ``usage.input``, LangSmith's
    ``usage_metadata.input_tokens``. Anthropic's cache-read and cache-creation
    counts are added to the base count, which excludes them. Absent counts are 0.
    """
    total = _tokens(row, _token_paths(roots, _PROMPT_TOKEN_KEYS, ("input",)))
    return total + sum(
        _tokens(row, _token_paths(roots, (key,), ())) for key in _CACHED_PROMPT_TOKEN_KEYS
    )


def completion_tokens(row: Mapping[str, Any], *roots: str) -> int:
    """Completion tokens from a usage block at any of ``roots``; see :func:`prompt_tokens`."""
    return _tokens(row, _token_paths(roots, _COMPLETION_TOKEN_KEYS, ("output",)))


def text(value: object) -> str:
    """Stringify ``value`` (JSON for containers, empty for ``None``) and strip control characters."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, default=str)
    return sanitize_terminal_text(value)


def as_float(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        raise MalformedRowError(f"not a number: {value!r}")
    try:
        number = float(value)
    except ValueError as exc:
        raise MalformedRowError(str(exc)) from exc
    if math.isnan(number):
        raise MalformedRowError("not a number: nan")
    return number


def as_int(value: object) -> int:
    return int(as_float(value))


def optional_float(value: object) -> float | None:
    """``None`` for an absent value (``None`` or an empty CSV cell), else :func:`as_float`."""
    return None if value is None or value == "" else as_float(value)


def parse_ts(value: object) -> datetime:
    """Accept ISO-8601 text, epoch seconds or a datetime; always return a UTC-aware datetime."""
    try:
        if isinstance(value, datetime):
            ts = value
        elif isinstance(value, (int, float)):
            ts = datetime.fromtimestamp(value, tz=UTC)
        elif isinstance(value, str):
            ts = datetime.fromisoformat(value)
        else:
            raise MalformedRowError(f"not a timestamp: {value!r}")
    except (ValueError, OverflowError, OSError) as exc:
        raise MalformedRowError(f"not a timestamp: {value!r}") from exc
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


def content_text(content: object) -> str:
    """Flatten a chat ``content`` value (a string or a list of typed parts) to plain text."""
    if isinstance(content, list):
        return "".join(
            text(part.get("text")) if isinstance(part, dict) else text(part) for part in content
        )
    return text(content)


def split_prompt(prompt: object) -> tuple[str | None, tuple[Message, ...]]:
    """Split a chat prompt into ``(system, turns)``.

    ``prompt`` may be a list of ``{role, content}`` messages, a mapping holding
    such a list under ``messages``, or a bare string (one user turn). System
    turns are joined into ``system``; assistant turns are dropped.
    """
    if isinstance(prompt, Mapping):
        prompt = prompt.get("messages", prompt)
    if not isinstance(prompt, list):
        return None, (Message("user", text(prompt)),)
    system: list[str] = []
    turns: list[Message] = []
    for message in prompt:
        if not isinstance(message, Mapping) or "role" not in message:
            raise MalformedRowError(f"not a chat message: {message!r}")
        role = text(message["role"])
        content = content_text(message.get("content"))
        if role in _SYSTEM_ROLES:
            system.append(content)
        elif role != "assistant":
            turns.append(Message(role, content))
    if not turns:
        raise MalformedRowError("prompt has no user turn")
    return ("\n".join(system) or None), tuple(turns)


def tool_calls(value: object) -> tuple[dict[str, Any], ...]:
    """A response's tool calls as a tuple of JSON objects (empty when absent)."""
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(call, dict) for call in value):
        raise MalformedRowError(f"tool_calls is not a list of objects: {value!r}")
    return tuple(value)
