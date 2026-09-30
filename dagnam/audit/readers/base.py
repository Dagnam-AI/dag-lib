"""Shared reader machinery: streaming rows through polars, malformed accounting, field helpers.

A reader never loads a whole export into memory. JSONL is read line by line,
gzipped or not (each line parsed here rather than by ``pl.scan_ndjson``, which
aborts the whole file on one bad line and cannot represent a column whose JSON
type varies between rows, both of which real vendor exports contain); Parquet
and CSV stream through ``pl.scan_parquet`` / ``pl.scan_csv`` in batches of
:data:`BATCH_ROWS`, a gzipped one inflated to a temporary file first. A
``.json`` document (a UI export's array) is the one format read whole.

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
from datetime import UTC, datetime, timedelta
import gzip
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, TextIO

import polars as pl

from dagnam._core.exceptions import DagnamError
from dagnam._core.text import sanitize_terminal_text
from dagnam.audit.record import TraceRecord

MALFORMED_FATAL_SHARE = 0.05
BATCH_ROWS = 1_000
UNKNOWN_MODEL = "unknown"
_FIRST_MALFORMED_KEPT = 3
EARLIEST_TS = datetime(2020, 1, 1, tzinfo=UTC)
"""No trace of a hosted chat model predates this; an earlier time is a zeroed or bogus field."""
CLOCK_SKEW = timedelta(days=1)
"""How far past now an exporter's clock may run before its timestamps are implausible."""
_EPOCH_UNITS = ((1e17, 1e9), (1e14, 1e6), (1e11, 1e3))  # (from, per second): ns, us, ms
_EPOCH_TEXT = re.compile(r"\d+(?:\.\d+)?")
_BASIC_DATE = re.compile(r"\d{8}")
"""``YYYYMMDD``, ISO 8601's basic date: as an epoch it would be 1970-73, never a trace's time."""

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


def _open_text(path: Path) -> TextIO:
    """The file as text, gunzipped on the fly for ``.gz``; undecodable bytes never stop a read."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def _jsonl_rows(path: Path) -> Iterator[Row | None]:
    with _open_text(path) as handle:
        for line in handle:
            if line.strip():  # a blank line is not a row
                yield _parse_line(line)


def _json_rows(path: Path) -> Iterator[Row | None]:
    """A ``.json`` export: one array of rows (a UI "Export JSON"), an API page's ``data``, or JSONL."""
    # ponytail: one JSON document is parsed whole; a UI export is small, and a
    # large export comes as JSONL, which streams. Add a streaming parser if not.
    with _open_text(path) as handle:
        text = handle.read()
    try:
        document = json.loads(text)
    except ValueError:  # JSON Lines under a .json name; U+2028 etc. may sit inside a string
        yield from (_parse_line(line) for line in text.split("\n") if line.strip())
        return
    items = document.get("data") if isinstance(document, dict) else document
    for item in items if isinstance(items, list) else [document]:
        yield item if isinstance(item, dict) else None


def _frame_rows(scan: Callable[[Path], pl.LazyFrame]) -> Callable[[Path], Iterator[Row | None]]:
    def rows(path: Path) -> Iterator[Row | None]:
        for batch in scan(path).collect_batches(chunk_size=BATCH_ROWS):
            yield from batch.iter_rows(named=True)

    return rows


_TEXT_SUFFIXES = frozenset({".jsonl", ".ndjson", ".json"})
_SCANNERS: dict[str, Callable[[Path], Iterator[Row | None]]] = {
    ".jsonl": _jsonl_rows,
    ".ndjson": _jsonl_rows,
    ".json": _json_rows,
    ".parquet": _frame_rows(pl.scan_parquet),
    # Every CSV cell arrives as a string; the field helpers coerce numbers.
    ".csv": _frame_rows(lambda path: pl.scan_csv(path, infer_schema=False)),
}


def iter_rows(path: Path) -> Iterator[Row | None]:
    """Yield each row of a JSONL/JSON/Parquet/CSV (optionally gzipped) export as a dict.

    A JSONL line that is not a JSON object yields ``None`` so the caller can
    count it as malformed without losing its index. Gzipped JSON and JSONL are
    read through ``gzip`` as they stream; Parquet and CSV, which polars reads
    from a file, are inflated to a temporary one first.
    """
    if path.suffix == ".gz" and Path(path.stem).suffix in _TEXT_SUFFIXES:
        yield from _SCANNERS[Path(path.stem).suffix](path)
        return
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
            f"unsupported export file type {path.suffix!r}: expected .jsonl, .json, .parquet or"
            " .csv (optionally gzipped)"
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
            missing = tuple(name for name in reader.required_fields if field(row, name) is None)
            if missing:
                raise UnsupportedExportError(source, missing)
            shape_checked = True
        try:
            record = reader.to_record(row)
        except (ValueError, LookupError, TypeError, AttributeError):
            # Any shape the reader did not expect (an empty ``choices`` list, a
            # stream chunk's ``delta``) is this row's problem, not the export's:
            # count it, and let the malformed share decide whether to stop.
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


def field(row: Mapping[str, Any], name: str) -> Any:
    """A column by its exact name, else by its dotted path (``input.inputBodyJson.messages``).

    The exact name wins, so a flattened CSV header like ``usage.prompt_tokens``
    is still one column; a present-but-null column reads as ``""``, present.
    """
    if name in row:
        return "" if row[name] is None else row[name]
    return get(row, name)


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
# Counts a vendor keeps beside its base count rather than inside it, added from
# the same usage block: Anthropic's cache reads and writes (its ``input_tokens``
# excludes them) and Gemini's thinking tokens (billed as output, but excluded
# from ``candidatesTokenCount``). Another block's copy is never added: LangChain's
# ``input_tokens`` / ``output_tokens`` already include both.
_EXCLUDED_ALONGSIDE: Mapping[str, tuple[str, ...]] = {
    "input_tokens": ("cache_read_input_tokens", "cache_creation_input_tokens"),
    "candidatesTokenCount": ("thoughtsTokenCount",),
    "candidates_token_count": ("thoughts_token_count",),
}
# Cache-read prompt tokens, which the price table's ``cached_input_per_m`` prices.
_CACHED_PROMPT_TOKEN_KEYS = (
    "prompt_tokens_details.cached_tokens",
    "input_tokens_details.cached_tokens",
    "input_cached_tokens",
    "input_cache_read",
    "cache_read_input_tokens",
    "input_token_details.cache_read",
    "cachedContentTokenCount",
    "cached_content_token_count",
)


def _bucket(key: str) -> bool:
    """Whether a Langfuse ``input_*`` / ``output_*`` key is a token bucket the base count excludes.

    Cost keys are dollars, and the ``*_priority*`` keys re-count tokens the
    other buckets already hold: Langfuse's own SDK never subtracts them (m2).
    """
    return "cost" not in key and "priority" not in key


def _count(
    row: Mapping[str, Any], roots: tuple[str, ...], keys: tuple[str, ...], nested: str
) -> int:
    """The first usage block's count under ``keys`` plus what it keeps alongside; 0 when absent."""
    for root in roots:
        block = get(row, root) if root else row
        if not isinstance(block, Mapping):
            continue
        for key in (*keys, nested) if root else keys:
            value = block.get(key)
            if value is not None:
                extra = _EXCLUDED_ALONGSIDE.get(key, ())
                if key == nested:  # Langfuse: ``input_*`` / ``output_*`` are exclusive buckets
                    extra = tuple(k for k in block if k.startswith(f"{key}_") and _bucket(k))
                return as_int(value or 0) + sum(as_int(get(block, k) or 0) for k in extra)
    return 0


def prompt_tokens(row: Mapping[str, Any], *roots: str) -> int:
    """Prompt tokens from the first usage block at ``roots`` (``""`` = the row itself).

    Every vendor spelling is tried per root, in the order the roots are given:
    OpenAI's ``prompt_tokens``, Anthropic's ``input_tokens``, Gemini's
    ``promptTokenCount``, Langfuse's ``usage.input``, LangSmith's
    ``usage_metadata.input_tokens``. A raw Anthropic block's cache-read and
    cache-creation counts are added to its base count, which excludes them.
    Absent counts are 0.
    """
    return _count(row, roots, _PROMPT_TOKEN_KEYS, "input")


def completion_tokens(row: Mapping[str, Any], *roots: str) -> int:
    """Completion tokens from a usage block at any of ``roots``; see :func:`prompt_tokens`.

    Gemini's thinking tokens are output the vendor bills, so they are added to
    its ``candidatesTokenCount``.
    """
    return _count(row, roots, _COMPLETION_TOKEN_KEYS, "output")


def cached_prompt_tokens(row: Mapping[str, Any], *roots: str) -> int:
    """How many of the prompt tokens were cache reads, in any vendor's spelling; 0 when absent."""
    paths = tuple(
        f"{root}.{key}" if root else key for root in roots for key in _CACHED_PROMPT_TOKEN_KEYS
    )
    return as_int(get(row, *paths) or 0)


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


def _epoch_seconds(value: float) -> float:
    """An epoch in seconds, milliseconds, microseconds or nanoseconds, read by its magnitude.

    Every plausible trace (2020 onward) is ~1.6e9 in seconds, so a thousandfold
    step up is the next unit; exporters (Helicone, PostHog, OTel) use them all.
    """
    for start, per_second in _EPOCH_UNITS:
        if abs(value) >= start:
            return value / per_second
    return value


def optional_outcome(value: object) -> float | None:
    """An outcome score when the export's value is a finite number, else ``None``.

    ``outcome`` is a customer convention the audit reads and does not use yet
    (R1-N11): a text value ("resolved", "thumbs up") is never a malformed row.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_ts(value: object) -> datetime:
    """Accept ISO-8601 text, an epoch (s/ms/us/ns, a number or a numeric string) or a datetime.

    A basic ISO date (``20260927``) is all digits and stays a date. Always
    returns a UTC-aware datetime. A time before :data:`EARLIEST_TS` or
    more than :data:`CLOCK_SKEW` in the future is malformed: one ``created: 0``
    among a day of calls would otherwise stretch the export's window to fifty
    years.
    """
    if (
        isinstance(value, str)
        and _EPOCH_TEXT.fullmatch(value.strip())
        and not _BASIC_DATE.fullmatch(value.strip())
    ):
        value = float(value)
    try:
        if isinstance(value, datetime):
            ts = value
        elif isinstance(value, (int, float)):
            ts = datetime.fromtimestamp(_epoch_seconds(value), tz=UTC)
        elif isinstance(value, str):
            ts = datetime.fromisoformat(value)
        else:
            raise MalformedRowError(f"not a timestamp: {value!r}")
    except (ValueError, OverflowError, OSError) as exc:
        raise MalformedRowError(f"not a timestamp: {value!r}") from exc
    ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)
    if not EARLIEST_TS <= ts <= datetime.now(UTC) + CLOCK_SKEW:
        raise MalformedRowError(f"implausible timestamp: {value!r}")
    return ts
