"""Trace-export readers: one module per source, dispatched through :data:`READERS`."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from functools import partial
from pathlib import Path
from typing import Literal

from dagnam.audit.readers import generic, langfuse, langsmith, openai_jsonl
from dagnam.audit.readers.base import (
    MALFORMED_FATAL_SHARE,
    MalformedExportError,
    Reader,
    ReadStats,
    UnsupportedExportError,
    read_records,
)
from dagnam.audit.record import TraceRecord

Source = Literal["langfuse", "langsmith", "openai", "jsonl", "csv"]

__all__ = [
    "MALFORMED_FATAL_SHARE",
    "READERS",
    "MalformedExportError",
    "ReadStats",
    "Reader",
    "Source",
    "UnsupportedExportError",
    "read_traces",
]


def _fixed(reader: Reader, column_map: Mapping[str, str] | None) -> Reader:
    if column_map is not None:
        raise ValueError("column_map applies to the jsonl and csv sources only")
    return reader


READERS: dict[str, Callable[[Mapping[str, str] | None], Reader]] = {
    "langfuse": partial(_fixed, langfuse.READER),
    "langsmith": partial(_fixed, langsmith.READER),
    "openai": partial(_fixed, openai_jsonl.READER),
    "jsonl": generic.bind,
    "csv": generic.bind,
}


def read_traces(
    path: Path,
    *,
    source: Source,
    column_map: Mapping[str, str] | None = None,
) -> tuple[Iterator[TraceRecord], ReadStats]:
    """Stream the export at ``path`` as :class:`TraceRecord` values.

    The iterator is lazy; the returned :class:`ReadStats` fills in as it is
    consumed and is final once it is exhausted. Raises
    :class:`UnsupportedExportError` when the first row lacks a field the
    source needs, and ends with :class:`MalformedExportError` when more than
    :data:`MALFORMED_FATAL_SHARE` of the rows could not be read.
    """
    factory = READERS.get(source)
    if factory is None:
        raise ValueError(f"unknown source {source!r}; expected one of {', '.join(sorted(READERS))}")
    return read_records(path, source, factory(column_map))
