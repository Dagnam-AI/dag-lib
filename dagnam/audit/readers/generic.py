"""Generic JSONL/CSV export -> :class:`TraceRecord` through a ``{target: source}`` column map.

The map names, per :class:`TraceRecord` field (the *target*), the export
column that holds it (the *source*). A target left out of the map reads the
column of its own name, so an export that already uses the record's field
names needs no map at all. The map is validated once, before any row is read:
an unknown target is a :class:`ValueError`; a mapped column that the export
lacks surfaces as :class:`UnsupportedExportError` on the first row.

Required targets: ``trace_id``, ``ts``, ``messages``, ``response``. Every
other target is optional and defaults (``model`` to ``"unknown"``, token
counts to 0, ``latency_ms`` to 0, the rest to ``None``). ``messages`` may be
a chat message list, a JSON string encoding one, or a plain prompt string;
``response_tool_calls`` a list of objects or a JSON string encoding one; an
explicit ``system`` column wins over system turns found in ``messages``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
import json

from dagnam.audit.readers.base import (
    UNKNOWN_MODEL,
    MalformedRowError,
    Reader,
    Row,
    as_int,
    optional_float,
    parse_ts,
    split_prompt,
    text,
    tool_calls,
)
from dagnam.audit.record import TraceRecord

TARGETS = tuple(field.name for field in fields(TraceRecord))
REQUIRED_TARGETS = ("trace_id", "ts", "messages", "response")


def _decoded(value: object) -> object:
    """A JSON string (as a CSV cell must carry a list) decodes to its value; anything else passes."""
    if isinstance(value, str) and value[:1] in ("[", "{"):
        try:
            return json.loads(value)
        except ValueError as exc:
            raise MalformedRowError(f"not JSON: {value!r}") from exc
    return value


def bind(column_map: Mapping[str, str] | None) -> Reader:
    """Validate ``column_map`` and return the reader that applies it."""
    columns = {target: target for target in TARGETS}
    for target, source in (column_map or {}).items():
        if target not in columns:
            raise ValueError(
                f"unknown column_map target {target!r}; expected one of {', '.join(TARGETS)}"
            )
        columns[target] = source

    def value(row: Row, target: str) -> object:
        raw = row.get(columns[target])
        return None if raw == "" else raw  # an empty CSV cell is an absent value

    def to_record(row: Row) -> TraceRecord:
        system, messages = split_prompt(_decoded(_required(row, "messages")))
        explicit_system = value(row, "system")
        return TraceRecord(
            trace_id=text(_required(row, "trace_id")),
            ts=parse_ts(_required(row, "ts")),
            model=text(value(row, "model") or UNKNOWN_MODEL),
            system=text(explicit_system) if explicit_system is not None else system,
            messages=messages,
            response=text(_required(row, "response")),
            response_tool_calls=tool_calls(_decoded(value(row, "response_tool_calls"))),
            prompt_tokens=as_int(value(row, "prompt_tokens") or 0),
            completion_tokens=as_int(value(row, "completion_tokens") or 0),
            latency_ms=optional_float(value(row, "latency_ms")) or 0.0,
            cost_usd=optional_float(value(row, "cost_usd")),
            session_id=_optional_text(value(row, "session_id")),
            outcome=optional_float(value(row, "outcome")),
            workload_hint=_optional_text(value(row, "workload_hint")),
        )

    def _required(row: Row, target: str) -> object:
        raw = value(row, target)
        if raw is None:
            raise MalformedRowError(f"missing {columns[target]}")
        return raw

    return Reader(tuple(columns[target] for target in REQUIRED_TARGETS), to_record)


def _optional_text(value: object) -> str | None:
    return None if value is None else text(value)
