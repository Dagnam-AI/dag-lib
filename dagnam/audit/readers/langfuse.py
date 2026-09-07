"""Langfuse observation export -> :class:`TraceRecord`.

Input: an observations export (JSONL or Parquet; ``.gz`` accepted), one
observation per row, as produced by Langfuse's batch/blob-storage export and
its public observations API. Field mapping, from the observations schema in
the API reference (https://api.reference.langfuse.com/ — ``GET
/api/public/observations``) and the export guide
(https://langfuse.com/docs/api-and-data-platform/features/export-from-ui):

| record field | observation field (aliases in priority order) |
|---|---|
| kept rows | ``type == "GENERATION"`` with a non-null ``output`` |
| ``trace_id`` | ``id`` (the observation id, unique per call) |
| ``session_id`` | ``sessionId`` when the export carries it, else ``traceId`` (one trace per ticket/turn is how session grouping works in a Langfuse export) |
| ``ts`` | ``startTime`` |
| ``latency_ms`` | ``endTime - startTime``; else ``latency`` (documented in **seconds**) * 1000; else 0 |
| ``model`` | ``model`` |
| ``system`` / ``messages`` | ``input`` (a chat message list, a mapping holding ``messages``, or a bare prompt string) |
| ``response`` | ``output`` (a string, or the assistant message object ``{role, content, tool_calls}``) |
| ``prompt_tokens`` | any vendor spelling under ``usage`` / ``usageDetails`` / ``usageMetadata`` / the row itself (see :func:`~dagnam.audit.readers.base.prompt_tokens`) |
| ``completion_tokens`` | the same, for the completion count |
| ``cost_usd`` | ``calculatedTotalCost`` / ``costDetails.total`` / ``totalCost`` |
| ``outcome`` | ``metadata.outcome`` (a convention, not a Langfuse field) |
| ``workload_hint`` | ``metadata.workload`` (a convention, not a Langfuse field) |

Unknown fields are ignored.
"""

from __future__ import annotations

from datetime import datetime

from dagnam.audit.readers.base import (
    UNKNOWN_MODEL,
    Reader,
    Row,
    completion_tokens,
    get,
    optional_float,
    parse_ts,
    prompt_tokens,
    require,
    split_prompt,
    text,
    tool_calls,
)
from dagnam.audit.record import TraceRecord

REQUIRED_FIELDS = ("id", "type", "startTime", "input", "output")
# A Langfuse row carries the counts under ``usage`` (or the older
# ``usageDetails``), a Gemini-shaped export under ``usageMetadata``, and the
# legacy export spellings at the top level.
_USAGE_ROOTS = ("usage", "usageDetails", "usageMetadata", "")


def _latency_ms(row: Row, start: datetime) -> float:
    end = get(row, "endTime")
    if end is not None:
        return (parse_ts(end) - start).total_seconds() * 1000.0
    seconds = optional_float(get(row, "latency"))
    return 0.0 if seconds is None else seconds * 1000.0


def to_record(row: Row) -> TraceRecord | None:
    """Convert one observation row; ``None`` for rows that are not a finished generation."""
    if row.get("type") != "GENERATION" or row.get("output") is None:
        return None
    ts = parse_ts(require(row, "startTime"))
    system, messages = split_prompt(require(row, "input"))
    output = row["output"]
    if isinstance(output, dict):
        response, calls = text(output.get("content")), tool_calls(output.get("tool_calls"))
    else:
        response, calls = text(output), ()
    return TraceRecord(
        trace_id=text(require(row, "id")),
        ts=ts,
        model=text(get(row, "model") or UNKNOWN_MODEL),
        system=system,
        messages=messages,
        response=response,
        response_tool_calls=calls,
        prompt_tokens=prompt_tokens(row, *_USAGE_ROOTS),
        completion_tokens=completion_tokens(row, *_USAGE_ROOTS),
        latency_ms=_latency_ms(row, ts),
        cost_usd=optional_float(get(row, "calculatedTotalCost", "costDetails.total", "totalCost")),
        session_id=text(require(row, "sessionId", "traceId")),
        outcome=optional_float(get(row, "metadata.outcome")),
        workload_hint=_optional_text(get(row, "metadata.workload")),
    )


def _optional_text(value: object) -> str | None:
    return None if value is None else text(value)


READER = Reader(REQUIRED_FIELDS, to_record)
