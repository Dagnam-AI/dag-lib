"""Langfuse observation export -> :class:`TraceRecord`.

Input: an observations export, one observation per row, in any of the shapes
Langfuse hands out (``.gz`` accepted):
- the public observations API (JSONL, camelCase), as
  ``scripts/audit_fixture/export_traces.py`` writes it;
- the blob-storage export (``observations_v2/``: Parquet, CSV, JSON or JSONL),
  in snake_case with ``input`` / ``output`` / ``metadata`` as JSON strings;
- the UI's CSV or JSON export (camelCase, the same JSON strings; JSON is one
  array).

Field mapping, from the observations schema in the API reference
(https://api.reference.langfuse.com/ — ``GET /api/public/observations``), the
blob export (https://langfuse.com/docs/api-and-data-platform/features/export-to-blob-storage)
and the UI export guide
(https://langfuse.com/docs/api-and-data-platform/features/export-from-ui):

| record field | observation field (aliases in priority order) |
|---|---|
| kept rows | ``type == "GENERATION"`` with a non-null ``output`` |
| ``trace_id`` | ``id`` (the observation id, unique per call) |
| ``session_id`` | ``sessionId`` / ``session_id`` when the export carries it, else ``traceId`` / ``trace_id`` (one trace per ticket/turn is how session grouping works in a Langfuse export) |
| ``ts`` | ``startTime`` / ``start_time`` |
| ``latency_ms`` | ``endTime - startTime``; else ``latency`` (documented in **seconds**) * 1000; else 0 |
| ``model`` | ``model`` / ``providedModelName`` / ``provided_model_name`` |
| ``system`` / ``messages`` | ``input``, decoded when it is a JSON string (a chat message list; a mapping holding ``messages`` with the Anthropic ``system`` beside them, or the Responses API's ``input`` items; or a bare prompt string) |
| ``response`` | ``output``, decoded when it is a JSON string: a string, the assistant message ``{role, content, tool_calls}`` whose ``content`` may be typed blocks, a Responses ``message`` / ``function_call`` item or Responses object, or a list of messages whose last is the reply; any other object (``{"type": "refund", ...}``) is the answer itself, as JSON text |
| ``prompt_tokens`` | any vendor spelling under ``usage`` / ``usageDetails`` / ``usage_details`` / ``usageMetadata`` / the row itself (see :func:`~dagnam.audit.readers.base.prompt_tokens`); Langfuse's ``input_*`` buckets are added to ``input``, which excludes them |
| ``completion_tokens`` | the same, for the completion count |
| ``cached_prompt_tokens`` | the cache-read count in any vendor spelling (see :func:`~dagnam.audit.readers.base.cached_prompt_tokens`) |
| ``cost_usd`` | ``calculatedTotalCost`` / ``costDetails.total`` / ``cost_details.total`` / ``totalCost`` / ``total_cost`` |
| ``outcome`` | ``metadata.outcome`` (a convention, not a Langfuse field) |
| ``workload_hint`` | ``metadata.workload`` (a convention), else the registered prompt's ``promptName`` / ``prompt_name`` |
| ``has_media`` / ``signature`` | an image/audio/file part in ``input``; the ``response_format`` schema or forced ``tool_choice`` name beside the input items (never the offered ``tools``) |

Unknown fields are ignored.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import json
from typing import Any

from dagnam.audit.readers.base import (
    UNKNOWN_MODEL,
    Reader,
    Row,
    cached_prompt_tokens,
    completion_tokens,
    get,
    optional_float,
    optional_outcome,
    parse_ts,
    prompt_tokens,
    require,
    text,
)
from dagnam.audit.readers.messages import (
    has_media,
    is_reply,
    reply_of,
    split_prompt,
    task_signature,
)
from dagnam.audit.record import TraceRecord

REQUIRED_FIELDS = ("id", "type", "input", "output")
# A Langfuse row carries the counts under ``usage`` (or ``usageDetails``, or
# the blob export's ``usage_details``), a Gemini-shaped export under
# ``usageMetadata``, and the legacy export spellings at the top level.
_USAGE_ROOTS = ("usage", "usageDetails", "usage_details", "usageMetadata", "")
_JSON_FIELDS = ("input", "usage", "usageDetails", "usage_details", "metadata")
"""Fields the UI and blob exports store as JSON strings; ``output`` is decoded apart."""


def _decoded(value: object) -> Any:
    """A JSON-string field decoded; anything else, or text that is not JSON, as it came."""
    if isinstance(value, str) and value[:1] in ("[", "{"):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _output(value: object) -> Any:
    """The reply: a message, output item or Responses object, decoded when it is a JSON string.

    An answer that is itself a JSON object is its JSON text (B2-1).
    """
    decoded = _decoded(value)
    if isinstance(decoded, Mapping):
        return decoded if is_reply(decoded) else text(value)
    if (
        isinstance(decoded, list)
        and decoded
        and all(isinstance(item, Mapping) and "role" in item for item in decoded)
    ):
        return decoded[-1]
    if (
        isinstance(decoded, list)
        and decoded
        and all(
            isinstance(item, Mapping)
            and item.get("type") in {"reasoning", "message", "function_call"}
            for item in decoded
        )
    ):
        return decoded
    return text(value) if isinstance(decoded, list) else value


def _latency_ms(row: Row, start: datetime) -> float:
    end = get(row, "endTime", "end_time")
    if end is not None:
        return (parse_ts(end) - start).total_seconds() * 1000.0
    seconds = optional_float(get(row, "latency"))
    return 0.0 if seconds is None else seconds * 1000.0


def to_record(row: Row) -> TraceRecord | None:
    """Convert one observation row; ``None`` for rows that are not a finished generation."""
    if row.get("type") != "GENERATION" or row.get("output") is None:
        return None
    row = {**row, **{key: _decoded(row[key]) for key in _JSON_FIELDS if key in row}}
    ts = parse_ts(require(row, "startTime", "start_time"))
    prompt = require(row, "input")
    system, messages = split_prompt(prompt)
    response, calls = reply_of(_output(row["output"]))
    return TraceRecord(
        trace_id=text(require(row, "id")),
        ts=ts,
        model=text(get(row, "model", "providedModelName", "provided_model_name") or UNKNOWN_MODEL),
        system=system,
        messages=messages,
        response=response,
        response_tool_calls=calls,
        prompt_tokens=prompt_tokens(row, *_USAGE_ROOTS),
        completion_tokens=completion_tokens(row, *_USAGE_ROOTS),
        cached_prompt_tokens=cached_prompt_tokens(row, *_USAGE_ROOTS),
        latency_ms=_latency_ms(row, ts),
        cost_usd=optional_float(
            get(
                row,
                "calculatedTotalCost",
                "costDetails.total",
                "cost_details.total",
                "totalCost",
                "total_cost",
            )
        ),
        session_id=text(require(row, "sessionId", "session_id", "traceId", "trace_id")),
        outcome=optional_outcome(get(row, "metadata.outcome")),
        workload_hint=_optional_text(get(row, "metadata.workload", "promptName", "prompt_name")),
        has_media=has_media(prompt),
        signature=task_signature(prompt),
    )


def _optional_text(value: object) -> str | None:
    return None if value is None else text(value)


READER = Reader(REQUIRED_FIELDS, to_record)
