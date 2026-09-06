"""OpenAI JSONL (Batch API and stored completions) -> :class:`TraceRecord`.

Input: one JSON object per line carrying **both** the request and the chat
completion it produced. The shapes accepted, from the Batch guide
(https://developers.openai.com/api/docs/guides/batch) and the chat completion
object reference (https://developers.openai.com/api/docs/api-reference/chat/object):

- a Batch **input** line (``{custom_id, method, url, body}``) joined with its
  **output** line (``{id, custom_id, response: {status_code, request_id,
  body}, error}``) on ``custom_id`` — the two files must be joined first,
  because neither alone holds a request/response pair;
- a ``{request, response}`` pair, where ``response`` is the completion object;
- a flat line that is a completion object with the request's ``messages``
  merged in (stored completions do not carry their input messages, so an
  export of them must add the messages the same way).

| record field | source (aliases in priority order) |
|---|---|
| kept rows | ``error`` is null and ``response.status_code``, when present, is 200 |
| request body | ``body`` / ``request`` / the row itself |
| completion | ``response.body`` / ``response`` / the row itself |
| ``trace_id`` | completion ``id`` / ``custom_id`` |
| ``session_id`` | request ``metadata.session_id`` / completion ``metadata.session_id`` |
| ``ts`` | completion ``created`` (Unix seconds) |
| ``latency_ms`` | top-level ``latency_ms`` when the exporter recorded one, else 0 |
| ``model`` | completion ``model`` / request ``model`` |
| ``system`` / ``messages`` | request ``messages`` |
| ``response`` | ``choices[0].message.content`` (+ ``tool_calls``) |
| ``prompt_tokens`` / ``completion_tokens`` | ``usage.prompt_tokens`` / ``usage.completion_tokens`` |
| ``cost_usd`` | never present (``None``) |
| ``outcome`` / ``workload_hint`` | ``metadata.outcome`` / ``metadata.workload`` on the request, then the completion |
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dagnam.audit.readers.base import (
    UNKNOWN_MODEL,
    Reader,
    Row,
    as_int,
    get,
    optional_float,
    parse_ts,
    require,
    split_prompt,
    text,
    tool_calls,
)
from dagnam.audit.record import TraceRecord

REQUIRED_FIELDS = ()


def _request(row: Row) -> Mapping[str, Any]:
    body = get(row, "body", "request")
    return body if isinstance(body, Mapping) else row


def _completion(row: Row) -> Mapping[str, Any]:
    body = get(row, "response.body", "response")
    return body if isinstance(body, Mapping) else row


def to_record(row: Row) -> TraceRecord | None:
    """Convert one line; ``None`` for a request that failed."""
    status = get(row, "response.status_code")
    if row.get("error") is not None or (status is not None and as_int(status) != 200):
        return None
    request, completion = _request(row), _completion(row)
    system, messages = split_prompt(require(request, "messages"))
    message = require(completion, "choices")[0]["message"]
    meta = (get(request, "metadata"), get(completion, "metadata"))
    metadata = {**_mapping(meta[1]), **_mapping(meta[0])}
    return TraceRecord(
        trace_id=text(
            require(completion, "id") if get(completion, "id") else require(row, "custom_id")
        ),
        ts=parse_ts(require(completion, "created")),
        model=text(get(completion, "model") or get(request, "model") or UNKNOWN_MODEL),
        system=system,
        messages=messages,
        response=text(get(message, "content")),
        response_tool_calls=tool_calls(get(message, "tool_calls")),
        prompt_tokens=as_int(get(completion, "usage.prompt_tokens") or 0),
        completion_tokens=as_int(get(completion, "usage.completion_tokens") or 0),
        latency_ms=optional_float(get(row, "latency_ms")) or 0.0,
        cost_usd=None,
        session_id=_optional_text(metadata.get("session_id")),
        outcome=optional_float(metadata.get("outcome")),
        workload_hint=_optional_text(metadata.get("workload")),
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_text(value: object) -> str | None:
    return None if value is None else text(value)


READER = Reader(REQUIRED_FIELDS, to_record)
