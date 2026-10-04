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
| shapes | Chat Completions, and the Responses API (``instructions`` + ``input``; an ``output`` item list; ``created_at``) |
| request body | ``body`` / ``request`` / the row itself |
| completion | ``response.body`` / ``response`` / the row itself |
| ``trace_id`` | completion ``id`` / ``custom_id`` |
| ``session_id`` | request ``metadata.session_id`` / completion ``metadata.session_id`` |
| ``ts`` | completion ``created`` / ``created_at`` (Unix seconds) |
| ``latency_ms`` | top-level ``latency_ms`` when the exporter recorded one, else 0 |
| ``model`` | completion ``model`` / request ``model`` |
| ``system`` / ``messages`` | request ``messages``, or Responses ``instructions`` / ``input`` |
| ``response`` | ``choices[0].message`` (``content`` without inline ``<think>``, + ``tool_calls``), or the Responses ``output`` items |
| ``prompt_tokens`` / ``completion_tokens`` | ``usage.prompt_tokens`` / ``usage.completion_tokens`` |
| ``cached_prompt_tokens`` | ``usage.prompt_tokens_details.cached_tokens`` (already inside ``prompt_tokens``) |
| ``cost_usd`` | never present (``None``) |
| ``outcome`` / ``workload_hint`` | ``metadata.outcome`` / ``metadata.workload`` on the request, then the completion; a Responses request's ``prompt.id`` names the workload too |
| ``has_media`` / ``signature`` | an image/audio/file part in the prompt; the request's ``response_format`` schema or forced ``tool_choice`` name (never its ``tools``) |
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dagnam.audit.readers.base import (
    UNKNOWN_MODEL,
    Reader,
    Row,
    as_int,
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
from dagnam.audit.readers.messages import final_reply, has_media, split_prompt, task_signature
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
    prompt = require(request, "messages", "input")
    system, messages = split_prompt(request)
    choices = get(completion, "choices")
    # A Responses API completion has no ``choices``: its reply is the ``output`` item list.
    reply = choices[0]["message"] if choices is not None else require(completion, "output")
    answer = final_reply(reply)
    meta = (get(request, "metadata"), get(completion, "metadata"))
    metadata = {**_mapping(meta[1]), **_mapping(meta[0])}
    return TraceRecord(
        trace_id=text(
            require(completion, "id") if get(completion, "id") else require(row, "custom_id")
        ),
        ts=parse_ts(require(completion, "created", "created_at")),
        model=text(get(completion, "model") or get(request, "model") or UNKNOWN_MODEL),
        system=system,
        messages=messages,
        response=answer.text,
        response_tool_calls=answer.calls,
        prompt_tokens=prompt_tokens(completion, "usage"),
        completion_tokens=completion_tokens(completion, "usage"),
        cached_prompt_tokens=cached_prompt_tokens(completion, "usage"),
        latency_ms=optional_float(get(row, "latency_ms")) or 0.0,
        cost_usd=None,
        session_id=_optional_text(metadata.get("session_id")),
        outcome=optional_outcome(metadata.get("outcome")),
        workload_hint=_optional_text(metadata.get("workload") or get(request, "prompt.id")),
        has_media=has_media(prompt),
        signature=task_signature(request),
        reasoning_only=answer.reasoning_only,
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_text(value: object) -> str | None:
    return None if value is None else text(value)


READER = Reader(REQUIRED_FIELDS, to_record)
