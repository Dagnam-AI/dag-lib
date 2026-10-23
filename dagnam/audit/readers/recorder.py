"""The local recorder's trace file (envelope v1) -> :class:`TraceRecord`.

Input: one envelope per line, validated by ``dagnam_contracts.trace.parse_envelope``. A line
carries the call's raw ``request`` and ``response`` bodies, a provider-neutral ``norm`` block, or
both. Raw wins, because the audit's own parsers read every provider's spelling; ``norm`` is the
fallback for a line without raw (``--no-raw``) and for ``provider == "other"``, whose raw bodies
have no shape this reader knows.

| record field | source |
|---|---|
| kept rows | ``status`` is 200 (any other call is skipped, not malformed) |
| ``trace_id`` / ``ts`` / ``latency_ms`` | the envelope's, never the completion's |
| OpenAI raw | the OpenAI reader over ``{request, response}`` (Chat Completions and Responses) |
| Anthropic raw | ``system`` + ``messages`` in, ``content`` blocks and ``usage`` out |
| ``norm`` | the same fields as the recorder derived them, tool calls OpenAI-shaped |
| ``has_media`` | the prompt's media parts, or a ``media_omitted`` marker the recorder left in their place |
| newer ``v`` / ``norm_v`` | one malformed row, so it counts toward the export's 5%; on the first row, an unsupported export |

``session_id``, ``outcome`` and ``workload_hint`` come from OpenAI request ``metadata`` only;
``cost_usd`` is never present. ``signature`` needs the request, so a norm-only row has none.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from dagnam_contracts.trace import (
    EnvelopeError,
    Norm,
    TraceEnvelope,
    UnsupportedEnvelopeVersion,
    parse_envelope,
)

from dagnam.audit.readers.base import (
    UNKNOWN_MODEL,
    MalformedRowError,
    Reader,
    Row,
    cached_prompt_tokens,
    completion_tokens,
    parse_ts,
    prompt_tokens,
    text,
)
from dagnam.audit.readers.calls import tool_calls
from dagnam.audit.readers.messages import (
    final_reply,
    has_media,
    reply_text,
    split_prompt,
    task_signature,
)
from dagnam.audit.readers.openai_jsonl import to_record as openai_record
from dagnam.audit.record import Message, TraceRecord

REQUIRED_FIELDS = ("v", "trace_id", "ts")
_OMITTED = "media_omitted"


def _omitted(value: object) -> bool:
    """Whether the recorder replaced a media part with its size anywhere in ``value``."""
    if isinstance(value, Mapping):
        return value.get("type") == _OMITTED or any(_omitted(item) for item in value.values())
    return isinstance(value, list) and any(_omitted(item) for item in value)


def _from_openai(
    env: TraceEnvelope, request: dict[str, Any], response: dict[str, Any]
) -> TraceRecord | None:
    record = openai_record({"request": request, "response": response, "latency_ms": env.latency_ms})
    if record is None:
        return None
    return replace(
        record,
        trace_id=env.trace_id,
        ts=parse_ts(env.ts),
        latency_ms=env.latency_ms,
        has_media=record.has_media or _omitted(request),
    )


def _from_anthropic(
    env: TraceEnvelope, request: dict[str, Any], response: dict[str, Any]
) -> TraceRecord:
    system, messages = split_prompt(request)
    answer = final_reply(response)
    return TraceRecord(
        trace_id=env.trace_id,
        ts=parse_ts(env.ts),
        model=text(response.get("model") or request.get("model") or UNKNOWN_MODEL),
        system=system,
        messages=messages,
        response=answer.text,
        response_tool_calls=answer.calls,
        prompt_tokens=prompt_tokens(response, "usage"),
        completion_tokens=completion_tokens(response, "usage"),
        cached_prompt_tokens=cached_prompt_tokens(response, "usage"),
        latency_ms=env.latency_ms,
        cost_usd=None,
        session_id=None,
        outcome=None,
        workload_hint=None,
        has_media=has_media(request.get("messages")) or _omitted(request),
        signature=task_signature(request),
        reasoning_only=answer.reasoning_only,
    )


def _from_norm(env: TraceEnvelope, norm: Norm) -> TraceRecord:
    messages = tuple(Message(m.role, text(m.content)) for m in norm.messages)
    if all(m.role == "assistant" for m in messages):  # as the raw parsers: nothing to answer
        raise MalformedRowError("prompt has no user turn")
    return TraceRecord(
        trace_id=env.trace_id,
        ts=parse_ts(env.ts),
        model=text(norm.model),
        system=text(norm.system) or None,
        messages=messages,
        response=reply_text(norm.response),
        response_tool_calls=tool_calls(list(norm.tool_calls)),
        prompt_tokens=norm.prompt_tokens,
        completion_tokens=norm.completion_tokens,
        cached_prompt_tokens=norm.cached_prompt_tokens,
        latency_ms=env.latency_ms,
        cost_usd=None,
        session_id=None,
        outcome=None,
        workload_hint=None,
        has_media=norm.has_media,
    )


def to_record(row: Row) -> TraceRecord | None:
    """Convert one line; ``None`` for a call that did not return 200."""
    try:
        env = parse_envelope(row)
    except UnsupportedEnvelopeVersion as exc:
        raise MalformedRowError(f"{exc} (v={row.get('v')}, norm_v={row.get('norm_v')})") from exc
    if env.status != 200:
        return None
    request, response = env.request, env.response
    if request is not None and response is not None:
        if env.provider == "openai":
            return _from_openai(env, request, response)
        if env.provider == "anthropic":
            return _from_anthropic(env, request, response)
    if env.norm is None:
        raise MalformedRowError(f"no norm for a {env.provider} line with no readable raw")
    return _from_norm(env, env.norm)


def first_row_problem(row: Row) -> str | None:
    """Why an export whose first line is a newer envelope cannot be read, else ``None``.

    The header fields a newer line shares with v1 pass the field check, so without this a file
    written by a newer recorder would read as all malformed. Any other fault in the first line
    is left to the malformed share, like every later line's.
    """
    try:
        parse_envelope(row)
    except UnsupportedEnvelopeVersion:
        return (
            f"written by a newer recorder (v={row.get('v')}, norm_v={row.get('norm_v')}); "
            "upgrade dagnam to read it"
        )
    except EnvelopeError:
        pass
    return None


READER = Reader(REQUIRED_FIELDS, to_record, first_row_problem)
