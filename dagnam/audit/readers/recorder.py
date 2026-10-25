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
    require,
    text,
)
from dagnam.audit.readers.calls import tool_calls
from dagnam.audit.readers.messages import (
    content_text,
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
_SHOWN_VERSION_BELOW = 1_000
_VENDOR_RESULT = "_tool_result"


def _omitted(value: object) -> bool:
    """Whether the recorder replaced a media part with its size anywhere in ``value``."""
    if isinstance(value, Mapping):
        return value.get("type") == _OMITTED or any(_omitted(item) for item in value.values())
    return isinstance(value, list) and any(_omitted(item) for item in value)


def _vendor_result(part: object) -> bool:
    """A block that is the result of a tool the vendor ran (web search, code execution, MCP).

    The user's own ``tool_result`` does not end in ``_tool_result`` and is never one.
    """
    kind = part.get("type") if isinstance(part, Mapping) else None
    return isinstance(kind, str) and kind.endswith(_VENDOR_RESULT)


def _said(value: Any) -> Any:
    """``value`` without the vendor-run tool results among its content blocks, at any depth.

    What the assistant said is its text and its calls; the payload of a search it ran is not,
    and the shared parsers would print it as JSON into the reply. Applied to the assistant's side
    of a call only, so every other source and the user's turns read as before.
    """
    if isinstance(value, list):
        return [_said(part) for part in value if not _vendor_result(part)]
    if isinstance(value, Mapping):
        return {key: _said(part) for key, part in value.items()}
    return value


def _assistant_side(
    request: dict[str, Any], response: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The call's bodies with the vendor-run tool results removed from what the assistant said."""
    for key in ("messages", "input"):
        turns = request.get(key)
        if isinstance(turns, list):
            turns = [
                _said(turn)
                if isinstance(turn, Mapping) and turn.get("role") == "assistant"
                else turn
                for turn in turns
            ]
            request = {**request, key: turns}
    return request, _said(response)


def _parse_schema(request: Mapping[str, Any]) -> str | None:
    """The schema a Responses ``parse()`` call asked for: the recorder keeps its ``text_format``."""
    schema = request.get("text_format")
    return task_signature({"text": {"format": schema}}) if isinstance(schema, Mapping) else None


def _from_openai(
    env: TraceEnvelope, request: dict[str, Any], response: dict[str, Any]
) -> TraceRecord | None:
    # The shared parser wants the completion's own id and creation time, which the envelope
    # replaces below; a body without them (a proxy's, a trimmed one) is still a whole call.
    completion = {
        **response,
        "id": response.get("id") or env.trace_id,
        "created": response.get("created") or response.get("created_at") or env.ts,
    }
    request, completion = _assistant_side(request, completion)
    record = openai_record(
        {"request": request, "response": completion, "latency_ms": env.latency_ms}
    )
    if record is None:
        return None
    return replace(
        record,
        signature=record.signature or _parse_schema(request),
        trace_id=env.trace_id,
        ts=parse_ts(env.ts),
        latency_ms=env.latency_ms,
        has_media=record.has_media or _omitted(request),
    )


def _from_anthropic(
    env: TraceEnvelope, request: dict[str, Any], response: dict[str, Any]
) -> TraceRecord:
    # As the OpenAI parser: split_prompt alone would read a request with no ``messages`` whole as
    # the prompt, and a response with no ``content`` would be an empty answer for 0 tokens.
    require(request, "messages")
    require(response, "content")
    request, response = _assistant_side(request, response)
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
    # Each turn is read as the raw parsers read it: an assistant turn without the reasoning the
    # model wrote before its answer (the recorder already put its tool calls in the text), any
    # other as content, so a text that is JSON of a message reads the same from either form.
    messages = tuple(
        Message(m.role, reply_text(m.content) if m.role == "assistant" else content_text(m.content))
        for m in norm.messages
    )
    if all(m.role == "assistant" for m in messages):  # as the raw parsers: nothing to answer
        raise MalformedRowError("prompt has no user turn")
    return TraceRecord(
        trace_id=env.trace_id,
        ts=parse_ts(env.ts),
        model=text(norm.model),
        system=content_text(norm.system) or None,
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


def _versions(row: Row) -> str:
    """`` (v=2, norm_v=1)`` naming the version fields that hold a small integer, else nothing.

    A version field is the file's own text, so any other value (a string, an object, a number
    no recorder wrote) is left out of a message: these refusals quote nothing from the export.
    """
    shown = [
        f"{key}={row[key]}"
        for key in ("v", "norm_v")
        if type(row.get(key)) is int and 0 <= row[key] < _SHOWN_VERSION_BELOW
    ]
    return f" ({', '.join(shown)})" if shown else ""


def to_record(row: Row) -> TraceRecord | None:
    """Convert one line; ``None`` for a call that did not return 200."""
    # ``row`` is already parsed JSON, so the contract's 1 MB line cap is not applied here, as for
    # the other readers; the recorder drops an oversize line before it is written.
    try:
        env = parse_envelope(row)
    except UnsupportedEnvelopeVersion as exc:
        raise MalformedRowError(f"{exc}{_versions(row)}") from exc
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
        return f"written by a newer recorder{_versions(row)}; upgrade dagnam to read it"
    except EnvelopeError:
        pass
    return None


READER = Reader(REQUIRED_FIELDS, to_record, first_row_problem, reads_directory=True)
