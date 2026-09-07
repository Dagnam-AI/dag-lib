"""LangSmith run export -> :class:`TraceRecord`.

Input: a run export (JSONL or Parquet; ``.gz`` accepted), one run per row.
Field mapping, from the run data format reference
(https://docs.langchain.com/langsmith/run-data-format) and the threads guide
(https://docs.langchain.com/langsmith/threads):

| record field | run field (aliases in priority order) |
|---|---|
| kept rows | ``run_type == "llm"`` |
| ``trace_id`` | ``id`` (the run id, unique per call) |
| ``session_id`` | ``extra.metadata.session_id`` / ``extra.metadata.thread_id`` (the documented thread keys), else ``trace_id``. LangSmith's own top-level ``session_id`` is the tracing *project* id and is deliberately not used |
| ``ts`` | ``start_time`` |
| ``latency_ms`` | ``end_time - start_time``; 0 when ``end_time`` is absent |
| ``model`` | ``extra.invocation_params.model`` / ``extra.invocation_params.model_name`` / ``extra.metadata.ls_model_name`` / ``outputs.model`` |
| ``system`` / ``messages`` | ``inputs.messages`` / ``inputs.contents``: OpenAI-style ``{role, content}`` dicts (the ``wrap_openai`` shape, and what ``wrap_anthropic`` produces once it folds its ``system`` kwarg in as a system turn), LangChain-serialized messages (``{"lc": 1, "id": [..., "HumanMessage"], "kwargs": {"content"}}``, possibly nested one list deep), or Gemini ``{role, parts: [{text}]}`` turns; a bare ``inputs.prompt`` / ``inputs.input`` string is one user turn |
| ``response`` | ``outputs.choices[0].message`` / ``outputs.messages[-1]`` / ``outputs.generations[0][0]`` (``text`` or its serialized ``message``) / ``outputs`` itself when it carries ``content`` (the ``wrap_anthropic`` / ``wrap_gemini`` shape: a string, or typed blocks whose ``text`` is joined) / ``outputs.output`` |
| ``prompt_tokens`` | any vendor spelling at the top level or under ``usage_metadata`` / ``outputs.usage`` / ``outputs.usage_metadata`` / ``outputs.llm_output.token_usage`` (see :func:`~dagnam.audit.readers.base.prompt_tokens`) |
| ``completion_tokens`` | the same, for the completion count |
| ``cost_usd`` | ``total_cost`` |
| ``outcome`` | ``extra.metadata.outcome`` (a convention, not a LangSmith field) |
| ``workload_hint`` | ``extra.metadata.workload`` (a convention, not a LangSmith field) |

Unknown fields are ignored.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dagnam.audit.readers.base import (
    UNKNOWN_MODEL,
    MalformedRowError,
    Reader,
    Row,
    completion_tokens,
    content_text,
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

REQUIRED_FIELDS = ("id", "run_type", "start_time", "inputs", "outputs")

_LC_ROLES = {"SystemMessage": "system", "HumanMessage": "user", "AIMessage": "assistant"}
# The run's own counts first, then LangChain's ``usage_metadata`` and the
# per-provider usage blocks the run carries through from the response.
_USAGE_ROOTS = (
    "",
    "usage_metadata",
    "outputs.usage",
    "outputs.usage_metadata",
    "outputs.llm_output.token_usage",
)


def _plain_message(message: object) -> Any:
    """Turn a LangChain-serialized or Gemini ``parts`` message into ``{role, content}``.

    Anything else passes through: OpenAI-style ``{role, content}`` dicts are
    already in the target shape.
    """
    if isinstance(message, Mapping) and "kwargs" in message:
        kind = text(message.get("id", ["?"])[-1])
        return {
            "role": _LC_ROLES.get(kind, kind.lower()),
            "content": get(message, "kwargs.content"),
            "tool_calls": get(message, "kwargs.tool_calls"),
        }
    if isinstance(message, Mapping) and "parts" in message:
        return {"role": message.get("role", "user"), "content": message["parts"]}
    return message


def _messages(inputs: Mapping[str, Any]) -> list[Any] | str:
    # ``wrap_gemini`` normalizes ``contents`` to ``messages``; a run that kept the
    # raw Gemini request carries ``contents`` instead.
    raw = get(inputs, "messages", "contents")
    if raw is None:
        return text(require(inputs, "prompt", "input"))
    if not isinstance(raw, list):
        raise MalformedRowError(f"inputs.messages is not a list: {raw!r}")
    if raw and isinstance(raw[0], list):  # LangChain nests one batch of messages per prompt
        raw = raw[0]
    messages = [_plain_message(m) for m in raw]
    # ``wrap_gemini`` leaves the system prompt in ``config.system_instruction``
    # rather than as a turn; without it every step of an agent hashes to the
    # same template and collapses into one workload.
    system = get(inputs, "config.system_instruction")
    if system is not None:
        messages.insert(
            0,
            {
                "role": "system",
                "content": _plain_message(system)["content"]
                if isinstance(system, Mapping) and "parts" in system
                else system,
            },
        )
    return messages


def _response(outputs: Mapping[str, Any]) -> Any:
    """The assistant reply as an OpenAI-style message object, or a bare string."""
    choice = get(outputs, "choices")
    if isinstance(choice, list) and choice:
        return get(choice[0], "message")
    messages = get(outputs, "messages")
    if isinstance(messages, list) and messages:
        return _plain_message(messages[-1])
    generations = get(outputs, "generations")
    if isinstance(generations, list) and generations:
        first = generations[0][0] if isinstance(generations[0], list) else generations[0]
        message = get(first, "message")
        return _plain_message(message) if message is not None else get(first, "text")
    if get(outputs, "content") is not None:
        # ``wrap_anthropic`` / ``wrap_gemini`` dump the reply message straight into
        # ``outputs``: ``content`` (a string or typed blocks) beside ``tool_calls``.
        return outputs
    return get(outputs, "output")


def to_record(row: Row) -> TraceRecord | None:
    """Convert one run row; ``None`` for runs that are not LLM calls."""
    if row.get("run_type") != "llm":
        return None
    ts = parse_ts(require(row, "start_time"))
    inputs, outputs = require(row, "inputs"), require(row, "outputs")
    system, messages = split_prompt(_messages(inputs))
    reply = _response(outputs)
    if isinstance(reply, Mapping):
        response = content_text(reply.get("content"))
        calls = tool_calls(reply.get("tool_calls"))
    else:
        response, calls = text(reply), ()
    end = get(row, "end_time")
    return TraceRecord(
        trace_id=text(require(row, "id")),
        ts=ts,
        model=text(
            get(
                row,
                "extra.invocation_params.model",
                "extra.invocation_params.model_name",
                "extra.metadata.ls_model_name",
                "outputs.model",
            )
            or UNKNOWN_MODEL
        ),
        system=system,
        messages=messages,
        response=response,
        response_tool_calls=calls,
        prompt_tokens=prompt_tokens(row, *_USAGE_ROOTS),
        completion_tokens=completion_tokens(row, *_USAGE_ROOTS),
        latency_ms=0.0 if end is None else (parse_ts(end) - ts).total_seconds() * 1000.0,
        cost_usd=optional_float(get(row, "total_cost")),
        session_id=text(
            require(row, "extra.metadata.session_id", "extra.metadata.thread_id", "trace_id")
        ),
        outcome=optional_float(get(row, "extra.metadata.outcome")),
        workload_hint=_optional_text(get(row, "extra.metadata.workload")),
    )


def _optional_text(value: object) -> str | None:
    return None if value is None else text(value)


READER = Reader(REQUIRED_FIELDS, to_record)
