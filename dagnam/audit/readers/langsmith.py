"""LangSmith run export -> :class:`TraceRecord`.

Input: a run export (JSONL or Parquet; ``.gz`` accepted), one run per row.
Field mapping, from the run data format reference
(https://docs.langchain.com/langsmith/run-data-format) and the threads guide
(https://docs.langchain.com/langsmith/threads):

| record field | run field (aliases in priority order) |
|---|---|
| kept rows | ``run_type == "llm"`` with a non-empty ``outputs``, no ``error`` and a ``status`` other than ``"error"`` |
| ``trace_id`` | ``id`` (the run id, unique per call) |
| ``session_id`` | ``extra.metadata.session_id`` / ``extra.metadata.thread_id`` (the documented thread keys), else ``trace_id``. LangSmith's own top-level ``session_id`` is the tracing *project* id and is deliberately not used |
| ``ts`` | ``start_time`` |
| ``latency_ms`` | ``end_time - start_time``; 0 when ``end_time`` is absent |
| ``model`` | ``extra.invocation_params.model`` / ``extra.invocation_params.model_name`` / ``extra.metadata.ls_model_name`` / ``outputs.model`` |
| ``system`` / ``messages`` | the Responses API's ``inputs.instructions`` / ``inputs.input``, else ``inputs.messages`` / ``inputs.contents``: OpenAI-style ``{role, content}`` dicts (the ``wrap_openai`` shape, and what ``wrap_anthropic`` produces once it folds its ``system`` kwarg in as a system turn), LangChain-serialized messages (``{"lc": 1, "id": [..., "HumanMessage"], "kwargs": {"content"}}``, possibly nested one list deep), or Gemini ``{role, parts: [{text}]}`` turns; a bare ``inputs.prompt`` / ``inputs.input`` string is one user turn |
| ``response`` | ``outputs.choices[0].message`` / ``outputs.messages[-1]`` / ``outputs.generations[0][0]`` (``text`` or its serialized ``message``) / ``outputs`` itself when it carries ``content`` (the ``wrap_anthropic`` / ``wrap_gemini`` shape: a string, or typed blocks whose ``text`` is joined, ``thinking`` and Gemini ``thought`` parts dropped, ``tool_use`` blocks read as tool calls) / ``outputs.output`` (a string, or the Responses API's item list) |
| ``prompt_tokens`` | any vendor spelling at the top level or under ``usage_metadata`` / ``outputs.usage`` / ``outputs.usage_metadata`` / ``outputs.llm_output.token_usage`` (see :func:`~dagnam.audit.readers.base.prompt_tokens`) |
| ``completion_tokens`` | the same, for the completion count |
| ``cached_prompt_tokens`` | the cache-read count in any vendor spelling (see :func:`~dagnam.audit.readers.base.cached_prompt_tokens`) |
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
    Reply,
    final_reply,
    has_media,
    is_reply,
    response_of,
    split_prompt,
    task_signature,
)
from dagnam.audit.readers.reasoning import carries_reasoning
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


def _messages(inputs: Mapping[str, Any]) -> list[Any] | str | Mapping[str, Any]:
    # ``wrap_gemini`` normalizes ``contents`` to ``messages``; a run that kept the
    # raw Gemini request carries ``contents`` instead.
    raw = get(inputs, "messages", "contents")
    if raw is None and (
        get(inputs, "instructions") is not None or isinstance(get(inputs, "input"), list)
    ):
        return inputs  # the Responses API: ``instructions`` beside the ``input`` items
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


def _response(outputs: Mapping[str, Any]) -> Reply:
    """The assistant reply of a run's ``outputs``: a message, an envelope that wraps one, or text."""
    choice = get(outputs, "choices")
    if isinstance(choice, list) and choice and get(choice[0], "message") is not None:
        return final_reply(get(choice[0], "message"))
    messages = get(outputs, "messages")
    if isinstance(messages, list) and messages:
        return final_reply(_plain_message(messages[-1]))
    generations = get(outputs, "generations")
    if isinstance(generations, list) and generations:
        first = generations[0][0] if isinstance(generations[0], list) else generations[0]
        message = get(first, "message")
        return final_reply(_plain_message(message) if message is not None else get(first, "text"))
    if get(outputs, "content") is not None or is_reply(outputs):
        # ``wrap_anthropic`` / ``wrap_gemini`` dump the reply message straight into
        # ``outputs``: ``content`` (a string or typed blocks) beside ``tool_calls``. So does
        # a client that logs a whole response object (Ollama, Gemini, Bedrock...).
        return final_reply(outputs)
    output = get(outputs, "output")
    if output is None and carries_reasoning(outputs, strict=True):
        # A response object nobody recognises, with reasoning in it: kept as a call, no answer,
        # like every other reader (never an empty reply that the warning leaves out).
        return Reply("", (), reasoning_only=True)
    # ``outputs.output`` is a string, a Responses item list, or an answer that is JSON itself.
    return response_of(output)


def to_record(row: Row) -> TraceRecord | None:
    """Convert one run row; ``None`` for runs that are not LLM calls or that errored.

    An errored run (a 429, a timeout) carries ``outputs: null`` or an empty
    ``outputs``: it is a call that produced no answer, skipped the way the
    Langfuse and OpenAI readers skip theirs, never a malformed row.
    """
    if row.get("run_type") != "llm":
        return None
    if not row.get("outputs") or row.get("error") or row.get("status") == "error":
        return None
    ts = parse_ts(require(row, "start_time"))
    inputs, outputs = require(row, "inputs"), row["outputs"]
    prompt = _messages(inputs)
    system, messages = split_prompt(prompt)
    reply = _response(outputs)
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
        response=reply.text,
        response_tool_calls=reply.calls,
        prompt_tokens=prompt_tokens(row, *_USAGE_ROOTS),
        completion_tokens=completion_tokens(row, *_USAGE_ROOTS),
        cached_prompt_tokens=cached_prompt_tokens(row, *_USAGE_ROOTS),
        latency_ms=0.0 if end is None else (parse_ts(end) - ts).total_seconds() * 1000.0,
        cost_usd=optional_float(get(row, "total_cost")),
        session_id=text(
            require(row, "extra.metadata.session_id", "extra.metadata.thread_id", "trace_id")
        ),
        outcome=optional_outcome(get(row, "extra.metadata.outcome")),
        workload_hint=_optional_text(get(row, "extra.metadata.workload")),
        has_media=has_media(prompt),
        signature=task_signature(inputs) or task_signature(get(row, "extra.invocation_params")),
        reasoning_only=reply.reasoning_only,
    )


def _optional_text(value: object) -> str | None:
    return None if value is None else text(value)


READER = Reader(REQUIRED_FIELDS, to_record)
