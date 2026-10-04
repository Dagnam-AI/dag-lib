"""How a Langfuse generation's output is read as a reply: shapes, JSON text, reasoning."""

from __future__ import annotations

import json

import pytest

from dagnam.audit import TraceRecord
from dagnam.audit.readers import langfuse
from dagnam.audit.readers.messages import effective_response


def _generation() -> dict[str, object]:
    return {
        "id": "gen-1",
        "traceId": "trace-1",
        "type": "GENERATION",
        "startTime": "2026-08-01T09:00:00.000Z",
        "endTime": "2026-08-01T09:00:00.250Z",
        "model": "gpt-4o-mini",
        "input": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        "output": {"role": "assistant", "content": "hello"},
        "usage": {"input": 5, "output": 1},
        "calculatedTotalCost": 0.001,
    }


AI_MESSAGE = {
    "content": "billing",
    "additional_kwargs": {},
    "response_metadata": {"model_name": "gpt-4o-mini"},
    "type": "ai",
    "name": None,
    "id": "run-123",
    "tool_calls": [],
}
"""A LangChain ``AIMessage`` dump, as Langfuse stores one an ``@observe`` generation returned."""


@pytest.mark.parametrize(
    ("output", "response"),
    [
        # Role-less, and read whole as JSON -- its per-call ``id`` made every
        # answer distinct; it is a reply because it carries ``content``.
        (AI_MESSAGE, "billing"),
        # Answers that only look like replies stay the model's JSON answer.
        ({"output": ["a", "b"]}, json.dumps({"output": ["a", "b"]})),
        ({"name": "Bob", "role": "admin"}, json.dumps({"name": "Bob", "role": "admin"})),
        (
            {"type": "message", "priority": "high"},
            json.dumps({"type": "message", "priority": "high"}),
        ),
    ],
    ids=["langchain-ai-message", "output-list-answer", "role-field-answer", "type-field-answer"],
)
def test_a_reply_is_what_carries_a_reply_and_the_rest_is_the_answer(
    output: dict[str, object], response: str
) -> None:
    for exported in (output, json.dumps(output)):
        record = langfuse.to_record({**_generation(), "output": exported})
        assert record is not None
        assert record.response == response


def test_priority_buckets_are_not_added_twice() -> None:
    # Langfuse's own SDK marks ``input_priority*`` / ``output_priority*`` as not
    # exclusive (its CallbackHandler never subtracts them): they count tokens the
    # other buckets already hold. 1,800 prompt tokens were read for 900.
    usage = {
        "input": 100,
        "input_cache_read": 800,
        "input_priority": 900,
        "input_priority_cache_read": 800,
        "output": 5,
        "output_priority": 5,
        "input_cost": 0.1,
    }
    record = langfuse.to_record({**_generation(), "usage_details": usage, "usage": None})
    assert record is not None
    assert (record.prompt_tokens, record.cached_prompt_tokens, record.completion_tokens) == (
        900,
        800,
        5,
    )


def test_a_prompt_that_only_looks_like_json_stays_a_prompt() -> None:
    record = langfuse.to_record({**_generation(), "input": "{not json, a plain prompt"})
    assert record is not None
    assert record.messages[0].content == "{not json, a plain prompt"


def test_a_text_outcome_in_the_metadata_is_ignored() -> None:
    record = langfuse.to_record({**_generation(), "metadata": {"outcome": "resolved"}})
    assert record is not None
    assert record.outcome is None


@pytest.mark.parametrize("serialized", [False, True])
def test_responses_output_array_keeps_calls_and_drops_reasoning(serialized: bool) -> None:
    output = [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "secret"}]},
        {"type": "function_call", "name": "route", "arguments": '{"team":"billing"}'},
    ]
    record = langfuse.to_record(
        {**_generation(), "output": json.dumps(output) if serialized else output}
    )
    assert record is not None
    assert record.response == ""
    assert (
        effective_response(record.response, record.response_tool_calls)
        == '{"arguments": {"team": "billing"}, "name": "route"}'
    )


@pytest.mark.parametrize("serialized", [False, True])
def test_serialized_arbitrary_array_stays_answer_text(serialized: bool) -> None:
    output = '[{"type": "refund", "amount": 12}, {"type": "message", "priority": "high"}]'
    record = langfuse.to_record(
        {**_generation(), "output": output if serialized else json.loads(output)}
    )
    assert record is not None
    assert record.response == output
    assert record.response_tool_calls == ()


REASONING = {
    "type": "reasoning",
    "id": "rs_1",
    "summary": [{"type": "summary_text", "text": "the customer sounds angry"}],
    "encrypted_content": "gAAAAB-opaque-reasoning",
}
ANSWER = {
    "type": "message",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "billing"}],
}


def _leaks(record: TraceRecord) -> bool:
    kept = json.dumps([record.response, record.response_tool_calls])
    return any(secret in kept for secret in ("angry", "gAAAAB", "rs_1"))


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize(
    "other",
    [
        {"type": "web_search_call", "id": "ws_1", "status": "completed"},
        {"type": "mcp_call", "id": "mcp_1", "name": "lookup", "arguments": "{}", "output": "ok"},
        {"type": "custom_tool_call", "call_id": "c1", "name": "run", "input": "ls"},
        {"type": "file_search_call", "id": "fs_1", "queries": ["refund policy"]},
        {"type": "a_type_no_reader_knows", "payload": 1},
        "a bare string",
    ],
    ids=["web-search", "mcp", "custom-tool", "file-search", "unknown", "not-an-object"],
)
def test_reasoning_never_reaches_the_reply_whatever_else_the_output_array_holds(
    other: object, serialized: bool
) -> None:
    # The array was a reply only when EVERY item was reasoning, a message or a function
    # call: one hosted-tool item beside them and the whole array, reasoning summary and
    # encrypted reasoning included, was serialized as the answer and uploaded.
    output = [REASONING, other, ANSWER]
    record = langfuse.to_record(
        {**_generation(), "output": json.dumps(output) if serialized else output}
    )
    assert record is not None
    assert not _leaks(record)
    assert record.response == ("a bare stringbilling" if isinstance(other, str) else "billing")


@pytest.mark.parametrize(
    ("output", "response"),
    [
        (
            [
                {"type": "thinking", "thinking": "the customer sounds angry", "signature": "s"},
                {"type": "redacted_thinking", "data": "gAAAAB"},
                {"type": "text", "text": "billing"},
            ],
            "billing",
        ),
        ([{"text": "the customer sounds angry", "thought": True}, {"text": "billing"}], "billing"),
    ],
    ids=["anthropic-blocks", "gemini-parts"],
)
def test_a_bare_array_holding_reasoning_is_read_as_reply_parts(
    output: list[object], response: str
) -> None:
    for exported in (output, json.dumps(output)):
        record = langfuse.to_record({**_generation(), "output": exported})
        assert record is not None
        assert record.response == response
        assert record.response_tool_calls == ()
        assert not _leaks(record)


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize(
    "output",
    [
        [REASONING, {"type": "web_search_call", "id": "ws_1"}],  # no answer at all
        # An array item typed as reasoning is reasoning, whatever it carries: the array is
        # never serialized whole. An answer that is a JSON array pays for it only when one
        # of its own objects says ``"type": "reasoning"``: no row, not a leaked one.
        [{"type": "refund", "amount": 12}, {"type": "reasoning", "amount": 9}],
    ],
    ids=["no-answer", "typed-as-reasoning"],
)
def test_an_array_with_nothing_but_reasoning_is_no_training_row(
    output: list[object], serialized: bool
) -> None:
    exported = json.dumps(output) if serialized else output
    record = langfuse.to_record({**_generation(), "output": exported})
    assert record is not None
    assert (record.response, record.reasoning_only) == ("", True)  # counted, priced, no row
