"""What the recorder reader does that no other source's reader does, and what it leaves alone.

The shared parsers behave as before for every other source. The two readings that are the
recorder's alone (a Responses ``parse()`` schema, the result of a tool the vendor ran) are done in
``recorder.py``, and these tests pin both sides: the other sources keep the old reading, the
recorder gets the new one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dagnam.audit.readers import langfuse, openai_jsonl
from dagnam.audit.readers.messages import reply_of, split_prompt, task_signature
from dagnam.audit.readers.recorder import to_record
from dagnam.audit.record import Message

FIXTURES = Path(__file__).parents[1] / "fixtures"
VENDOR_RESULTS = (
    "web_search_tool_result",
    "code_execution_tool_result",
    "mcp_tool_result",
    "function_tool_result",
)
SCHEMA = {"type": "json_schema", "name": "Ticket"}


def fixture_row(name: str, number: int) -> dict[str, Any]:
    lines = (FIXTURES / name).read_text(encoding="utf-8").splitlines()
    return json.loads(lines[number - 1])


def result(kind: str) -> dict[str, Any]:
    return {"type": kind, "text": "RESULT", "content": "RESULT"}


def test_text_format_names_a_schema_for_the_recorder_alone() -> None:
    assert task_signature({"text_format": SCHEMA}) is None
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]
    answer = {
        "choices": [{"message": {"role": "assistant", "content": "a"}}],
        "id": "c",
        "created": 1_791_763_200,
    }
    batch = openai_jsonl.to_record(
        {"request": {"messages": messages, "text_format": SCHEMA}, "response": answer}
    )
    assert batch is not None
    assert batch.signature is None
    observation = {
        "id": "g", "traceId": "t", "type": "GENERATION", "name": "x", "model": "gpt-4o-mini",
        "startTime": "2026-08-01T09:00:00.000Z", "endTime": "2026-08-01T09:00:00.500Z",
        "input": {"messages": messages, "text_format": SCHEMA},
        "output": {"role": "assistant", "content": "a"},
        "usageDetails": {"input": 1, "output": 1}, "metadata": {},
    }  # fmt: skip
    seen = langfuse.to_record(observation)
    assert seen is not None
    assert seen.signature is None
    row = fixture_row("recorder_openai_parse.jsonl", 2)
    assert row["request"]["text_format"]["name"] == "Ticket"
    record = to_record(row)
    assert record is not None
    assert record.signature == "schema=Ticket"


def test_a_schema_the_request_already_names_wins_over_text_format() -> None:
    row = fixture_row("recorder_openai_parse.jsonl", 1)  # a Chat parse(): response_format.name
    row["request"]["text_format"] = {"name": "Other"}
    record = to_record(row)
    assert record is not None
    assert record.signature == "schema=Ticket"


@pytest.mark.parametrize("bad", [None, "Ticket", ["x"], {"type": "json_schema"}])
def test_a_text_format_without_a_name_is_no_schema(bad: Any) -> None:
    row = fixture_row("recorder_openai_parse.jsonl", 2)
    row["request"]["text_format"] = bad
    record = to_record(row)
    assert record is not None
    assert record.signature is None


@pytest.mark.parametrize("kind", VENDOR_RESULTS)
def test_the_shared_readers_keep_the_text_of_any_such_block(kind: str) -> None:
    assert reply_of({"content": [result(kind), {"type": "text", "text": "A"}]}) == ("RESULTA", ())
    turns = split_prompt(
        [
            {"role": "user", "content": [result(kind)]},
            {"role": "assistant", "content": [result(kind), {"type": "text", "text": "A"}]},
            {"role": "user", "content": "q"},
        ]
    )[1]
    assert turns == (
        Message("user", "RESULT"),
        Message("assistant", "RESULTA"),
        Message("user", "q"),
    )


def anthropic_row(kind: str) -> dict[str, Any]:
    row = fixture_row("recorder_anthropic.jsonl", 8)  # a history with an assistant turn
    row["request"]["messages"] = [
        {"role": "user", "content": [result(kind), {"type": "text", "text": "Q"}]},
        {"role": "assistant", "content": [result(kind), {"type": "text", "text": "H"}]},
        {"role": "user", "content": "go"},
    ]
    row["response"]["content"] = [result(kind), {"type": "text", "text": "A"}]
    return row


@pytest.mark.parametrize("kind", VENDOR_RESULTS)
def test_the_recorder_reader_drops_the_result_in_what_the_assistant_said(kind: str) -> None:
    record = to_record(anthropic_row(kind))
    assert record is not None
    assert record.response == "A"
    # The assistant's own history turn loses it; the user's turns are as for any other source.
    assert record.messages == (
        Message("user", "RESULTQ"),
        Message("assistant", "H"),
        Message("user", "go"),
    )


@pytest.mark.parametrize("kind", ["web_search_result", "search_result", "tool_result", "result"])
def test_only_a_block_typed_as_a_vendor_tool_result_is_dropped(kind: str) -> None:
    row = anthropic_row("web_search_tool_result")
    row["response"]["content"] = [result(kind), {"type": "text", "text": "A"}]
    row["request"]["messages"][1]["content"] = [result(kind), {"type": "text", "text": "H"}]
    record = to_record(row)
    assert record is not None
    assert record.response == "RESULTA"
    assert record.messages[1] == Message("assistant", "RESULTH")


def test_the_openai_shapes_of_the_recorder_reader_drop_it_too() -> None:
    chat = fixture_row("recorder_openai_chat.jsonl", 1)
    chat["response"]["choices"][0]["message"]["content"] = [
        result("function_tool_result"),
        {"type": "text", "text": "A"},
    ]
    from_chat = to_record(chat)
    assert from_chat is not None
    assert from_chat.response == "A"
    chat["request"]["messages"] = [
        {
            "role": "assistant",
            "content": [result("mcp_tool_result"), {"type": "text", "text": "H"}],
        },
        {"role": "user", "content": "go"},
    ]
    with_history = to_record(chat)
    assert with_history is not None
    assert with_history.messages[0] == Message("assistant", "H")
    responses = fixture_row("recorder_openai_responses.jsonl", 1)
    responses["response"]["output"][0]["content"].insert(0, result("function_tool_result"))
    from_responses = to_record(responses)
    assert from_responses is not None
    assert from_responses.response == "high"
    responses["request"]["input"] = [
        {
            "role": "assistant",
            "content": [result("mcp_tool_result"), {"type": "output_text", "text": "H"}],
        },
        {"role": "user", "content": "go"},
    ]
    history = to_record(responses)
    assert history is not None
    assert history.messages[0] == Message("assistant", "H")
