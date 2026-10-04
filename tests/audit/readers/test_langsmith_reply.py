"""How a LangSmith run's outputs are read as a reply: Responses items, tool calls, thought parts."""

from __future__ import annotations

import json

import pytest

from dagnam.audit import Message
from dagnam.audit.readers import langsmith
from dagnam.audit.readers.messages import effective_response

REPLAYED_INPUT = [
    {"role": "user", "content": "ticket 1"},
    {"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAA-opaque"},
    {"type": "web_search_call", "id": "ws_1", "status": "completed"},
    {"type": "mcp_call", "id": "mcp_1", "name": "kb", "arguments": "{}", "output": "tool-says"},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "a1"}]},
    {"role": "user", "content": "ticket 2"},
]
"""A Responses ``input`` that replays an earlier turn which ran hosted tools."""


def _llm_run() -> dict[str, object]:
    return {
        "id": "run-1",
        "trace_id": "trace-1",
        "session_id": "project-uuid",
        "run_type": "llm",
        "start_time": "2026-08-01T09:00:00.000000",
        "end_time": "2026-08-01T09:00:00.400000",
        "inputs": {
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        },
        "outputs": {"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
        "extra": {"invocation_params": {"model": "gpt-4o-mini"}, "metadata": {}},
        "prompt_tokens": 5,
        "completion_tokens": 1,
        "total_cost": 0.001,
    }


def test_a_responses_api_run_reads_its_instructions_and_output_items() -> None:
    # wrap_openai on the Responses API: the system prompt is ``instructions`` and the
    # reply an ``output`` item list, so every run was unstructured free text.
    row = {
        **_llm_run(),
        "inputs": {
            "instructions": "Classify the ticket intent. Reply with one word.",
            "input": [{"role": "user", "content": "ticket 7"}],
            "model": "gpt-5-mini",
        },
        "outputs": {
            "id": "resp_7",
            "object": "response",
            "output": [
                {"type": "reasoning", "id": "rs_7", "summary": []},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "billing", "annotations": []}],
                },
            ],
        },
    }
    record = langsmith.to_record(row)
    assert record is not None
    assert record.system == "Classify the ticket intent. Reply with one word."
    assert record.messages == (Message("user", "ticket 7"),)
    assert record.response == "billing"


def test_gemini_thought_parts_are_not_the_answer() -> None:
    row = {
        **_llm_run(),
        "outputs": {
            "content": [
                {"text": "The user seems upset, so this is urgent.", "thought": True},
                {"text": "low"},
            ]
        },
    }
    record = langsmith.to_record(row)
    assert record is not None
    assert record.response == "low"


def test_tool_results_and_the_calls_they_answer_stay_in_the_prompt() -> None:
    # Anthropic tool_result blocks were blanked, and the assistant turn that made
    # the call was dropped: the label depended on text the student never saw.
    row = {
        **_llm_run(),
        "inputs": {
            "messages": [
                {"role": "system", "content": "Summarize the tool output as a label."},
                {"role": "user", "content": "order 3"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "x", "name": "lookup", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "x", "content": "status=shipped 3"}
                    ],
                },
            ]
        },
        "outputs": {"role": "assistant", "content": [{"type": "text", "text": "shipped"}]},
    }
    record = langsmith.to_record(row)
    assert record is not None
    assert record.messages == (
        Message("user", "order 3"),
        Message("assistant", '{"arguments": {}, "name": "lookup"}'),
        Message("user", "status=shipped 3"),
    )
    assert record.response == "shipped"


def test_an_anthropic_tool_use_reply_is_a_tool_call() -> None:
    # Forced-tool JSON through a raw Anthropic Message: 0 of 300 rows derived.
    block = {"type": "tool_use", "id": "t", "name": "record_contact", "input": {"city": "Berlin"}}
    row = {**_llm_run(), "outputs": {"role": "assistant", "content": [block]}}
    record = langsmith.to_record(row)
    assert record is not None
    assert record.response == ""
    assert record.response_tool_calls == (block,)


def test_a_langchain_anthropic_reply_carries_its_call_once() -> None:
    # ``ChatAnthropic`` keeps the ``tool_use`` block in ``content`` and lists the same
    # call in ``tool_calls``: read from both, one call became a two-call list.
    ai_message = {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain", "schema", "messages", "AIMessage"],
        "kwargs": {
            "content": [
                {"type": "text", "text": "Routing it."},
                {"type": "tool_use", "id": "toolu_01", "name": "route", "input": {"team": "a"}},
            ],
            "tool_calls": [
                {"name": "route", "args": {"team": "a"}, "id": "toolu_01", "type": "tool_call"}
            ],
        },
    }
    row = {**_llm_run(), "outputs": {"generations": [[{"text": "", "message": ai_message}]]}}
    record = langsmith.to_record(row)
    assert record is not None
    assert len(record.response_tool_calls) == 1
    assert effective_response(record.response, record.response_tool_calls) == (
        '{"arguments": {"team": "a"}, "name": "route"}'
    )
    # As the earlier turn of the next call, it is one call too.
    human = {"lc": 1, "id": ["langchain", "schema", "HumanMessage"], "kwargs": {"content": "hi"}}
    later = langsmith.to_record({**_llm_run(), "inputs": {"messages": [[human, ai_message]]}})
    assert later is not None
    assert later.messages[1] == Message(
        "assistant", '{"arguments": {"team": "a"}, "name": "route"}'
    )


def test_a_responses_output_with_hosted_tool_items_reads_its_message() -> None:
    output = [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "secret"}]},
        {"type": "web_search_call", "id": "ws_1", "status": "completed"},
        {"type": "message", "content": [{"type": "output_text", "text": "billing"}]},
    ]
    record = langsmith.to_record({**_llm_run(), "outputs": {"output": output}})
    assert record is not None
    assert (record.response, record.response_tool_calls) == ("billing", ())


def test_a_replayed_hosted_tool_item_and_a_custom_tool_call_are_read() -> None:
    custom = {"type": "custom_tool_call", "call_id": "c1", "name": "apply_patch", "input": "x"}
    row = {
        **_llm_run(),
        "inputs": {"instructions": "Route.", "input": REPLAYED_INPUT},
        "outputs": {"output": [{"type": "reasoning", "summary": []}, custom]},
    }
    record = langsmith.to_record(row)
    assert record is not None
    assert [(m.role, m.content) for m in record.messages] == [
        ("user", "ticket 1"),
        ("assistant", "a1"),
        ("user", "ticket 2"),
    ]
    assert (record.response, record.response_tool_calls) == ("", (custom,))


def test_a_legacy_function_call_message_is_a_tool_call() -> None:
    legacy = {"name": "route", "arguments": '{"team": "a"}'}
    message = {"role": "assistant", "content": None, "function_call": legacy}
    record = langsmith.to_record({**_llm_run(), "outputs": {"choices": [{"message": message}]}})
    assert record is not None
    assert (record.response, record.response_tool_calls) == ("", (legacy,))


def test_a_text_outcome_in_the_metadata_is_ignored() -> None:
    record = langsmith.to_record({**_llm_run(), "extra": {"metadata": {"outcome": "thumbs up"}}})
    assert record is not None
    assert record.outcome is None


@pytest.mark.parametrize("serialized", [False, True])
def test_a_json_array_answer_in_outputs_output_keeps_every_byte(serialized: bool) -> None:
    nodes = [{"type": "paragraph", "content": "Hello"}, {"type": "heading", "content": "Title"}]
    output = json.dumps(nodes) if serialized else nodes

    record = langsmith.to_record({**_llm_run(), "outputs": {"output": output}})

    assert record is not None
    assert record.response == json.dumps(nodes)
    assert not record.reasoning_only


def test_a_whole_response_object_in_outputs_is_read_through_its_message() -> None:
    whole = {"model": "m", "message": {"role": "assistant", "content": "billing", "thinking": "hm"}}

    record = langsmith.to_record({**_llm_run(), "outputs": whole})

    assert record is not None
    assert record.response == "billing"
