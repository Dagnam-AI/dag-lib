"""Which items and arrays of a reply or a prompt are conversation, and which are left alone."""

from __future__ import annotations

import json
from typing import Any

import pytest

from dagnam.audit import Message
from dagnam.audit.readers.base import MalformedRowError
from dagnam.audit.readers.messages import (
    Reply,
    effective_response,
    final_reply,
    has_reply_parts,
    is_reply,
    is_text_parts,
    merge_calls,
    reply_of,
    response_of,
    split_prompt,
)

USER = {"role": "user", "content": "q"}
ASSISTANT = {
    "type": "message",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "a"}],
}


@pytest.mark.parametrize(
    "kind",
    [
        "compaction",
        "compaction_trigger",
        "tool_search_call",
        "tool_search_output",
        "program",
        "program_output",
        "configuration_update",
        "web_search_call",
        "mcp_approval_response",
        "image_generation_call",
        "a_type_added_next_year",
    ],
)
def test_a_role_less_input_item_that_is_not_conversation_is_skipped(kind: str) -> None:
    # Only messages, tool calls and their outputs are conversation. The rule was a list of
    # hosted-tool suffixes, so every new item type made the row malformed.
    item = {"type": kind, "id": "x", "output": "tool-says", "summary": [{"text": "tool-says"}]}

    system, turns = split_prompt({"instructions": "Route.", "input": [USER, item, ASSISTANT, USER]})

    assert system == "Route."
    assert turns == (Message("user", "q"), Message("assistant", "a"), Message("user", "q"))


def test_the_items_that_are_conversation_are_still_read() -> None:
    _, turns = split_prompt(
        [
            USER,
            {"type": "function_call", "call_id": "c1", "name": "route", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "done"},
            {"type": "custom_tool_call", "call_id": "c2", "name": "patch", "input": "x"},
            {"type": "custom_tool_call_output", "call_id": "c2", "output": "applied"},
            USER,
        ]
    )

    assert [turn.role for turn in turns] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "user",
    ]
    assert turns[2].content == "done"


def test_a_typeless_reasoning_block_in_an_input_list_is_skipped() -> None:
    _, turns = split_prompt([USER, {"reasoningContent": {"reasoningText": {"text": "why"}}}, USER])
    assert turns == (Message("user", "q"), Message("user", "q"))


def test_a_json_array_that_is_the_answer_keeps_every_byte() -> None:
    # ``{type, content}`` nodes are an answer, not content parts.
    nodes = [{"type": "paragraph", "content": "Hello"}, {"type": "heading", "content": "Title"}]
    assert not has_reply_parts(nodes)
    text = json.dumps(nodes)
    assert response_of(nodes) == Reply(text, ())
    assert response_of(text) == Reply(text, ())
    # What carries a reply's own parts is still read.
    assert has_reply_parts([{"role": "assistant", "content": "x"}])
    assert has_reply_parts([{"type": "ai", "content": "x"}])
    assert has_reply_parts([{"type": "tool_use", "name": "route", "input": {}}])
    assert not has_reply_parts([{"type": "message", "id": 1}])  # no content: not a message


def test_text_parts_are_the_blocks_a_content_column_holds() -> None:
    assert is_text_parts([{"text": "a"}, {"type": "text", "text": "b"}, "c"])
    assert is_text_parts([{"type": "output_text", "text": "a"}])
    assert not is_text_parts([])
    assert not is_text_parts([{"type": "paragraph", "text": "a"}])
    assert not is_text_parts([{"type": "text"}])  # no text in it
    assert not is_text_parts([1])


def test_a_chat_completions_object_is_a_reply_and_reads_its_first_message() -> None:
    completion = {
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "billing"}},
            {"index": 1, "message": {"role": "assistant", "content": "sales"}},
        ],
    }
    assert is_reply(completion)
    assert reply_of(completion) == ("billing", ())
    assert reply_of(json.dumps(completion)) == ("billing", ())
    # A mapping whose own ``choices`` field is the answer is not a reply.
    for answer in ({"choices": ["a", "b"]}, {"choices": []}, {"choices": [{"message": "x"}]}):
        assert not is_reply(answer)
        assert response_of(answer) == Reply(json.dumps(answer), ())


def test_a_lone_reasoning_item_is_a_reply_and_has_no_answer() -> None:
    item = {"type": "reasoning", "id": "rs", "summary": [{"type": "summary_text", "text": "why"}]}
    assert is_reply(item)
    assert reply_of(item) == ("", ())
    assert final_reply(item) == Reply("", (), reasoning_only=True)
    assert response_of(json.dumps(item)) == Reply("", (), reasoning_only=True)


def test_a_reply_that_is_empty_and_holds_no_reasoning_stays_a_row() -> None:
    for value in ("", {"role": "assistant", "content": ""}, [{"type": "web_search_call"}]):
        assert final_reply(value) == Reply("", ())  # not reasoning, so not a reason to drop it


def test_an_answer_with_a_field_named_reasoning_is_kept_whole() -> None:
    # A structured answer may carry its own ``reasoning`` field: that is the answer.
    answer = {"reasoning": "the lid is cracked", "label": "returns"}
    assert not is_reply(answer)
    assert response_of(answer) == Reply(json.dumps(answer), ())


def test_two_calls_whose_ids_are_both_empty_are_two_calls() -> None:
    first = {"id": "", "name": "route", "args": {"queue": "billing"}}
    second = {"id": "", "name": "route", "args": {"queue": "sales"}}
    assert merge_calls([first], [second]) == (first, second)
    # An empty id names nothing, so the same name and arguments are one call.
    assert merge_calls([first], [{"id": "", "name": "route", "args": {"queue": "billing"}}]) == (
        first,
    )
    assert merge_calls([first], [{"id": None, **second}]) == (first, {"id": None, **second})


def test_arguments_that_are_json_but_not_an_object_compare_by_value() -> None:
    a: dict[str, Any] = {"function": {"name": "pick", "arguments": "[1,2]"}}
    b: dict[str, Any] = {"name": "pick", "args": [1, 2]}
    c: dict[str, Any] = {"name": "pick", "arguments": "[1, 2]"}
    assert merge_calls([a], [b, c]) == (a, c)  # each absorbs one: b is a's twin, c is another
    assert merge_calls([a], [c]) == (a,)
    assert merge_calls([a], [{"name": "pick", "arguments": "[2, 1]"}]) != (a,)
    # What is trained is unchanged: a non-object argument stays what it is.
    assert effective_response("", ({"name": "pick", "arguments": "[1,2]"},)) == (
        '{"arguments": "[1,2]", "name": "pick"}'
    )


def test_a_prompt_with_only_a_skipped_item_has_no_user_turn() -> None:
    with pytest.raises(MalformedRowError, match="no user turn"):
        split_prompt([{"type": "compaction", "id": "c"}])


@pytest.mark.parametrize("answer", ["[URGENT] refund this", "{not json", "  [1, 2]", ' {"a": 1}'])
def test_text_that_only_looks_like_json_stays_the_answer(answer: str) -> None:
    assert response_of(answer) == Reply(answer, ())


def test_json_text_with_leading_whitespace_is_still_a_reply() -> None:
    reply = ' \n{"role": "assistant", "content": [{"type": "thinking", "thinking": "why"}, "ok"]}'
    assert response_of(reply) == Reply("ok", ())


@pytest.mark.parametrize(
    "answer",
    [
        {"candidates": [{"name": "a"}]},
        {"candidates": ["x"]},
        {"candidates": []},
        {"message": {"text": "hi"}},
        {"message": "hi"},
        {"output": {"message": "x"}},
        {"output": {"text": "x"}},
        {"response": "x"},
        {"response": "x", "done": "yes"},
    ],
)
def test_an_object_that_only_looks_like_an_envelope_is_still_the_answer(answer: object) -> None:
    assert response_of(answer) == Reply(json.dumps(answer), ())


def test_each_envelope_is_read_through_the_field_that_holds_its_answer() -> None:
    gemini = {"candidates": [{"content": {"role": "model", "parts": [{"text": "a"}]}}]}
    for envelope in (
        gemini,
        {"message": {"role": "assistant", "content": "a", "thinking": "r"}},
        {"output": {"message": {"role": "assistant", "content": [{"text": "a"}]}}},
        {"response": "a", "thinking": "r", "done": True},
        {"choices": [{"message": {"role": "assistant", "content": "a"}}]},
    ):
        assert is_reply(envelope)
        assert response_of(envelope) == Reply("a", ())
