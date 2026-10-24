"""Chat content in every provider's shape: text, turns, tool calls, reasoning, media, signature."""

from __future__ import annotations

import json

import pytest

from dagnam.audit import Message
from dagnam.audit.readers.base import MalformedRowError
from dagnam.audit.readers.messages import (
    content_text,
    effective_response,
    has_media,
    has_reply_parts,
    is_reply,
    merge_calls,
    reply_of,
    reply_text,
    split_prompt,
    task_signature,
    tool_calls,
)


def test_content_text_flattens_parts_and_keeps_tool_results() -> None:
    assert content_text([{"type": "text", "text": "a"}, {"type": "image_url"}, "b"]) == "ab"
    assert content_text("plain") == "plain"
    assert content_text(None) == ""
    # Tool results arrive as content parts in four spellings; they were blanked.
    assert content_text([{"type": "tool_result", "tool_use_id": "x", "content": "shipped"}]) == (
        "shipped"
    )
    nested = [{"type": "tool_result", "content": [{"type": "text", "text": "nested"}]}]
    assert content_text(nested) == "nested"
    assert content_text([{"type": "tool-result", "result": {"ok": True}}]) == '{"ok": true}'
    assert content_text([{"type": "tool_call_response", "response": "42"}]) == "42"
    gemini = [{"functionResponse": {"name": "lookup", "response": {"status": "late"}}}]
    assert content_text(gemini) == '{"status": "late"}'
    # OTel GenAI text parts carry their text under ``content``.
    assert content_text([{"type": "text", "content": "semconv"}]) == "semconv"
    assert content_text([{"inlineData": {"mimeType": "image/png"}}]) == ""


def test_reasoning_is_never_part_of_a_reply() -> None:
    # Gemini thought parts, Anthropic thinking blocks, Responses reasoning items,
    # and an inline <think> all read as the answer before.
    parts = [
        {"text": "The user seems upset.", "thought": True},
        {"type": "thinking", "thinking": "hmm"},
        {"type": "redacted_thinking", "data": "x"},
        {"type": "reasoning", "summary": [{"text": "why"}]},
        {"text": "low"},
    ]
    assert reply_text(parts) == "low"
    assert reply_text("<think>long\nthought</think>\n\npos") == "pos"
    assert reply_text("  <think>x</think>neg") == "neg"
    assert reply_text("pos <think>not leading</think>") == "pos <think>not leading</think>"
    # Cut off before the reasoning closed (the token limit): there is no answer at all.
    assert reply_text("<think>The customer sounds upset, so the label is prob") == ""
    assert reply_text("<think>one</think>neg<think>two") == "neg<think>two"


def test_reply_of_reads_every_reply_shape() -> None:
    call = {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    assert reply_of("text") == ("text", ())
    assert reply_of({"role": "assistant", "content": None, "tool_calls": [call]}) == ("", (call,))
    tool_use = {"type": "tool_use", "id": "t", "name": "record", "input": {"a": 1}}
    assert reply_of({"content": [{"type": "text", "text": "ok"}, tool_use]}) == ("ok", (tool_use,))
    gemini = {"functionCall": {"name": "route", "args": {"team": "a"}}}
    assert reply_of({"content": [gemini]}) == ("", ({"name": "route", "args": {"team": "a"}},))
    item = {"type": "function_call", "call_id": "c", "name": "route", "arguments": "{}"}
    message = {"type": "message", "content": [{"type": "output_text", "text": "billing"}]}
    assert reply_of(item) == ("", (item,))
    assert reply_of({"output": [{"type": "reasoning"}, message]}) == ("billing", ())
    assert reply_of([message, item]) == ("billing", (item,))
    assert reply_of({"role": "assistant", "parts": [{"type": "text", "content": "hi"}]}) == (
        "hi",
        (),
    )
    with pytest.raises(MalformedRowError):
        reply_of({"content": "x", "tool_calls": "not a list"})


def test_tool_calls_accepts_only_a_list_of_objects() -> None:
    assert tool_calls(None) == ()
    assert tool_calls([{"id": 1}]) == ({"id": 1},)
    with pytest.raises(MalformedRowError):
        tool_calls([{"id": 1}, "x"])


@pytest.mark.parametrize(
    "call",
    [
        {"function": {"name": "route", "arguments": '{"queue": "billing"}'}},
        {"name": "route", "args": {"queue": "billing"}},
        {"name": "route", "arguments": {"queue": "billing"}},
        {"type": "tool_use", "name": "route", "input": {"queue": "billing"}},
    ],
)
def test_a_tool_call_is_judged_as_its_name_and_arguments(call: dict[str, object]) -> None:
    assert effective_response("preamble", (call,)) == (
        '{"arguments": {"queue": "billing"}, "name": "route"}'
    )


def test_several_tool_calls_are_judged_as_the_ordered_list() -> None:
    # A reply that sets a product AND a team was judged on the product alone.
    calls = (
        {"function": {"name": "set_product", "arguments": '{"product": "A"}'}},
        {"function": {"name": "set_team", "arguments": '{"team": "x"}'}},
    )
    assert json.loads(effective_response("", calls)) == [
        {"arguments": {"product": "A"}, "name": "set_product"},
        {"arguments": {"team": "x"}, "name": "set_team"},
    ]
    assert effective_response("no calls", ()) == "no calls"
    assert effective_response("", ({"arguments": "not json"},)) == (
        '{"arguments": "not json", "name": null}'
    )
    assert effective_response("", ({"arguments": "[1]"},)) == '{"arguments": "[1]", "name": null}'


TOOL_CALL = {"name": "route", "args": {"queue": "billing"}, "id": "toolu_1", "type": "tool_call"}
TOOL_USE = {"type": "tool_use", "id": "toolu_1", "name": "route", "input": {"queue": "billing"}}
ROUTE = '{"arguments": {"queue": "billing"}, "name": "route"}'


def test_a_call_carried_in_tool_calls_and_in_a_content_block_is_one_call() -> None:
    # A LangChain ``AIMessage`` from an Anthropic model holds each call twice: in
    # ``tool_calls`` and as the ``tool_use`` block. Concatenated, one call was trained
    # and scored as a two-call list.
    message = {"role": "assistant", "content": [TOOL_USE], "tool_calls": [TOOL_CALL]}
    reply, calls = reply_of(message)
    assert calls == (TOOL_CALL,)
    assert effective_response(reply, calls) == ROUTE
    # The same reply as a context turn of a later call.
    _, turns = split_prompt([{"role": "user", "content": "q"}, message])
    assert turns[1] == Message("assistant", ROUTE)


def test_merge_calls_drops_a_repeat_and_keeps_every_different_call() -> None:
    other = {"type": "tool_use", "id": "toolu_2", "name": "route", "input": {"queue": "sales"}}
    assert merge_calls([TOOL_CALL], [TOOL_USE, other]) == (TOOL_CALL, other)
    assert merge_calls([], [TOOL_USE, other]) == (TOOL_USE, other)
    assert merge_calls([TOOL_CALL], []) == (TOOL_CALL,)
    # Responses: the item's ``call_id`` is what the chat-style call names ``id``.
    item = {"type": "function_call", "id": "fc_9", "call_id": "toolu_1", "name": "route"}
    assert merge_calls([TOOL_CALL], [item]) == (TOOL_CALL,)
    # Without an id on either side, the same name and arguments are the same call...
    bare = {"function": {"name": "route", "arguments": '{"queue": "billing"}'}}
    gemini = {"name": "route", "args": {"queue": "billing"}}
    assert merge_calls([bare], [gemini]) == (bare,)
    assert merge_calls([TOOL_CALL], [gemini]) == (TOOL_CALL,)
    # ...and other arguments, another name or another id are another call.
    assert merge_calls([bare], [{"name": "route", "args": {"queue": "sales"}}]) == (
        bare,
        {"name": "route", "args": {"queue": "sales"}},
    )
    assert merge_calls([bare], [{"name": "escalate", "args": {"queue": "billing"}}]) == (
        bare,
        {"name": "escalate", "args": {"queue": "billing"}},
    )
    again = {**TOOL_USE, "id": "toolu_2"}  # the same call made twice, under two ids
    assert merge_calls([TOOL_CALL], [TOOL_USE, again]) == (TOOL_CALL, again)
    # Each call absorbs one block: a call the teacher really made twice stays two.
    assert merge_calls([bare, bare], [gemini, gemini, gemini]) == (bare, bare, gemini)


def test_has_reply_parts_tells_reply_items_from_an_answer_that_is_a_json_array() -> None:
    message = {"type": "message", "content": [{"type": "output_text", "text": "low"}]}
    assert has_reply_parts([{"type": "web_search_call"}, message])
    assert has_reply_parts([{"type": "function_call", "name": "route", "arguments": "{}"}])
    assert has_reply_parts([{"type": "reasoning", "summary": []}, {"type": "mcp_call"}])
    assert has_reply_parts([{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": "x"}])
    assert has_reply_parts([{"text": "hm", "thought": True}, {"text": "x"}])
    assert has_reply_parts(["loose", {"type": "redacted_thinking", "data": "x"}])
    assert not has_reply_parts([])
    assert not has_reply_parts(["a", "b"])
    assert not has_reply_parts([{"type": "refund", "amount": 12}, {"type": "message", "id": 1}])
    assert not has_reply_parts([{"text": "x", "thought": False}])


def test_split_prompt_separates_system_and_keeps_assistant_context() -> None:
    system, turns = split_prompt(
        [
            {"role": "system", "content": "one"},
            {"role": "developer", "content": "two"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "prior answer"},
            {"role": "tool", "content": "42"},
        ]
    )
    assert system == "one\ntwo"
    # An earlier assistant turn is context now, so a later tool result has its call.
    assert [(m.role, m.content) for m in turns] == [
        ("user", "q"),
        ("assistant", "prior answer"),
        ("tool", "42"),
    ]
    assert split_prompt("bare prompt") == (None, (Message("user", "bare prompt"),))
    assert split_prompt({"messages": [{"role": "user", "content": "x"}]})[1][0].content == "x"
    assert split_prompt({"prompt": "x"}) == (None, (Message("user", '{"prompt": "x"}'),))
    with pytest.raises(MalformedRowError):
        split_prompt([{"role": "user"}, "loose string"])
    with pytest.raises(MalformedRowError, match="no user turn"):
        split_prompt([])
    with pytest.raises(MalformedRowError, match="no user turn"):
        split_prompt([{"role": "assistant", "content": "only me"}])
    with pytest.raises(MalformedRowError, match="not a chat message"):
        split_prompt([{"content": "no role"}])


def test_split_prompt_reads_an_anthropic_system_beside_the_messages() -> None:
    system, turns = split_prompt(
        {
            "system": "Classify the ticket intent.",
            "messages": [{"role": "user", "content": "where is my order"}],
        }
    )
    assert system == "Classify the ticket intent."
    assert turns == (Message("user", "where is my order"),)
    blocks, _ = split_prompt(
        {
            "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "messages": [{"role": "system", "content": "c"}, {"role": "user", "content": "q"}],
        }
    )
    assert blocks == "ab\nc"


def test_a_gemini_model_turn_is_an_assistant_turn() -> None:
    convo = [
        {"role": "user", "content": "q1"},
        {"role": "model", "parts": [{"text": "a1"}]},
        {"role": "user", "content": "q2"},
    ]
    as_openai = [
        {"role": "assistant", "content": "a1"} if m["role"] == "model" else m for m in convo
    ]
    expected = (
        None,
        (Message("user", "q1"), Message("assistant", "a1"), Message("user", "q2")),
    )
    assert split_prompt(convo) == split_prompt(as_openai) == expected


def test_responses_api_input_items_become_turns() -> None:
    # ``instructions`` is the system prompt; a ``function_call`` item is the
    # assistant's call and its ``function_call_output`` the tool's answer.
    system, turns = split_prompt(
        {
            "instructions": "Route the ticket.",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "ticket 1"}]},
                {"type": "reasoning", "id": "rs"},
                {"type": "function_call", "call_id": "c", "name": "lookup", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c", "output": "late"},
            ],
        }
    )
    assert system == "Route the ticket."
    assert turns == (
        Message("user", "ticket 1"),
        Message("assistant", '{"arguments": {}, "name": "lookup"}'),
        Message("tool", "late"),
    )
    assert split_prompt({"instructions": "Classify.", "input": "ticket 2"}) == (
        "Classify.",
        (Message("user", "ticket 2"),),
    )


HOSTED_ITEMS = [
    {"type": "web_search_call", "id": "ws_1", "status": "completed"},
    {"type": "mcp_list_tools", "id": "mcpl_1", "server_label": "docs", "tools": []},
    {"type": "mcp_call", "id": "mcp_1", "name": "search", "arguments": "{}", "output": "tool-says"},
    {"type": "file_search_call", "id": "fs_1", "queries": ["refund"], "status": "completed"},
    {"type": "computer_call_output", "call_id": "cc_1", "output": {"type": "input_image"}},
    {"type": "item_reference", "id": "msg_1"},
]
"""What a Responses ``input`` list replays from an earlier turn that ran a hosted tool."""


def test_a_replayed_hosted_tool_item_is_skipped_and_never_breaks_the_row() -> None:
    # One such item made the whole row "not a chat message"; above 5% of rows the scan aborted.
    assistant = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "a1"}],
    }
    system, turns = split_prompt(
        {
            "instructions": "Route.",
            "input": [
                {"role": "user", "content": "q1"},
                {"type": "reasoning", "id": "rs", "encrypted_content": "gAAAA-opaque"},
                *HOSTED_ITEMS,
                assistant,
                {"role": "user", "content": "q2"},
            ],
        }
    )
    assert system == "Route."
    assert turns == (Message("user", "q1"), Message("assistant", "a1"), Message("user", "q2"))
    assert not any("tool-says" in turn.content or "gAAAA" in turn.content for turn in turns)


def test_a_role_less_item_with_no_type_is_still_not_a_chat_message() -> None:
    for item in ({"amount": 12}, "loose", {"type": "message", "content": "no role"}):
        with pytest.raises(MalformedRowError, match="not a chat message"):
            split_prompt([{"role": "user", "content": "q"}, item])
    with pytest.raises(MalformedRowError, match="no user turn"):
        split_prompt([HOSTED_ITEMS[0]])  # skipping it leaves nothing the student could answer


CUSTOM_CALL = {
    "type": "custom_tool_call",
    "call_id": "c1",
    "name": "apply_patch",
    "input": "*** Begin",
}
CUSTOM_JSON = '{"arguments": "*** Begin", "name": "apply_patch"}'


def test_a_custom_tool_call_is_a_tool_call() -> None:
    assert reply_of(CUSTOM_CALL) == ("", (CUSTOM_CALL,))
    message = {"type": "message", "content": [{"type": "output_text", "text": "patching"}]}
    assert reply_of({"output": [{"type": "reasoning"}, message, CUSTOM_CALL]}) == (
        "patching",
        (CUSTOM_CALL,),
    )
    assert effective_response("", (CUSTOM_CALL,)) == CUSTOM_JSON
    assert is_reply(CUSTOM_CALL)
    assert has_reply_parts([{"type": "web_search_call"}, CUSTOM_CALL])
    # Replayed in a later request it is the assistant's call, and its output the tool's answer.
    _, turns = split_prompt(
        [
            {"role": "user", "content": "fix it"},
            CUSTOM_CALL,
            {"type": "custom_tool_call_output", "call_id": "c1", "output": "applied"},
        ]
    )
    assert turns == (
        Message("user", "fix it"),
        Message("assistant", CUSTOM_JSON),
        Message("tool", "applied"),
    )
    # The Chat Completions spelling nests the same two fields under ``custom``.
    chat = {"id": "c1", "type": "custom", "custom": {"name": "apply_patch", "input": "*** Begin"}}
    assert reply_of({"role": "assistant", "content": None, "tool_calls": [chat]}) == ("", (chat,))
    assert effective_response("", (chat,)) == CUSTOM_JSON


def test_a_legacy_top_level_function_call_is_a_tool_call() -> None:
    legacy = {"name": "route", "arguments": '{"team": "a"}'}
    message = {"role": "assistant", "content": None, "function_call": legacy}
    judged = '{"arguments": {"team": "a"}, "name": "route"}'
    assert is_reply(message)
    assert is_reply({"role": "assistant", "function_call": legacy})  # no ``content`` key at all
    text_and_call = reply_of({**message, "content": "routing"})
    assert text_and_call == ("routing", (legacy,))
    reply, calls = reply_of(message)
    assert (reply, effective_response(reply, calls)) == ("", judged)
    # The same call carried in ``tool_calls`` as well is one call, not two.
    modern = {"id": "c", "type": "function", "function": legacy}
    assert reply_of({**message, "tool_calls": [modern]})[1] == (modern,)
    # As an earlier turn of the next request it is the assistant's call.
    _, turns = split_prompt(
        [
            {"role": "user", "content": "q"},
            message,
            {"role": "function", "name": "route", "content": "done"},
        ]
    )
    assert turns == (
        Message("user", "q"),
        Message("assistant", judged),
        Message("function", "done"),
    )


def test_has_media_finds_an_image_audio_or_file_part_anywhere() -> None:
    image = {"type": "image_url", "image_url": {"url": "https://img/1.png"}}
    assert has_media([{"role": "user", "content": [{"type": "text", "text": "x"}, image]}])
    assert has_media({"messages": [{"role": "user", "content": [{"type": "input_audio"}]}]})
    assert has_media([{"role": "user", "parts": [{"inlineData": {"mimeType": "image/png"}}]}])
    assert has_media([{"role": "user", "content": [{"type": "document", "source": {}}]}])
    assert not has_media([{"role": "user", "content": "text only"}])
    assert not has_media("a string")
    assert not has_media(
        {"input": [{"role": "user", "content": "x"}], "tools": [{"type": "function"}]}
    )


def test_task_signature_names_the_schema_and_never_the_tool_set() -> None:
    # Two tasks under one system prompt differ only in the schema they ask for.
    invoice = {"response_format": {"type": "json_schema", "json_schema": {"name": "invoice"}}}
    assert task_signature(invoice) == "schema=invoice"
    assert task_signature({"text": {"format": {"type": "json_schema", "name": "resume"}}}) == (
        "schema=resume"
    )
    # The tools a request OFFERS change per call (a deploy adds one, permissions
    # or retrieval pick them), so they never split a workload.
    tools = [{"type": "function", "function": {"name": "b"}}, {"type": "function", "name": "a"}]
    assert task_signature({"tools": tools, **invoice}) == "schema=invoice"
    assert task_signature({"tools": tools}) is None
    # A FORCED tool is the schema, in OpenAI's and Anthropic's spellings; a
    # choice that names no tool ("auto", "required", Anthropic's "any") is not.
    forced = {"type": "function", "function": {"name": "extract_invoice"}}
    assert task_signature({"tools": tools, "tool_choice": forced}) == "schema=extract_invoice"
    anthropic = {"type": "tool", "name": "extract_invoice"}
    assert task_signature({"tools": tools, "tool_choice": anthropic}) == "schema=extract_invoice"
    for choice in ("auto", "required", {"type": "any"}):
        assert task_signature({"tools": tools, "tool_choice": choice}) is None
    assert task_signature({"function_call": {"name": "extract"}}) == "schema=extract"  # legacy
    assert task_signature({"function_call": "auto"}) is None
    assert task_signature({"messages": []}) is None
    assert task_signature("not a request") is None


def test_is_reply_tells_a_reply_from_an_answer_that_is_json() -> None:
    assert is_reply({"role": "assistant", "content": "low"})
    assert is_reply({"type": "message", "content": [{"type": "output_text", "text": "low"}]})
    assert is_reply({"type": "function_call", "name": "route", "arguments": "{}"})
    assert is_reply({"object": "response", "output": [{"type": "message", "content": []}]})
    assert is_reply({"type": "ai", "content": "billing", "id": "run-1"})  # a LangChain AIMessage
    assert is_reply({"role": "assistant", "parts": [{"type": "text", "content": "low"}]})
    # An answer's own ``type``, ``role`` or ``output`` field is not a reply.
    assert not is_reply({"type": "refund", "amount": 12})
    assert not is_reply({"type": "message", "priority": "high"})
    assert not is_reply({"name": "Bob", "role": "admin"})
    assert not is_reply({"output": ["a", "b"]})
    assert not is_reply({"output": []})
    assert not is_reply({"content": "a JSON answer with a content field"})
    assert not is_reply({"amount": 12})
