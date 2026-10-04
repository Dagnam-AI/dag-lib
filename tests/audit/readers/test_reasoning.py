"""The reasoning table: what names reasoning, and where inline reasoning is cut."""

from __future__ import annotations

import tracemalloc

import pytest

from dagnam.audit.readers.reasoning import (
    REASONING_MESSAGE_FIELDS,
    REASONING_PLAIN_FIELDS,
    REASONING_TAGS,
    carries_reasoning,
    is_reasoning,
    is_reasoning_type,
    strip_inline_reasoning,
)


def test_what_is_not_reasoning_is_left_alone() -> None:
    for kind in ("text", "output_text", "tool_use", "message", "refund", None, 7):
        assert not is_reasoning_type(kind)
    assert not is_reasoning({"type": "text", "text": "x", "thought": False})
    assert not is_reasoning({"text": "an answer"})
    assert is_reasoning({"text": "x", "thought": True})
    assert is_reasoning({"reasoningContent": {"reasoningText": {"text": "x"}}})
    assert is_reasoning({"encrypted_content": "x"})
    assert not is_reasoning({"text": "an answer", "reasoningContent": None})


def test_a_reasoning_field_of_a_message_counts_as_reasoning_only_when_it_holds_something() -> None:
    for name in REASONING_MESSAGE_FIELDS:
        assert carries_reasoning({"role": "assistant", "content": "", name: "why"})
        assert not carries_reasoning({"role": "assistant", "content": "", name: ""})
        assert not carries_reasoning({"role": "assistant", "content": "", name: None})


def test_strict_reasoning_leaves_out_the_fields_a_structured_answer_may_use() -> None:
    for name in REASONING_PLAIN_FIELDS:
        answer = {name: "because", "label": "x"}
        assert carries_reasoning(answer)
        assert not carries_reasoning(answer, strict=True)
    for name in set(REASONING_MESSAGE_FIELDS) - set(REASONING_PLAIN_FIELDS):
        assert carries_reasoning({name: "why", "label": "x"}, strict=True)


def test_carries_reasoning_looks_through_lists_and_text() -> None:
    assert carries_reasoning([{"a": [{"type": "thinking"}]}])
    assert carries_reasoning("<think>why</think>")
    assert carries_reasoning("why\n</think>\nanswer")
    assert not carries_reasoning("a sentence that mentions </think> in passing")
    assert not carries_reasoning([{"a": ["plain", 1, None]}, {"type": "text"}])
    assert not carries_reasoning("an answer that mentions <think> mid-text")


@pytest.mark.parametrize(
    ("text", "answer"),
    [
        ("<think>a</think>x", "x"),
        ("  <think>a</think>\n\n<think>b</think>  x", "x"),
        ("<think a='1'>a</think >x", "x"),
        ("<think>never closed", ""),
        ("x <think>not leading</think>", "x <think>not leading</think>"),
        ("a reasoning\n</think>\nx", "x"),
        ("two lines\nof reasoning\n  </think>  \n\nx", "x"),
        ("a reasoning\r\n</think>\r\nx", "x"),
        ("<thought>a</thought>x", "x"),
        ("<Thinking>a</Thinking>x", "x"),
        ("﻿<think>a</think>x", "x"),
        ("<thinker>not a reasoning tag</thinker>", "<thinker>not a reasoning tag</thinker>"),
        ("no tag at all", "no tag at all"),
        # The closing tag of the template family, alone on its line, once, with an answer after:
        # anything else is an answer that mentions it, and is kept whole.
        ("a</think>b", "a</think>b"),  # in the middle of a line
        ("a</think>b</think>c", "a</think>b</think>c"),  # twice
        ("why\n</think>\nA\n</think>\nB", "why\n</think>\nA\n</think>\nB"),  # twice, on lines
        ("</think>", "</think>"),  # no reasoning before it, nothing after: just the text
        ("\n</think>\n", "\n</think>\n"),
        ("why\n</think>", ""),  # nothing after it: the call ended inside its reasoning
        ("why\n</think>\n  \n", ""),  # only blanks after it: the same
        ("why\r\n</think>\r\n", ""),
        ("why\n</think>\r\n\r\n  indented", "  indented"),  # the answer keeps its indentation
        ("why</think>\nA", "why</think>\nA"),  # the reasoning's last line ends in the tag
        ("why\n</think> x\nA", "why\n</think> x\nA"),  # the answer starts on the tag's line
        ("why\n</think>\n<think>x", "why\n</think>\n<think>x"),  # an opening tag anywhere
        ("<p>x</p></thought>\nA", "<p>x</p></thought>\nA"),  # another tag: not the template's
        ("close it with </reasoning>\nthen A", "close it with </reasoning>\nthen A"),
        ("why\n</THINK>\nA", "why\n</THINK>\nA"),  # the template emits it in lower case
    ],
)
def test_inline_reasoning_is_cut_before_the_answer_and_nowhere_else(text: str, answer: str) -> None:
    assert strip_inline_reasoning(text) == answer


def test_every_tag_on_the_table_is_cut_when_it_opens_and_only_think_when_it_does_not() -> None:
    for tag in REASONING_TAGS:
        assert strip_inline_reasoning(f"<{tag}>why</{tag}>x") == "x"
        bare = f"why\n</{tag}>\nx"
        assert strip_inline_reasoning(bare) == ("x" if tag == "think" else bare)


def test_a_long_reply_costs_no_more_memory_than_its_length() -> None:
    # A pattern over the lines before the tag kept a backtrack frame for each: 115 MB a million lines.
    reply = "a line of reasoning\n" * 1_000_000 + "</think>\nthe answer"
    tracemalloc.start()
    try:
        assert strip_inline_reasoning(reply) == "the answer"
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 3 * len(reply)
