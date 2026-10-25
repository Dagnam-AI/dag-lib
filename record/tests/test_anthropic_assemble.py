"""The message assembler on odd event sequences the real SDK would not hand it."""

from __future__ import annotations

from typing import Any, Dict, List

from dagnam_record._anthropic_assemble import MessageAssembler


def fold(*events: Dict[str, Any]) -> MessageAssembler:
    assembler = MessageAssembler()
    for event in events:
        assembler.add(event)
    return assembler


START = {
    "type": "message_start",
    "message": {
        "id": "m",
        "role": "assistant",
        "content": [{"x": 1}],
        "usage": {"input_tokens": 3},
    },
}
STOP = {"type": "message_stop"}


def opened(index: Any, block: Any) -> Dict[str, Any]:
    return {"type": "content_block_start", "index": index, "content_block": block}


def delta(index: Any, **fields: Any) -> Dict[str, Any]:
    return {"type": "content_block_delta", "index": index, "delta": fields}


def test_a_stop_without_a_start_is_not_a_message() -> None:
    assert fold(STOP).complete is False
    assert fold(START).complete is False
    assert fold(START, STOP).complete is True


def test_the_content_of_message_start_is_ignored_and_the_blocks_come_from_their_events() -> None:
    assembler = fold(START, STOP)
    assert assembler.result() == {
        "id": "m",
        "role": "assistant",
        "content": [],
        "usage": {"input_tokens": 3},
    }


def test_a_start_that_is_not_a_message_and_events_before_it_are_ignored() -> None:
    assembler = fold(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_start", "message": "nonsense"},
        STOP,
    )
    assert assembler.complete is False  # still no message
    assert assembler.result()["usage"] == {"output_tokens": 2}


def test_malformed_blocks_and_deltas_are_skipped() -> None:
    not_an_index: Any = True
    assembler = fold(
        START,
        opened("zero", {"type": "text", "text": ""}),  # no usable index
        opened(not_an_index, {"type": "text", "text": ""}),  # a bool is not an index
        opened(0, "not a block"),
        opened(1, {"type": "text", "text": "a"}),
        delta(None, type="text_delta", text="x"),
        delta(7, type="text_delta", text="never opened"),
        delta(1, type="text_delta", text=None),
        delta(1, type="signature_delta", signature=3),
        {"type": "message_delta", "delta": "nonsense", "usage": "nonsense"},
        STOP,
    )
    assert assembler.result()["content"] == [{"type": "text", "text": "a"}]


def test_a_block_opened_again_starts_over() -> None:
    assembler = fold(
        START,
        opened(0, {"type": "text", "text": ""}),
        delta(0, type="text_delta", text="old"),
        opened(0, {"type": "text", "text": ""}),
        delta(0, type="text_delta", text="new"),
        STOP,
    )
    assert assembler.result()["content"] == [{"type": "text", "text": "new"}]


def test_citations_are_appended_to_what_the_block_started_with() -> None:
    first = {"type": "char_location", "cited_text": "a"}
    second = {"type": "char_location", "cited_text": "b"}
    assembler = fold(
        START,
        opened(0, {"type": "text", "text": "", "citations": None}),
        delta(0, type="citations_delta", citation=first),
        delta(0, type="citations_delta", citation=second),
        STOP,
    )
    assert assembler.result()["content"][0]["citations"] == [first, second]


def tool_input(pieces: List[str], stop_reason: str = "tool_use", kind: str = "tool_use") -> Any:
    """The assembled ``input`` of one tool block that started with ``{"a": 1}``."""
    events = [opened(0, {"type": kind, "id": "t", "name": "n", "input": {"a": 1}})]
    events += [delta(0, type="input_json_delta", partial_json=piece) for piece in pieces]
    stop = {"type": "message_delta", "delta": {"stop_reason": stop_reason}}
    return fold(START, *events, stop, STOP).result()["content"][0]["input"]


def test_tool_input_pieces_are_parsed_when_they_make_an_object() -> None:
    assert tool_input(['{"a":', "2}"]) == {"a": 2}
    assert tool_input([""]) == {}  # a tool with no parameters
    assert tool_input(["  "]) == {}
    assert tool_input([]) == {"a": 1}  # no pieces at all: what the block started with


def test_tool_input_that_is_not_an_object_or_is_cut_short_stays_the_verbatim_text() -> None:
    assert tool_input(['{"a": ', "1"], "max_tokens") == '{"a": 1'
    assert tool_input(["[1, 2]"]) == "[1, 2]"
    assert tool_input(['{"a'], "tool_use") == '{"a'  # invalid even though the stop looks normal


def test_a_tool_call_with_nothing_in_it_when_max_tokens_stopped_it_has_empty_arguments() -> None:
    assert tool_input([""], "max_tokens") == ""
    assert tool_input([], "max_tokens", kind="server_tool_use") == {"a": 1}  # it started complete
    events = [opened(0, {"type": "tool_use", "id": "t", "name": "n", "input": {}})]
    stop = {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}}
    assert fold(START, *events, stop, STOP).result()["content"][0]["input"] == ""


def test_only_the_last_block_can_have_been_cut_short() -> None:
    first = opened(0, {"type": "tool_use", "id": "a", "name": "n", "input": {}})
    second = opened(1, {"type": "tool_use", "id": "b", "name": "n", "input": {"k": 1}})
    stop = {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}}
    content = fold(
        START, first, delta(0, type="input_json_delta", partial_json=""), second, stop, STOP
    ).result()["content"]
    assert content[0]["input"] == {}  # finished before the last block began
    assert content[1]["input"] == {"k": 1}


def test_an_error_event_makes_a_failed_call_whatever_came_before() -> None:
    assembler = fold(
        START, STOP, {"type": "error", "error": {"type": "overloaded_error", "message": "x"}}
    )
    assert assembler.complete is True
    assert (assembler.status, assembler.result()) == (529, {"error": {"type": "overloaded_error"}})
    anonymous = fold({"type": "error", "error": "plain text"})
    assert (anonymous.complete, anonymous.status, anonymous.result()) == (
        True,
        None,
        {"error": {"type": "error"}},
    )


def test_a_good_stream_has_status_200() -> None:
    assert fold(START, STOP).status == 200


def test_deltas_for_a_block_that_was_never_opened_change_nothing_and_raise_nothing() -> None:
    assembler = fold(
        START,
        delta(3, type="signature_delta", signature="sig"),
        delta(3, type="citations_delta", citation={"type": "char_location"}),
        delta(3, type="thinking_delta", thinking="hm"),
        delta(3, type="input_json_delta", partial_json="{}"),
        STOP,
    )
    assert assembler.result()["content"] == []


def test_a_block_that_starts_with_text_keeps_it_before_the_pieces() -> None:
    assembler = fold(
        START,
        opened(0, {"type": "text", "text": "abc"}),
        delta(0, type="text_delta", text="def"),
        STOP,
    )
    assert assembler.result()["content"] == [{"type": "text", "text": "abcdef"}]
