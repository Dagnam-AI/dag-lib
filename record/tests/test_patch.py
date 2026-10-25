"""The shared patch: another library's patch survives ``uninstall``; a huge stream is let go."""

from __future__ import annotations

import copy
import functools
from typing import Any, List, Optional

from _anthropic_bodies import PATH as MESSAGES_PATH, text_events
from _anthropic_calls import ARGS
from _mock_server import Event, Reply
from _openai_bodies import CHAT_PATH, MODEL, chat_body, chat_text_events
from _openai_calls import unpatched
import httpx
from openai.resources.chat.completions import Completions
import pytest

from dagnam_record import _openai
from dagnam_record._openai_assemble import ChatAssembler
from dagnam_record._patch import MAX_STREAM_CHARS

MESSAGES: List[Any] = [{"role": "user", "content": "Hi"}]
METHOD = "create"
BIG = "x" * 600_000


def test_a_patch_another_library_put_on_top_of_ours_survives_uninstall(rig: Any) -> None:
    ours = Completions.__dict__[METHOD]
    original = unpatched(ours)
    calls: List[str] = []

    @functools.wraps(ours)
    def foreign(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append("foreign")
        return ours(self, *args, **kwargs)

    setattr(Completions, METHOD, foreign)
    try:
        _openai.uninstall()
        assert Completions.__dict__[METHOD] is foreign  # not overwritten by the SDK's own
        rig.server.on(CHAT_PATH, Reply(body=chat_body()))
        with rig.client() as client:
            result = client.chat.completions.create(model=MODEL, messages=MESSAGES)
        assert result.choices[0].message.content == "Hello there"
        assert calls == ["foreign"]
        assert rig.raw() == []  # ours is inert now: it passes the call straight through
    finally:
        setattr(Completions, METHOD, original)


def test_uninstall_puts_the_sdks_own_function_back_when_nothing_else_touched_it(rig: Any) -> None:
    ours = Completions.__dict__[METHOD]
    _openai.uninstall()
    assert Completions.__dict__[METHOD] is unpatched(ours)


def set_content(event: Event, text: str) -> None:
    """Make the text of a chat chunk ``text``."""
    _, body = event
    assert isinstance(body, dict)
    body["choices"][0]["delta"]["content"] = text


def big_chat_events(events: Optional[List[Event]] = None) -> List[Event]:
    events = events if events is not None else chat_text_events()
    for position in (1, 2):
        set_content(events[position], BIG)
    return events


def test_a_chat_stream_over_the_line_limit_is_let_go_while_the_caller_reads_all_of_it(
    rig: Any,
) -> None:
    rig.server.on(CHAT_PATH, Reply(events=big_chat_events()))
    with rig.client() as client:
        stream = client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True)
        text = "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
    assert len(text) == 1_200_000
    assert rig.raw() == []
    assert rig.writer.counts["dropped_large"] == 1
    assert rig.writer.counts["dropped_partial"] == 0


def test_a_stream_past_the_limit_stops_being_kept_chunk_by_chunk(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    sizes: List[int] = []
    real_add = ChatAssembler.add

    def spy(self: ChatAssembler, chunk: Any) -> None:
        real_add(self, chunk)
        sizes.append(self.size)

    monkeypatch.setattr(ChatAssembler, "add", spy)
    events = chat_text_events(usage=False)
    for _ in range(30):  # thirty more stop chunks after the two big ones
        events.insert(3, copy.deepcopy(events[3]))
    rig.server.on(CHAT_PATH, Reply(events=big_chat_events(events)))
    with rig.client() as client:
        chunks = list(client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True))
    assert len(chunks) == 34
    assert len(sizes) == 3  # the role chunk and the two big ones; the other 31 were not kept
    assert sizes[-1] > MAX_STREAM_CHARS
    assert rig.writer.counts["dropped_large"] == 1


def test_an_oversized_stream_that_is_then_cut_is_counted_as_large_and_the_error_still_reaches_the_caller(
    rig: Any,
) -> None:
    events = big_chat_events()[:3]  # role, two big chunks, then the connection drops
    rig.server.on(CHAT_PATH, Reply(events=events, cut=True))
    with rig.client() as client:
        iterator = iter(client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True))
        for _ in range(3):
            next(iterator)
        with pytest.raises(httpx.RemoteProtocolError):
            next(iterator)
    assert rig.raw() == []
    assert (rig.writer.counts["dropped_large"], rig.writer.counts["dropped_error"]) == (1, 0)


def test_an_anthropic_stream_over_the_line_limit_is_let_go_while_the_caller_reads_all_of_it(
    arig: Any,
) -> None:
    arig.server.on(MESSAGES_PATH, Reply(events=text_events(BIG, BIG)))
    with arig.anthropic() as client:
        events = list(client.messages.create(stream=True, **ARGS))
    assert (
        sum(len(getattr(e.delta, "text", "")) for e in events if e.type == "content_block_delta")
        == 1_200_000
    )
    assert arig.raw() == []
    assert arig.writer.counts["dropped_large"] == 1


def test_a_stream_just_under_the_limit_is_still_recorded(rig: Any) -> None:
    events = chat_text_events()
    set_content(events[1], "y" * 400_000)
    rig.server.on(CHAT_PATH, Reply(events=events))
    with rig.client() as client:
        list(client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True))
    (line,) = rig.lines()
    assert len(line["response"]["choices"][0]["message"]["content"]) == 400_002
    assert rig.writer.counts["dropped_large"] == 0
