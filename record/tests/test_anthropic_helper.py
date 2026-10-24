"""The ``messages.stream()`` helper (and its beta twin): recorded once, however it is left."""

from __future__ import annotations

import asyncio
from typing import Any

from _anthropic_bodies import MODEL, PATH, error_body, message_body, text_events, tool_events
from _anthropic_calls import ARGS, run
from _mock_server import Reply
import anthropic
from anthropic.lib.streaming import MessageStream, MessageStreamManager
import pytest

from dagnam_record import _anthropic

EXPECTED = message_body([{"type": "text", "text": "Hello"}])


def test_a_helper_stream_read_by_text_is_recorded_once_with_the_final_message(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=text_events("Hel", "lo")))
    with arig.anthropic() as client, client.messages.stream(**ARGS) as stream:
        assert "".join(stream.text_stream) == "Hello"
        final = stream.get_final_message()
    assert final.model_dump()["content"][0]["text"] == "Hello"
    (line,) = arig.lines()
    assert line["response"] == EXPECTED
    assert line["request"] == ARGS  # the helper adds `stream` itself: not an argument of the call
    assert (line["provider"], line["endpoint"], line["status"]) == ("anthropic", "messages", 200)
    assert arig.writer.counts["dropped_partial"] == 0


def test_get_final_message_alone_records_it(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=tool_events()))
    with arig.anthropic() as client, client.messages.stream(**ARGS) as stream:
        final = stream.get_final_message()
    assert [block.type for block in final.content] == ["text", "tool_use", "tool_use", "tool_use"]
    (line,) = arig.lines()
    assert [b.get("input") for b in line["response"]["content"][1:]] == [
        {"city": "Paris"},
        {"tz": "CET"},
        {},
    ]


def test_leaving_the_with_block_early_writes_nothing_and_counts(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=text_events()))
    with arig.anthropic() as client:
        with client.messages.stream(**ARGS) as stream:
            for _text in stream.text_stream:
                break
        manager = client.messages.stream(**ARGS)
        manager.__enter__().close()  # closing the message stream is the same early exit
    assert arig.raw() == []
    assert arig.writer.counts["dropped_partial"] == 2


def test_leaving_the_with_block_after_message_stop_records_it_once(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=text_events()))
    with arig.anthropic() as client, client.messages.stream(**ARGS) as stream:
        for event in stream:
            if event.type == "message_stop":
                break  # the iteration itself was never asked to end
    assert len(arig.lines()) == 1
    assert arig.writer.counts["dropped_partial"] == 0


def test_the_manager_and_the_message_stream_keep_their_sdk_types(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=text_events()))
    with arig.anthropic() as client:
        manager = client.messages.stream(**ARGS)
        assert isinstance(manager, MessageStreamManager)
        with manager as stream:
            assert isinstance(stream, MessageStream)
            inside: Any = stream
            assert type(inside._raw_stream) is anthropic.Stream  # the SDK's own, watched in place
            assert isinstance(inside._raw_stream, anthropic.Stream)
            assert stream.response.status_code == 200
            stream.get_final_message()
    assert len(arig.lines()) == 1


def test_the_manager_is_the_sdks_own_object_and_a_second_entry_records_again(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=text_events()))
    with arig.anthropic() as client:
        manager = client.messages.stream(**ARGS)
        assert type(manager).__name__ == "MessageStreamManager"
        for _ in range(2):
            with manager as stream:
                stream.get_final_message()
    assert len(arig.lines()) == 2


def test_the_request_is_taken_when_stream_is_called_not_when_it_is_entered(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=text_events()))
    messages: Any = [{"role": "user", "content": "before"}]
    with arig.anthropic() as client:
        manager = client.messages.stream(model=MODEL, max_tokens=8, messages=messages)
        messages[0]["content"] = "after"
        with manager as stream:
            stream.get_final_message()
    (line,) = arig.lines()
    assert line["request"]["messages"] == [{"role": "user", "content": "before"}]


def test_a_refused_helper_request_is_a_failed_call_raised_from_the_with(arig: Any) -> None:
    arig.server.on(PATH, Reply(529, error_body("overloaded_error")))
    with arig.anthropic() as client:
        manager = client.messages.stream(**ARGS)
        assert arig.lines() == []  # nothing has been asked of the server yet
        with pytest.raises(anthropic.APIStatusError) as caught, manager:
            pytest.fail("not entered")
    assert caught.value.status_code == 529
    (line,) = arig.lines()
    assert line["status"] == 529
    assert line["response"] == {"error": {"type": "OverloadedError"}}


def test_an_error_event_in_a_helper_stream_is_a_failed_call(arig: Any) -> None:
    events = [*text_events()[:4], ("error", error_body("api_error"))]
    arig.server.on(PATH, Reply(events=events))
    with arig.anthropic() as client:
        manager = client.messages.stream(**ARGS)
        with pytest.raises(anthropic.APIStatusError), manager as stream:
            stream.get_final_message()
    (line,) = arig.lines()
    assert line["status"] == 500


def test_an_async_helper_stream_is_recorded_once(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=text_events("Hel", "lo")))

    async def call(client: anthropic.AsyncAnthropic) -> Any:
        async with client.messages.stream(**ARGS) as stream:
            text = [piece async for piece in stream.text_stream]
            return text, await stream.get_final_message()

    text, final = run(call, arig)
    assert text == ["Hel", "lo"]
    assert final.model_dump()["stop_reason"] == "end_turn"
    (line,) = arig.lines()
    assert line["response"] == EXPECTED


def test_an_async_helper_left_early_or_cancelled_is_counted_not_written(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=text_events(), cut=True, hold=1.0))

    async def early(client: anthropic.AsyncAnthropic) -> None:
        async with client.messages.stream(**ARGS) as stream:
            async for _text in stream.text_stream:
                break

    run(early, arig)
    assert arig.writer.counts["dropped_partial"] == 1

    arig.server.on(PATH, Reply(body=message_body(), delay=0.5))

    async def cancelled(client: anthropic.AsyncAnthropic) -> None:
        async def enter() -> None:
            async with client.messages.stream(**ARGS):
                pass

        task = asyncio.ensure_future(enter())
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)

    run(cancelled, arig)
    assert arig.raw() == []
    assert arig.writer.counts["dropped_partial"] == 2
    assert arig.writer.counts["dropped_error"] == 0


def test_a_beta_helper_stream_is_recorded_with_its_betas(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=text_events("Hel", "lo")))
    helper = arig.anthropic().beta.messages.stream(betas=["files-api-2025-04-14"], **ARGS)
    with helper as stream:
        assert stream.get_final_text() == "Hello"
    (line,) = arig.lines()
    assert line["response"] == EXPECTED
    assert line["request"]["betas"] == ["files-api-2025-04-14"]


def test_a_beta_create_and_parse_are_recorded_once_each(arig: Any) -> None:
    arig.server.on(PATH, Reply(body=message_body()), Reply(body=message_body()))
    with arig.anthropic() as client:
        client.beta.messages.create(betas=["x-1"], **ARGS)
        parsed = client.beta.messages.parse(**ARGS)
    assert parsed.model_dump()["content"][0]["text"] == "Hello there"
    first, second = arig.lines()
    assert first["request"]["betas"] == ["x-1"]
    assert second["response"]["content"][0]["text"] == "Hello there"


def test_a_tool_runner_records_each_request_it_makes(arig: Any) -> None:
    @anthropic.beta_tool
    def weather(city: str) -> str:
        """The weather.

        Args:
            city: Where.
        """
        return f"sunny in {city}"

    first = message_body(
        [{"type": "tool_use", "id": "toolu_01", "name": "weather", "input": {"city": "Paris"}}],
        "tool_use",
    )
    arig.server.on(PATH, Reply(body=first), Reply(body=message_body()))
    with arig.anthropic() as client:
        runner = client.beta.messages.tool_runner(tools=[weather], **ARGS)
        final = runner.until_done()
    assert final.model_dump()["stop_reason"] == "end_turn"
    lines = arig.lines()
    assert [line["response"]["stop_reason"] for line in lines] == ["tool_use", "end_turn"]
    assert lines[0]["request"]["tools"][0]["name"] == "weather"
    assert "tool_result" in str(lines[1]["request"]["messages"])


def test_parse_never_returns_a_stream_even_when_asked_for_one(arig: Any) -> None:
    arig.server.on(PATH, Reply(body=message_body()))
    with arig.anthropic() as client:
        parsed = client.messages.parse(stream=True, **ARGS)
    assert type(parsed).__name__.startswith("ParsedMessage")
    (line,) = arig.lines()
    assert line["response"]["content"][0]["text"] == "Hello there"
    assert arig.writer.counts["dropped_error"] == 0


def test_parse_with_an_output_format_records_the_schema_and_the_answer(arig: Any) -> None:
    import pydantic

    class Weather(pydantic.BaseModel):
        city: str

    body = message_body([{"type": "text", "text": '{"city": "Paris"}'}])
    arig.server.on(PATH, Reply(body=body))
    with arig.anthropic() as client:
        parsed = client.messages.parse(output_format=Weather, **ARGS)
    assert parsed.parsed_output == Weather(city="Paris")
    (line,) = arig.lines()
    assert line["request"]["output_format"]["name"] == "Weather"
    assert line["request"]["output_format"]["schema"]["properties"] == {
        "city": {"title": "City", "type": "string"}
    }
    assert line["response"]["content"][0]["text"] == '{"city": "Paris"}'
    assert line["norm"]["response"] == '{"city": "Paris"}'


def test_an_async_refused_helper_request_is_a_failed_call(arig: Any) -> None:
    arig.server.on(PATH, Reply(429, error_body("rate_limit_error")))

    async def call(client: anthropic.AsyncAnthropic) -> None:
        with pytest.raises(anthropic.RateLimitError):
            async with client.messages.stream(**ARGS):
                pytest.fail("not entered")

    run(call, arig)
    (line,) = arig.lines()
    assert line["status"] == 429


def test_a_manager_made_while_the_patch_was_off_is_entered_untouched(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=text_events()))
    _anthropic.uninstall()
    with arig.anthropic() as client:
        manager = client.messages.stream(**ARGS)
        assert _anthropic.install(arig.recorder)
        with manager as stream:
            assert stream.get_final_text() == "Hello"

    async def call(client: anthropic.AsyncAnthropic) -> None:
        _anthropic.uninstall()
        manager = client.messages.stream(**ARGS)
        assert _anthropic.install(arig.recorder)
        async with manager as stream:
            assert await stream.get_final_text() == "Hello"

    run(call, arig)
    assert arig.lines() == []
    assert arig.writer.counts["dropped_error"] == 0


def test_an_interrupt_while_the_helper_opens_its_request_is_counted_not_failed(
    arig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt(*args: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt

    with arig.anthropic() as client:
        monkeypatch.setattr(client.messages, "_post", interrupt)
        with pytest.raises(KeyboardInterrupt), client.messages.stream(**ARGS):
            pytest.fail("not entered")
    assert arig.raw() == []
    assert arig.writer.counts["dropped_partial"] == 1
    assert arig.writer.counts["dropped_error"] == 0
