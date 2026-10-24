"""Structured-output helpers (``parse``) and the ``stream()`` helpers are recorded like ``create``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

from _mock_server import Reply
from _openai_bodies import (
    CHAT_PATH,
    MODEL,
    RESPONSES_PATH,
    chat_body,
    chat_text_events,
    response_body,
)
from _openai_calls import run, unpatched
import openai
import openai.resources.beta as beta_module
from openai.resources.chat.completions import AsyncCompletions, Completions
import openai.resources.responses as responses_module
from pydantic import BaseModel
import pytest

from dagnam_record import _openai

MESSAGES: List[Any] = [{"role": "user", "content": "Give x"}]


class Answer(BaseModel):
    """The structured answer the tests ask for."""

    x: int


def test_chat_parse_is_recorded_once_with_its_schema_and_returns_the_parsed_answer(
    rig: Any,
) -> None:
    rig.server.on(CHAT_PATH, Reply(body=chat_body('{"x": 1}')))
    with rig.client() as client:
        result = client.chat.completions.parse(
            model=MODEL, messages=MESSAGES, response_format=Answer
        )
        beta = client.beta.chat.completions.parse(
            model=MODEL, messages=MESSAGES, response_format=Answer
        )
    assert result.choices[0].message.parsed == Answer(x=1)
    assert beta.choices[0].message.parsed == Answer(x=1)
    first, second = rig.lines()  # one line per call: parse never reaches the wrapped create
    assert (first["endpoint"], first["status"]) == ("chat.completions", 200)
    assert first["request"]["response_format"]["name"] == "Answer"
    assert first["request"]["response_format"]["schema"]["properties"]["x"]["type"] == "integer"
    assert first["norm"]["response"] == '{"x": 1}'
    assert second["norm"]["response"] == '{"x": 1}'
    assert rig.server.seen[0].body["response_format"]["type"] == "json_schema"


def test_async_chat_parse_is_recorded(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(body=chat_body('{"x": 2}')))

    async def call(client: openai.AsyncOpenAI) -> Any:
        return await client.chat.completions.parse(
            model=MODEL, messages=MESSAGES, response_format=Answer
        )

    assert run(call, rig).choices[0].message.parsed == Answer(x=2)
    (line,) = rig.lines()
    assert line["norm"]["response"] == '{"x": 2}'


def test_responses_parse_is_recorded_sync_and_async(rig: Any) -> None:
    rig.server.on(RESPONSES_PATH, Reply(body=response_body('{"x": 3}')))
    with rig.client() as client:
        result = client.responses.parse(model=MODEL, input="Give x", text_format=Answer)
    assert result.output_parsed == Answer(x=3)

    async def call(client: openai.AsyncOpenAI) -> Any:
        return await client.responses.parse(model=MODEL, input="Give x", text_format=Answer)

    assert run(call, rig).output_parsed == Answer(x=3)
    first, second = rig.lines()
    assert (first["endpoint"], first["norm"]["response"]) == ("responses", '{"x": 3}')
    assert first["request"]["text_format"]["name"] == "Answer"
    assert second["endpoint"] == "responses"


def test_a_parse_error_is_raised_unchanged_and_recorded(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(429, {"error": {"message": "slow"}}))
    with rig.client() as client, pytest.raises(openai.RateLimitError):
        client.chat.completions.parse(model=MODEL, messages=MESSAGES, response_format=Answer)
    (line,) = rig.lines()
    assert (line["status"], line["response"]) == (429, {"error": {"type": "RateLimitError"}})


def test_install_patches_parse_and_uninstall_restores_it(rig: Any) -> None:
    for owner in (Completions, AsyncCompletions, responses_module.Responses):
        assert hasattr(owner.parse, "__dagnam_record__")
        assert not hasattr(unpatched(owner.parse), "__dagnam_record__")
    originals = [unpatched(o.parse) for o in (Completions, responses_module.AsyncResponses)]
    _openai.uninstall()
    assert Completions.parse is originals[0]
    assert responses_module.AsyncResponses.parse is originals[1]


def test_an_older_sdk_with_a_separate_beta_class_gets_its_parse_patched(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Legacy:
        def parse(self, **kwargs: Any) -> str:
            return "parsed"

    original = Legacy.parse
    monkeypatch.setattr(
        beta_module,
        "chat",
        SimpleNamespace(completions=SimpleNamespace(Completions=Legacy)),
        raising=False,
    )
    _openai.uninstall()
    assert _openai.install(rig.recorder) is True
    assert hasattr(Legacy.parse, "__dagnam_record__")
    _openai.uninstall()
    assert Legacy.parse is original


def test_the_stream_helpers_are_recorded_through_the_wrapped_create(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(events=chat_text_events()))
    with rig.client() as client:
        with client.chat.completions.stream(model=MODEL, messages=MESSAGES) as stream:
            events = list(stream)
        assert any(event.type == "content.delta" for event in events)
        final = stream.get_final_completion()
    assert final.choices[0].message.content == "Hello"
    (line,) = rig.lines()
    assert line["norm"]["response"] == "Hello"
