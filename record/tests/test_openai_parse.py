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
from _sdk_features import FEATURES
import openai
import openai.resources.beta as beta_module
import openai.resources.responses as responses_module
from pydantic import BaseModel
import pytest

from dagnam_record import _openai

MESSAGES: List[Any] = [{"role": "user", "content": "Give x"}]

# Where an SDK has the chat helpers: `client.chat.completions` in recent ones, only the beta
# class in older ones (openai 1.66); `client.beta.chat.completions` is there in both.
CHAT_PARSE = [
    pytest.param(
        "chat",
        marks=pytest.mark.skipif(
            not FEATURES["openai chat.completions.parse"], reason="this SDK has no chat parse"
        ),
    ),
    "beta.chat",
]
CHAT_STREAM = [
    pytest.param(
        "chat",
        marks=pytest.mark.skipif(
            not FEATURES["openai chat.completions.stream"], reason="this SDK has no chat stream"
        ),
    ),
    "beta.chat",
]


def completions(client: Any, where: str) -> Any:
    """``client.chat.completions`` or ``client.beta.chat.completions``."""
    owner = client
    for name in (*where.split("."), "completions"):
        owner = getattr(owner, name)
    return owner


class Answer(BaseModel):
    """The structured answer the tests ask for."""

    x: int


@pytest.mark.parametrize("where", CHAT_PARSE)
def test_chat_parse_is_recorded_once_with_its_schema_and_returns_the_parsed_answer(
    rig: Any, where: str
) -> None:
    rig.server.on(CHAT_PATH, Reply(body=chat_body('{"x": 1}')))
    with rig.client() as client:
        result = completions(client, where).parse(
            model=MODEL, messages=MESSAGES, response_format=Answer
        )
    assert result.choices[0].message.parsed == Answer(x=1)
    (line,) = rig.lines()  # one line per call: parse never reaches the wrapped create
    assert (line["endpoint"], line["status"]) == ("chat.completions", 200)
    assert line["request"]["response_format"]["name"] == "Answer"
    assert line["request"]["response_format"]["schema"]["properties"]["x"]["type"] == "integer"
    assert line["norm"]["response"] == '{"x": 1}'
    assert rig.server.seen[0].body["response_format"]["type"] == "json_schema"


@pytest.mark.parametrize("where", CHAT_PARSE)
def test_async_chat_parse_is_recorded(rig: Any, where: str) -> None:
    rig.server.on(CHAT_PATH, Reply(body=chat_body('{"x": 2}')))

    async def call(client: openai.AsyncOpenAI) -> Any:
        return await completions(client, where).parse(
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


@pytest.mark.parametrize("where", CHAT_PARSE)
def test_a_parse_error_is_raised_unchanged_and_recorded(rig: Any, where: str) -> None:
    rig.server.on(CHAT_PATH, Reply(429, {"error": {"message": "slow"}}))
    with rig.client() as client, pytest.raises(openai.RateLimitError):
        completions(client, where).parse(model=MODEL, messages=MESSAGES, response_format=Answer)
    (line,) = rig.lines()
    assert (line["status"], line["response"]) == (429, {"error": {"type": "RateLimitError"}})


def test_install_patches_parse_and_uninstall_restores_it(rig: Any) -> None:
    # The classes the clients really use: one for `chat` and `beta.chat` in recent SDKs, a class
    # of its own for `beta.chat` in older ones (which have no chat `parse` at all).
    sync, aio = openai.OpenAI(api_key="unused"), openai.AsyncOpenAI(api_key="unused")
    chat = {type(sync.chat.completions), type(sync.beta.chat.completions)}
    chat |= {type(aio.chat.completions), type(aio.beta.chat.completions)}
    owners = [*chat, responses_module.Responses, responses_module.AsyncResponses]
    owners = [owner for owner in owners if hasattr(owner, "parse")]
    assert len(owners) >= 3  # the floor: beta chat (sync, async) and Responses
    for owner in owners:
        assert hasattr(owner.parse, "__dagnam_record__")
        assert not hasattr(unpatched(owner.parse), "__dagnam_record__")
    originals = [unpatched(owner.parse) for owner in owners]
    _openai.uninstall()
    assert [owner.parse for owner in owners] == originals


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


@pytest.mark.parametrize("where", CHAT_STREAM)
def test_the_stream_helpers_are_recorded_through_the_wrapped_create(rig: Any, where: str) -> None:
    rig.server.on(CHAT_PATH, Reply(events=chat_text_events()))
    with rig.client() as client:
        with completions(client, where).stream(model=MODEL, messages=MESSAGES) as stream:
            events = list(stream)
        assert any(event.type == "content.delta" for event in events)
        final = stream.get_final_completion()
    assert final.choices[0].message.content == "Hello"
    (line,) = rig.lines()
    assert line["norm"]["response"] == "Hello"
