"""Calls through the real SDKs against the mock server, for the guarantee tests.

``run_call(kind, server)`` makes one call and returns what the caller sees: the result's data or
the exception's class, status and text. The guarantee tests compare that with and without a
failing recorder.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List

from _anthropic_bodies import PATH as MESSAGES_PATH, error_body, message_body, text_events
from _mock_server import MockServer, Reply
from _openai_bodies import CHAT_PATH, MODEL, chat_body, chat_text_events
import anthropic
import httpx
import openai

import dagnam_record

CHAT_ARGS: Dict[str, Any] = {"model": MODEL, "messages": [{"role": "user", "content": "Hi"}]}
MESSAGE_ARGS: Dict[str, Any] = {
    "model": "claude-sonnet-4-5",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "Hi"}],
}


def openai_client(server: MockServer) -> openai.OpenAI:
    """A real client aimed at the mock server (no retries, no proxy from the environment)."""
    return openai.OpenAI(
        api_key="k",
        base_url=server.url,
        max_retries=0,
        http_client=httpx.Client(trust_env=False),
    )


def anthropic_client(server: MockServer) -> anthropic.Anthropic:
    """The Anthropic twin of ``openai_client``."""
    return anthropic.Anthropic(
        api_key="k",
        base_url=server.root,
        max_retries=0,
        http_client=httpx.Client(trust_env=False),
    )


def _openai_ok(server: MockServer) -> Any:
    server.on(CHAT_PATH, Reply(body=chat_body()))
    with openai_client(server) as client:
        return client.chat.completions.create(**CHAT_ARGS).model_dump()


def _openai_error(server: MockServer) -> Any:
    server.on(CHAT_PATH, Reply(429, {"error": {"message": "slow down"}}))
    with openai_client(server) as client:
        return client.chat.completions.create(**CHAT_ARGS).model_dump()


def _openai_stream(server: MockServer) -> Any:
    server.on(CHAT_PATH, Reply(events=chat_text_events()))
    with openai_client(server) as client:
        stream = client.chat.completions.create(stream=True, **CHAT_ARGS)
        return [chunk.model_dump() for chunk in stream]


def _anthropic_ok(server: MockServer) -> Any:
    server.on(MESSAGES_PATH, Reply(body=message_body()))
    with anthropic_client(server) as client:
        return client.messages.create(**MESSAGE_ARGS).model_dump()


def _anthropic_error(server: MockServer) -> Any:
    server.on(MESSAGES_PATH, Reply(429, error_body("rate_limit_error")))
    with anthropic_client(server) as client:
        return client.messages.create(**MESSAGE_ARGS).model_dump()


def _anthropic_stream(server: MockServer) -> Any:
    server.on(MESSAGES_PATH, Reply(events=text_events("Hel", "lo")))
    with anthropic_client(server) as client:
        events = client.messages.create(stream=True, **MESSAGE_ARGS)
        return [event.model_dump() for event in events]


def _anthropic_helper(server: MockServer) -> Any:
    server.on(MESSAGES_PATH, Reply(events=text_events("Hel", "lo")))
    with anthropic_client(server) as client, client.messages.stream(**MESSAGE_ARGS) as stream:
        return stream.get_final_message().model_dump()


CALLS: Dict[str, Callable[[MockServer], Any]] = {
    "openai ok": _openai_ok,
    "openai error": _openai_error,
    "openai stream": _openai_stream,
    "anthropic ok": _anthropic_ok,
    "anthropic error": _anthropic_error,
    "anthropic stream": _anthropic_stream,
    "anthropic helper": _anthropic_helper,
}
KINDS: List[str] = list(CALLS)


def run_call(kind: str, server: MockServer) -> Any:
    """What the caller of ``kind`` sees: the data returned, or the exception's class and text."""
    try:
        return CALLS[kind](server)
    except Exception as exc:
        return ("raised", type(exc).__name__, getattr(exc, "status_code", None), str(exc))


SLOWEST = 0.5
"""Seconds one recorded call may take with a stuck writer: honest calls take a millisecond."""


def quick(step: Callable[[], Any], what: str) -> None:
    """Run ``step`` and fail the moment it takes ``SLOWEST`` seconds (a blocked caller)."""
    began = time.perf_counter()
    step()
    took = time.perf_counter() - began
    assert took < SLOWEST, f"{what} was held up for {took:.2f} s"


def flood(count: int) -> None:
    """``count`` calls to ``record_call``, each of which must return at once."""
    for number in range(count):
        quick(
            lambda number=number: dagnam_record.record_call(
                provider="other", request={"n": number}, response={"r": 1}
            ),
            f"record_call number {number}",
        )
