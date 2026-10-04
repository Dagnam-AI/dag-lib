"""The chat endpoint and the rows and clock the orchestration tests share with ``FakePlatform``."""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any

from dagnam_contracts.prompts import render_chat_prompt
from tests.typing_helpers import RequestsMocker

CHAT_URL = "https://x/v1/chat/completions"


def serve_chat(
    requests_mock: RequestsMocker, answer: Callable[[list[dict[str, str]]], str | None]
) -> list[dict[str, Any]]:
    """Serve the OpenAI-compatible route from ``answer(messages)``; ``None`` answers 500."""
    seen: list[dict[str, Any]] = []

    def respond(request: Any, context: Any) -> dict[str, Any]:
        body = json.loads(request.text)
        seen.append({"body": body, "authorization": request.headers.get("Authorization")})
        content = answer(body["messages"])
        if content is None:
            context.status_code = 500
            return {"error": "replica crashed"}
        return {"choices": [{"message": {"role": "assistant", "content": content}}]}

    requests_mock.post(CHAT_URL, json=respond)
    return seen


def last_user(messages: list[dict[str, str]]) -> str:
    """The content of the last ``user`` turn, the way the fake endpoint keys its answers."""
    return next(m["content"] for m in reversed(messages) if m["role"] == "user")


def teacher(messages: list[dict[str, str]]) -> str:
    """The teacher's answer for a fixture prompt: ``ticket N`` -> a/b, ``order N`` -> JSON."""
    kind, _, number = last_user(messages).partition(" ")
    if kind == "ticket":
        return "a" if int(number) % 2 else "b"
    return json.dumps({"order_id": number, "product": "x"})


def label_row(i: int) -> dict[str, Any]:
    """A ``labeled-example`` row exactly as the scan derives one."""
    turns = [{"role": "user", "content": f"ticket {i}"}]
    return {
        "input": render_chat_prompt(turns, system="Classify the ticket"),
        "label": teacher(turns),
    }


def json_row(i: int) -> dict[str, Any]:
    """A ``chat-messages`` row exactly as the scan derives one."""
    turns = [{"role": "system", "content": "Extract"}, {"role": "user", "content": f"order {i}"}]
    return {"messages": [*turns, {"role": "assistant", "content": teacher(turns)}]}


class Clock:
    """A clock the waits drive: ``sleep`` advances ``now`` and records the request."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds
