"""What the fixture scenarios share: the mock server, one real client of each SDK, and the prompts."""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, List, Optional

from _fixture_bodies import chat_body, claude_body, response_body
from _mock_server import MockServer, Reply

CHAT = "/v1/chat/completions"
RESPONSES = "/v1/responses"
MESSAGES = "/v1/messages"
SYSTEM = "You are a support triage assistant. Reply with one intent label."
ORDER = "My widget order W-1001 arrived with a cracked lid."
LOOKUP = {
    "type": "function",
    "function": {
        "name": "lookup_order",
        "description": "Look up an order",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
}
ANTHROPIC_LOOKUP = {
    "name": "lookup_order",
    "description": "Look up an order",
    "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}},
}


class Env:
    """The mock server, one real client of each SDK, and the call number of the current file."""

    def __init__(self, server: MockServer, openai_client: Any, anthropic_client: Any) -> None:
        self.server = server
        self.openai = openai_client
        self.anthropic = anthropic_client
        self.epoch = 0
        self.n = 0

    def begin(self, epoch: int) -> None:
        """Start a file whose first call happens one second after ``epoch``."""
        self.epoch, self.n = epoch, 0

    def step(self) -> int:
        """The number of the next call (1, 2, ...)."""
        self.n += 1
        return self.n

    def chat(self, content: Optional[str], request: Dict[str, Any], **reply: Any) -> Any:
        """One Chat Completions call answered with ``content`` (and tool calls etc. in ``reply``)."""
        n = self.step()
        body = chat_body(self.epoch + n, n, content, **reply)
        self.server.on(CHAT, Reply(body=body))
        return self.openai.chat.completions.create(model="gpt-4o-mini", **request)

    def respond(self, output: List[Dict[str, Any]], request: Dict[str, Any], **usage: Any) -> Any:
        """One Responses call answered with the ``output`` items."""
        n = self.step()
        self.server.on(RESPONSES, Reply(body=response_body(self.epoch + n, n, output, **usage)))
        return self.openai.responses.create(model="gpt-4o-mini", **request)

    def claude(self, content: List[Dict[str, Any]], request: Dict[str, Any], **reply: Any) -> Any:
        """One Messages call answered with the ``content`` blocks."""
        n = self.step()
        self.server.on(MESSAGES, Reply(body=claude_body(n, content, **reply)))
        return self.anthropic.messages.create(model="claude-sonnet-4-5", max_tokens=64, **request)

    def fail(self, path: str, status: int, call: Callable[[], Any], cut: bool = False) -> None:
        """One call the server answers with an error status, or cuts off (``cut``)."""
        self.step()
        self.server.on(path, Reply(status, {"error": {"message": "no", "type": "error"}}, cut=cut))
        with contextlib.suppress(Exception):
            call()


def user(text: str) -> Dict[str, Any]:
    """A user message."""
    return {"role": "user", "content": text}
