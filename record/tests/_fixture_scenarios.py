"""The calls ``gen_fixtures.py`` makes with the real ``openai`` and ``anthropic`` clients.

One function per output file. Each makes its calls one after another against the mock server, so
the file holds one line per call in call order (the writer keeps the queue's order).
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, List, Optional

from _anthropic_bodies import (
    block_delta,
    block_start,
    block_stop,
    head,
    json_delta,
    tail,
    text_delta,
    text_events,
    tool_events,
)
from _fixture_bodies import (
    CLAUDE_MODEL,
    PDF,
    PNG,
    WAV,
    call_item,
    chat_body,
    claude_body,
    message_item,
    response_body,
    text_block,
    tool_block,
    tool_call,
)
from _mock_server import MockServer, Reply
from _openai_bodies import chat_text_events, chat_tool_events, response_events

import dagnam_record

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


def chat_file(env: Env) -> None:
    """Chat Completions: the shapes of a prompt and of an answer."""
    prompt = [{"role": "system", "content": SYSTEM}, user(ORDER)]
    env.chat("returns", {"messages": prompt})
    env.chat("returns", {"messages": prompt}, usage=(2000, 2, 1920))
    env.chat(
        None,
        {"messages": [{"role": "system", "content": SYSTEM}, user("Where is order W-1001?")]}
        | {"tools": [LOOKUP]},
        tool_calls=[tool_call(3, "lookup_order", '{"order_id":"W-1001"}')],
    )
    image = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}
    text = {"type": "text", "text": "What is wrong here?"}
    env.chat("damaged", {"messages": [{"role": "user", "content": [text, image]}]})
    env.chat(
        "refunds",
        {
            "messages": [
                {"role": "developer", "content": "Be brief."},
                user("Hi"),
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Hel"},
                        {"type": "text", "text": "lo"},
                    ],
                },
                user("Which team handles refunds?"),
            ]
        },
    )
    env.chat(
        "Checking both.",
        {"messages": [user("Order W-1001 and W-1002, please.")], "tools": [LOOKUP]},
        tool_calls=[
            tool_call(6, "lookup_order", '{"order_id":"W-1001"}'),
            tool_call(6, "lookup_order", '{"order_id":"W-1002"}'),
        ],
    )
    earlier = {
        "role": "assistant",
        "content": None,
        "tool_calls": [tool_call(7, "lookup_order", '{"order_id": "W-1001", "express": true}')],
    }
    env.chat(
        "It shipped on Monday.",
        {
            "messages": [
                {"role": "system", "content": SYSTEM},
                user("Where is order W-1001?"),
                earlier,
                {"role": "tool", "tool_call_id": "call_7", "content": '{"status":"shipped"}'},
                user("Thanks. And the refund?"),
            ]
        },
    )
    legacy = {"name": "lookup_order", "arguments": '{"order_id":"W-1002"}'}
    env.chat(
        None,
        {"messages": [user("Where is order W-1002?")], "functions": [LOOKUP["function"]]},
        function_call=legacy,
        finish="function_call",
    )
    env.chat(
        "Done.",
        {
            "messages": [
                user("Where is order W-1002?"),
                {"role": "assistant", "content": None, "function_call": legacy},
                {"role": "function", "name": "lookup_order", "content": "shipped"},
                user("Thanks."),
            ]
        },
    )
    env.chat(
        "ok",
        {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "developer", "content": "Rules."},
                user("Hi"),
            ]
        },
    )
    env.chat(
        "ok",
        {
            "messages": [
                {"role": "system", "content": "Be\x07 brief\x1b."},
                user("Hel\x00lo ‮ there\x7f"),
            ]
        },
    )
    custom = {"id": "call_12", "type": "custom", "custom": {"name": "run_shell", "input": "ls -l"}}
    env.chat(None, {"messages": [user("List the files.")]}, tool_calls=[custom])
    thought = "<think>The customer wants a status.</think>Sure, one moment."
    env.chat(
        "It shipped.",
        {
            "messages": [
                user("Where is order W-1001?"),
                {"role": "assistant", "content": thought},
                user("Well?"),
            ]
        },
    )


def openai_stream_file(env: Env) -> None:
    """Streams: Chat Completions text and tool calls, and a Responses stream."""
    request = {"stream": True, "stream_options": {"include_usage": True}}
    env.step()
    env.server.on(CHAT, Reply(events=chat_text_events()))
    for _ in env.openai.chat.completions.create(
        model="gpt-4o-mini", messages=[user("Say hello.")], **request
    ):
        pass
    env.step()
    env.server.on(CHAT, Reply(events=chat_tool_events()))
    for _ in env.openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[user("Weather and time in Paris?")],
        tools=[LOOKUP],
        **request,
    ):
        pass
    env.step()
    env.server.on(RESPONSES, Reply(events=response_events()))
    for _ in env.openai.responses.create(model="gpt-4o-mini", input="Say hello.", stream=True):
        pass


def parse_file(env: Env) -> None:
    """``parse()`` with a pydantic model: the request's schema is recorded, the answer carries ``parsed``."""
    from pydantic import BaseModel

    class Ticket(BaseModel):
        intent: str
        urgent: bool

    answer = '{"intent":"returns","urgent":false}'
    n = env.step()
    env.server.on(CHAT, Reply(body=chat_body(env.epoch + n, n, answer)))
    env.openai.chat.completions.parse(
        model="gpt-4o-mini", messages=[user(ORDER)], response_format=Ticket
    )
    n = env.step()
    env.server.on(RESPONSES, Reply(body=response_body(env.epoch + n, n, [message_item(n, answer)])))
    env.openai.responses.parse(model="gpt-4o-mini", input=ORDER, text_format=Ticket)


def responses_file(env: Env) -> None:
    """Responses: instructions and input in, output items out."""
    env.respond(
        [message_item(1, "high")],
        {"instructions": "Reply with one urgency label.", "input": "Refund me today!"},
    )
    env.respond(
        [call_item(2, "open_ticket", '{"priority":"high"}')],
        {"instructions": "Open a ticket.", "input": "Refund me today!", "tools": [LOOKUP]},
    )
    image = {"type": "input_image", "image_url": f"data:image/png;base64,{PNG}"}
    text = {"type": "input_text", "text": "What is wrong here?"}
    env.respond(
        [message_item(3, "damaged")], {"input": [{"role": "user", "content": [text, image]}]}
    )
    env.respond(
        [message_item(4, "It shipped on Monday.")],
        {
            "instructions": "Be brief.",
            "input": [
                user("Where is order W-1001?"),
                {
                    "type": "function_call",
                    "call_id": "call_9",
                    "name": "lookup_order",
                    "arguments": '{"order_id":"W-1001"}',
                },
                {"type": "function_call_output", "call_id": "call_9", "output": "shipped"},
                user("Thanks."),
            ],
        },
    )
    env.respond(
        [message_item(5, "ok")],
        {
            "instructions": "",
            "input": [{"role": "developer", "content": "Rules."}, user("Hi")],
        },
    )
    reasoning = {"id": "rs_6", "type": "reasoning", "summary": []}
    env.respond(
        [reasoning, message_item(6, "high"), call_item(6, "open_ticket", '{"priority":"high"}')],
        {"instructions": "Reply.", "input": "Refund me today!"},
    )
    custom = {
        "id": "ctc_7",
        "type": "custom_tool_call",
        "call_id": "call_7",
        "name": "run_shell",
        "input": "ls -l",
        "status": "completed",
    }
    env.respond([custom], {"input": "List the files."})
    earlier = [
        {"type": "custom_tool_call", "call_id": "call_8", "name": "run_shell", "input": "ls"},
        {"type": "custom_tool_call_output", "call_id": "call_8", "output": "a b"},
    ]
    env.respond(
        [message_item(8, "Two files.")],
        {"input": [{"role": "developer", "content": "Rules."}, user("List."), *earlier, user("?")]},
    )


def anthropic_file(env: Env) -> None:
    """Messages: system strings and blocks, cache counters, thinking, tool use and server tools."""
    ask = {"role": "user", "content": ORDER}
    env.claude(
        [text_block("returns")],
        {"system": "You are a triage bot. Answer in one word.", "messages": [ask]},
    )
    cached = {"type": "text", "text": "Triage bot.", "cache_control": {"type": "ephemeral"}}
    env.claude(
        [text_block("returns")],
        {"system": [cached, {"type": "text", "text": " One word."}], "messages": [ask]},
        usage=(12, 4, 900, 100),
    )
    thinking = {"type": "thinking", "thinking": "A cracked lid is a return.", "signature": "sig"}
    env.claude([thinking, text_block("returns")], {"messages": [ask]})
    env.claude(
        [tool_block(4, "lookup_order", {"order_id": "W-1001"})],
        {"messages": [user("Where is order W-1001?")], "tools": [ANTHROPIC_LOOKUP]},
        stop="tool_use",
    )
    document = {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": PDF},
    }
    env.claude(
        [text_block("invoice")],
        {
            "system": "Plain string system.",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "What is it?"}, document]}
            ],
        },
    )
    env.claude(
        [
            text_block("Checking both."),
            tool_block(6, "lookup_order", {"order_id": "W-1001"}),
            tool_block(6, "lookup_order", {"order_id": "W-1002"}),
        ],
        {"messages": [user("Orders W-1001 and W-1002?")], "tools": [ANTHROPIC_LOOKUP]},
        stop="tool_use",
    )
    env.claude([thinking], {"messages": [ask]}, stop="max_tokens")
    asked = {
        "role": "assistant",
        "content": [
            text_block("Let me look."),
            tool_block(8, "lookup_order", {"order_id": "W-1001", "express": True}),
        ],
    }
    answer = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_8", "content": "shipped"}],
    }
    env.claude(
        [text_block("It shipped.")],
        {"messages": [user("Where is order W-1001?"), asked, answer, user("Thanks.")]},
    )
    search = [
        {
            "type": "server_tool_use",
            "id": "srvtoolu_9",
            "name": "web_search",
            "input": {"query": "q"},
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_9",
            "content": [
                {
                    "type": "web_search_result",
                    "title": "T",
                    "url": "https://example.com",
                    "encrypted_content": "enc",
                    "page_age": None,
                }
            ],
        },
        text_block("It shipped on Monday."),
    ]
    env.claude(search, {"messages": [user("Search for order W-1001.")]})
    env.claude(
        [text_block("ok")],
        {"system": "", "messages": [user("Hi")]},
    )
    env.claude([text_block("ok")], {"system": "Be\x07 brief.", "messages": [user("Hel\x00lo")]})
    earlier = {"role": "assistant", "content": [thinking, text_block("Sure, one moment.")]}
    env.claude(
        [text_block("It shipped.")], {"messages": [user("Where is it?"), earlier, user("Well?")]}
    )


def anthropic_stream_file(env: Env) -> None:
    """Streams: text, parallel tool use, thinking then text, and a tool call cut at ``max_tokens``."""
    thinking = [
        *head(),
        block_start(0, {"type": "thinking", "thinking": "", "signature": ""}),
        block_delta(0, {"type": "thinking_delta", "thinking": "A cracked lid."}),
        block_delta(0, {"type": "signature_delta", "signature": "sig"}),
        block_stop(0),
        block_start(1, {"type": "text", "text": ""}),
        block_delta(1, text_delta("returns")),
        block_stop(1),
        *tail(),
    ]
    cut = [
        *head(),
        block_start(0, tool_block(0, "weather", {})),
        block_delta(0, json_delta('{"city": "Pa')),
        block_stop(0),
        *tail("max_tokens"),
    ]
    for events in (text_events(), tool_events(), thinking, cut):
        env.step()
        env.server.on(MESSAGES, Reply(events=events))
        for _ in env.anthropic.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=64,
            messages=[user("Weather and time in Paris?")],
            stream=True,
        ):
            pass


def media_file(env: Env) -> None:
    """Prompts that carry media of each kind: the recorder keeps only each part's size."""
    audio = {"type": "input_audio", "input_audio": {"data": WAV, "format": "wav"}}
    env.chat("shipping", {"messages": [{"role": "user", "content": [audio]}]})
    attached = {"type": "file", "file": {"filename": "a.pdf", "file_data": PDF}}
    env.chat(
        "invoice",
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Read it."}, attached]}
            ]
        },
    )
    image = {"type": "input_image", "image_url": f"data:image/png;base64,{PNG}"}
    env.respond([message_item(3, "damaged")], {"input": [{"role": "user", "content": [image]}]})
    source = {"type": "base64", "media_type": "image/png", "data": PNG}
    env.claude(
        [text_block("damaged")],
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is wrong?"},
                        {"type": "image", "source": source},
                    ],
                }
            ]
        },
    )
    shown = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_5",
                "content": [
                    {"type": "text", "text": "Screenshot:"},
                    {"type": "image", "source": source},
                ],
            }
        ],
    }
    asked = {"role": "assistant", "content": [tool_block(5, "screenshot", {})]}
    env.claude([text_block("damaged")], {"messages": [user("Look."), asked, shown]})


def norm_only_file(env: Env) -> None:
    """The same kinds of call as the other files, recorded with ``no_raw=True``."""
    env.chat(
        "shipping",
        {
            "messages": [
                {"role": "system", "content": "Label the intent."},
                user("Where is my order?"),
            ]
        },
        usage=(20, 2, 0),
    )
    env.claude(
        [tool_block(2, "close_account", {})],
        {"messages": [user("Close my account.")], "tools": [ANTHROPIC_LOOKUP]},
        stop="tool_use",
    )
    env.respond([message_item(3, "high")], {"instructions": "Label.", "input": "Refund me today!"})
    earlier = {
        "role": "assistant",
        "content": None,
        "tool_calls": [tool_call(4, "lookup_order", '{"order_id":"W-1001"}')],
    }
    env.chat(
        "It shipped.",
        {
            "messages": [
                user("Where is order W-1001?"),
                earlier,
                {"role": "tool", "tool_call_id": "call_4", "content": "shipped"},
                user("Thanks."),
            ]
        },
    )


def mixed_bad_file(env: Env) -> None:
    """Good calls among failures: a 429, a 500, a dead connection (599) and an overloaded 529."""
    prompt = {"messages": [{"role": "system", "content": SYSTEM}, user(ORDER)]}
    env.chat("returns", prompt)
    env.fail(CHAT, 429, lambda: env.openai.chat.completions.create(model="gpt-4o-mini", **prompt))
    env.chat("returns", prompt)
    env.fail(CHAT, 500, lambda: env.openai.chat.completions.create(model="gpt-4o-mini", **prompt))
    env.claude([text_block("returns")], {"messages": [user(ORDER)]})
    env.fail(
        CHAT,
        200,
        lambda: env.openai.chat.completions.create(model="gpt-4o-mini", **prompt),
        cut=True,
    )
    env.fail(
        MESSAGES,
        529,
        lambda: env.anthropic.messages.create(
            model=CLAUDE_MODEL, max_tokens=64, messages=[user(ORDER)]
        ),
    )


def other_file(env: Env) -> None:
    """Calls the SDK patches cannot see, handed over with ``record_call``: no norm for these."""
    env.step()
    dagnam_record.record_call(
        provider="other",
        request={"prompt": "Where is my order?"},
        response={"text": "shipping"},
        latency_ms=1.0,
    )
    env.step()
    dagnam_record.record_call(
        provider="openai",
        request={"input": "x"},
        response={"output": "y"},
        latency_ms=1.0,
    )
