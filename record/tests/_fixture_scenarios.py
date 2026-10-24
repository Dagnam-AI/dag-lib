"""The OpenAI calls ``gen_fixtures.py`` makes with the real ``openai`` client (one function per file).

Each makes its calls one after another against the mock server, so the file holds one line per call
in call order (the writer keeps the queue's order).
"""

from __future__ import annotations

from _fixture_bodies import (
    PNG,
    call_item,
    chat_body,
    message_item,
    response_body,
    tool_call,
)
from _fixture_env import CHAT, LOOKUP, ORDER, RESPONSES, SYSTEM, Env, user
from _mock_server import Reply
from _openai_bodies import chat_text_events, chat_tool_events, response_events


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
    ticket = '{"type": "ticket", "content": "Refund me."}'  # an agent's JSON prompt, not a message
    env.chat("refunds", {"messages": [{"role": "system", "content": ticket}, user(ticket)]})
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
