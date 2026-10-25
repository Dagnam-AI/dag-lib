"""The Anthropic, media, norm-only, failure and ``record_call`` calls ``gen_fixtures.py`` makes.

One function per output file, like ``_fixture_scenarios.py``.
"""

from __future__ import annotations

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
    message_item,
    text_block,
    tool_block,
    tool_call,
)
from _fixture_env import (
    ANTHROPIC_LOOKUP,
    CHAT,
    MESSAGES,
    ORDER,
    SYSTEM,
    Env,
    user,
)
from _mock_server import Reply

import dagnam_record


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
            tool_block(
                8, "lookup_order", {"order_id": "W-1001", "express": True, "note": "café 東京"}
            ),
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
    # Two gaps norm cannot close, pinned in EXPECTED_DIFFERENCES: the reader strips a leading
    # <think> block from each text part of a reply on its own, but norm has the parts joined; and
    # it reads a string that is JSON of a message as its content only for a plain-string reply.
    env.claude(
        [text_block("Hello "), text_block("<think>hmm</think>world")],
        {"messages": [user("Say hello.")]},
    )
    env.claude([text_block('{"type": "t", "content": "z"}')], {"messages": [user("Answer.")]})


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
