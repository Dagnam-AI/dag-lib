"""The shapes a model's reasoning arrives in, as data: what each reader must read from each."""

from __future__ import annotations

import json
from typing import Any

# Neutral placeholders: R* is reasoning and must never be read, A* is the answer.

R = ("R-one", "R-two", "R-sig", "R-three")


A = "A-one"
A2 = "A-two"


GONE = "gone"  # what a shape with no answer in it is expected to read as: reasoning only


REASONING_ITEM = {
    "type": "reasoning",
    "id": "rs_1",
    "summary": [{"type": "summary_text", "text": R[0]}],
    "content": [{"type": "reasoning_text", "text": R[0]}],
    "encrypted_content": R[1],
}


ANSWER_ITEM = {
    "type": "message",
    "role": "assistant",
    "content": [{"type": "output_text", "text": A}],
}


BEDROCK = {"reasoningContent": {"reasoningText": {"text": R[0], "signature": R[2]}}}


LANGCHAIN_AWS = {"type": "reasoning_content", "reasoning_content": {"text": R[0]}}


AI_SDK = [
    {"type": "redacted-reasoning", "data": R[1]},
    {"type": "reasoning", "text": R[0]},
]


CHAT_MESSAGE = {
    "role": "assistant",
    "content": A,
    "reasoning_content": R[0],
    "reasoning": R[0],
    "reasoning_details": [{"type": "reasoning.text", "text": R[0], "signature": R[2]}],
}


BLOCKS = [{"type": "thinking", "thinking": R[0], "signature": R[2]}, {"type": "text", "text": A}]


REPLY_OBJECT = {"role": "assistant", "content": BLOCKS}

# name: (where it sits, the value, the reply text it reads as or GONE)
#   message: a reply message; items: a list of blocks or items; object: a whole response object;
#   lone: one block or item as the whole reply; text: a bare string.


SHAPES: dict[str, tuple[str, Any, str]] = {
    "reasoning_text part in a message": (
        "message",
        {
            "role": "assistant",
            "content": [
                {"type": "reasoning_text", "text": R[0]},
                {"type": "output_text", "text": A},
            ],
        },
        A,
    ),
    "summary_text part in a message": (
        "message",
        {
            "role": "assistant",
            "content": [{"type": "summary_text", "text": R[0]}, {"type": "output_text", "text": A}],
        },
        A,
    ),
    "a type nobody listed that names reasoning": (
        "message",
        {
            "role": "assistant",
            "content": [{"type": "Reasoning_Trace", "text": R[0]}, {"type": "text", "text": A}],
        },
        A,
    ),
    "reasoning fields beside the content": ("message", CHAT_MESSAGE, A),
    "a message holding only reasoning_content": (
        "message",
        {"role": "assistant", "content": None, "reasoning_content": R[0]},
        GONE,
    ),
    "a message whose content is JSON text of blocks": (
        "message",
        {"role": "assistant", "content": json.dumps(BLOCKS)},
        A,
    ),
    "a message with reasoning in the first of two text parts": (
        "message",
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"{A} "},
                {"type": "text", "text": f"<think>{R[0]}</think>{A2}"},
            ],
        },
        f"{A} {A2}",
    ),
    "a chat completions object": (
        "object",
        {"id": "chatcmpl-1", "object": "chat.completion", "choices": [{"message": CHAT_MESSAGE}]},
        A,
    ),
    "a chat completions object with inline think": (
        "object",
        {
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": f"<think>{R[0]}</think>{A}"}}],
        },
        A,
    ),
    "an ollama chat response": (
        "object",
        {
            "model": "m",
            "message": {"role": "assistant", "content": A, "thinking": R[0]},
            "done": True,
        },
        A,
    ),
    "an ollama generate response": (
        "object",
        {"model": "m", "response": A, "thinking": R[0], "done": True},
        A,
    ),
    "a gemini generateContent response": (
        "object",
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": R[0], "thought": True}, {"text": A}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"thoughtsTokenCount": 9},
        },
        A,
    ),
    "a bedrock converse response": (
        "object",
        {
            "output": {"message": {"role": "assistant", "content": [BEDROCK, {"text": A}]}},
            "stopReason": "end_turn",
        },
        A,
    ),
    "a cohere v2 response": (
        "object",
        {
            "id": "c1",
            "message": {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": R[0]}, {"type": "text", "text": A}],
            },
            "finish_reason": "COMPLETE",
        },
        A,
    ),
    # Response objects no reader recognises: nothing says which field is the answer, so the
    # object is never written out whole; the call is kept, priced, and gives no row.
    "a stream chunk": (
        "object",
        {"choices": [{"index": 0, "delta": {"reasoning_content": R[0]}}]},
        GONE,
    ),
    "a vercel ai sdk result": (
        "object",
        {"text": A, "reasoningText": R[0], "reasoning": [{"type": "reasoning", "text": R[0]}]},
        GONE,
    ),
    "a langchain serialized message": (
        "object",
        {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain", "schema", "messages", "AIMessage"],
            "kwargs": {"content": A, "additional_kwargs": {"reasoning_content": R[0]}},
        },
        GONE,
    ),
    "an unknown object with a thought part inside": (
        "object",
        {"result": {"parts": [{"text": R[0], "thought": True}, {"text": A}]}},
        GONE,
    ),
    "an ai sdk v4 result": (
        "object",
        {
            "text": A,
            "reasoning": R[0],
            "reasoningDetails": [{"type": "text", "text": R[0]}],
            "finishReason": "stop",
        },
        GONE,
    ),
    "a llamaindex chat response": (
        "object",
        {
            "message": {
                "role": "assistant",
                "blocks": [
                    {"block_type": "thinking", "content": R[0]},
                    {"block_type": "text", "text": A},
                ],
            },
            "raw": {},
        },
        GONE,
    ),
    "a haystack chat message": (
        "object",
        {"_role": "assistant", "_content": [{"text": A}, {"reasoning": {"reasoning_text": R[0]}}]},
        GONE,
    ),
    "bedrock blocks": ("items", [BEDROCK, {"text": A}], A),
    "langchain-aws blocks": ("items", [LANGCHAIN_AWS, {"type": "text", "text": A}], A),
    "ai sdk parts": ("items", [*AI_SDK, {"type": "text", "text": A}], A),
    "a responses output": ("items", [REASONING_ITEM, ANSWER_ITEM], A),
    "a lone reasoning item": ("lone", REASONING_ITEM, GONE),
    "a lone bedrock block": ("lone", BEDROCK, GONE),
    "a lone ai sdk part": ("lone", AI_SDK[0], GONE),
    "reasoning beside a hosted tool item": (
        "items",
        [REASONING_ITEM, {"type": "web_search_call", "id": "ws_1"}],
        GONE,
    ),
    "two inline think blocks": (
        "text",
        f"<think>{R[0]}</think>\n<think>{R[3]}</think>{A}",
        A,
    ),
    "inline think with an attribute": ("text", f'<think effort="high">{R[0]}</think>{A}', A),
    "inline think in capitals": ("text", f"<THINK>{R[0]}</THINK>{A}", A),
    "inline think that never closed": ("text", f"<think>{R[0]}", GONE),
    "reasoning then a bare closing tag": ("text", f"{R[0]}\n{R[3]}\n</think>\n{A}", A),
    "a thought tag": ("text", f"<thought>{R[0]}</thought>{A}", A),
    "a thinking tag": ("text", f"<thinking>{R[0]}</thinking>\n\n{A}", A),
    # The call ended inside its reasoning: counted and priced, and no row, through every reader.
    "reasoning then a closing tag and nothing": ("text", f"{R[0]}\n{R[3]}\n</think>", GONE),
    "reasoning then a closing tag and blank lines": ("text", f"{R[0]}\n</think>\n\n  \n", GONE),
    "reasoning then a closing tag and nothing, CRLF": ("text", f"{R[0]}\r\n</think>\r\n", GONE),
    "reasoning then a closing tag, with CRLF line ends": (
        "text",
        f"{R[0]}\r\n</think>\r\n\r\n{A}",
        A,
    ),
    # A closing tag that is part of the answer is not the template's: kept whole, in every reader.
    "prose that mentions a closing reasoning tag": (
        "text",
        f"{A}. In XML you close it with </reasoning> and then {A2}.",
        f"{A}. In XML you close it with </reasoning> and then {A2}.",
    ),
    "code with a closing thought tag": (
        "text",
        f"{A}\n```html\n<p>x</p></thought>\n```\n{A2}",
        f"{A}\n```html\n<p>x</p></thought>\n```\n{A2}",
    ),
    "a JSON answer with a closing think tag in a value": (
        "text",
        json.dumps({"label": A, "html": "a</think>b", "note": A2}),
        json.dumps({"label": A, "html": "a</think>b", "note": A2}),
    ),
    "an answer that quotes the closing tag in the middle of a line": (
        "text",
        f"{A} then a literal </think> tag and {A2}",
        f"{A} then a literal </think> tag and {A2}",
    ),
    "a reasoning tag": ("text", f"<reasoning>{R[0]}</reasoning>{A}", A),
    "think after a byte order mark": ("text", f"﻿​<think>{R[0]}</think>{A}", A),
    "a reply double encoded as JSON text": (
        "text",
        json.dumps(json.dumps(REPLY_OBJECT)),
        A,
    ),
    "a reply as JSON text of blocks": ("text", json.dumps(BLOCKS), A),
}


# Objects no reader recognises whose only reasoning sits under a name a structured answer may use
# for itself: they cannot be told from an answer, so they are kept as written (the stated residual).
# Not on the table, and why: ``reasoning`` and ``thinking`` (plain words), ``thoughts`` and a
# ``thought`` that is text (the table's flag is ``thought: true``), ``rationale``, ``scratchpad``,
# ``thinking_text`` (no vendor documents these; a schema may), and ``analysis`` as a ``type`` or
# ``channel`` (an ordinary label).
RESIDUAL: dict[str, Any] = {
    "legacy completions": {"choices": [{"text": A, "reasoning": R[0], "finish_reason": "stop"}]},
    "answer and thinking": {"answer": A, "thinking": R[0]},
    "completion and thoughts": {"completion": A, "thoughts": R[0]},
    "result and rationale": {"result": A, "rationale": R[0]},
    "final and scratchpad": {"final": A, "scratchpad": R[0]},
    "text and a thought that is text": {"text": A, "thought": R[0]},
    "text and thinking_text": {"text": A, "thinking_text": R[0]},
    "output_text and reasoning": {"output_text": A, "reasoning": R[0]},
}
