"""Reasoning, in every provider's spelling: one table, and the rule that keeps it out of every row.

A model's reasoning is never what a student should answer and never context to
train on, so nothing the table below names reaches a prompt or a reply. The
rule has three parts, applied by :mod:`~dagnam.audit.readers.messages` for every reader:

- a reply is read only through the fields known to hold the answer (its
  ``content``, ``parts``, ``tool_calls``), so reasoning that sits beside them
  is never read;
- a response object that no reader recognises is never written out whole:
  when reasoning sits anywhere in it (:func:`carries_reasoning`, strict) its
  answer cannot be told from its reasoning, and the call is kept for its count
  and its spend but gives no training row;
- reasoning written before the answer as inline tags is cut
  (:func:`strip_inline_reasoning`): leading ``<think>``-style blocks, and the one
  shape a chat template produces without an opening tag (see :func:`_bare_close`).

What is not on the table is kept as written, on purpose: a name a structured
answer may use for itself (``reasoning``, ``thinking``, ``thoughts``, ``thought``
as text, ``rationale``, ``scratchpad``, ``thinking_text``), a ``type`` or
``channel`` called ``analysis``, and raw channel-delimited text. In an object no
reader recognises these cannot be told from the answer's own fields, so they are
not guessed at.

A new vendor spelling is one entry here, and a type that names ``reasoning`` or
``thinking`` is caught before anyone lists it.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

REASONING_TYPES = frozenset(
    {
        "thinking",  # Anthropic, Mistral, Cohere
        "redacted_thinking",  # Anthropic
        "redacted-thinking",
        "reasoning",  # OpenAI Responses output item, Vercel AI SDK part
        "reasoning_text",  # OpenAI Conversations message content
        "reasoning_summary_text",
        "summary_text",  # OpenAI reasoning summary part
        "reasoning_content",  # LangChain (AWS Bedrock) block
        "redacted-reasoning",  # Vercel AI SDK part
    }
)
"""The ``type`` of a content part or output item that holds the model's reasoning."""
REASONING_TYPE_MARKERS = ("reason", "think")
"""A ``type`` that contains either word, in any case, is reasoning whether or not it is listed."""
REASONING_KEYS = ("reasoningContent", "encrypted_content")
"""A typeless block that holds something under one of these is reasoning (Bedrock; OpenAI's opaque state)."""
REASONING_FLAGS = ("thought",)
"""A part with one of these ``True`` is reasoning (Gemini ``thought: true``)."""
REASONING_MESSAGE_FIELDS = (
    "reasoning_content",  # DeepSeek, OpenRouter, LiteLLM
    "reasoning_details",  # OpenRouter
    "reasoningDetails",  # Vercel AI SDK v4 results
    "reasoningContent",  # Bedrock Converse
    "reasoningText",  # Vercel AI SDK
    "reasoning_summary",  # OpenAI Responses
    "reasoning_text",  # OpenAI, Haystack
    "thinking_blocks",  # LiteLLM
    "reasoning",
    "thinking",
)
"""Fields of a chat message that hold its reasoning beside the ``content``: never read.

Every name above the last two is one a vendor or SDK documents for reasoning. ``reasoning``
and ``thinking`` are plain words (:data:`REASONING_PLAIN_FIELDS`)."""
REASONING_PLAIN_FIELDS = ("reasoning", "thinking")
"""The fields above that a structured answer can also use as its own (chain of thought in a schema)."""
REASONING_TAGS = ("think", "thinking", "thought", "reasoning")
"""Inline tags: ``<think>`` (DeepSeek, Qwen), ``<thought>`` (Gemini's compatible endpoint), and the rest."""
_TAG = "(?:" + "|".join(REASONING_TAGS) + ")"
_SKIP = r"\s\ufeff\u200b-\u200d\u2060"  # whitespace, a BOM, zero-width characters
_FLAGS = re.DOTALL | re.IGNORECASE
_LEADING_BLOCKS = re.compile(
    rf"\A(?:[{_SKIP}]*<{_TAG}\b[^>]*>.*?(?:</{_TAG}\s*>|\Z))+[{_SKIP}]*", _FLAGS
)
"""Blocks before the answer, any number; one cut off before it closed runs to the end."""
_OPENING = re.compile(rf"<{_TAG}\b", re.IGNORECASE)
_CLOSE = "</think>"
_BLANK_TO_END = re.compile(r"\s*\Z")


def is_reasoning_type(kind: object) -> bool:
    """Whether a ``type`` value names reasoning."""
    return isinstance(kind, str) and (
        kind in REASONING_TYPES or any(mark in kind.lower() for mark in REASONING_TYPE_MARKERS)
    )


def is_reasoning(part: Mapping[str, Any]) -> bool:
    """Whether a content part, output item or block is the model's reasoning."""
    return (
        is_reasoning_type(part.get("type"))
        or is_reasoning_type(part.get("block_type"))  # LlamaIndex
        or any(part.get(flag) is True for flag in REASONING_FLAGS)
        or any(part.get(key) for key in REASONING_KEYS)
    )


def strip_inline_reasoning(text: str) -> str:
    """``text`` without the reasoning a model may write before its answer.

    Leading ``<think>`` blocks (any tag of :data:`REASONING_TAGS`, any case,
    attributes allowed), or, for the template shape of :func:`_bare_close`,
    everything up to its ``</think>`` (all of it when no answer follows). A block
    in the middle of an answer, and any other closing tag, is left alone.
    """
    match = _LEADING_BLOCKS.match(text)
    if match:
        return text[match.end() :]
    end = _bare_close(text)
    return text if end is None else text[end:]


def _bare_close(text: str) -> int | None:
    """Where the answer starts in a reply of the shape a chat template produces, else ``None``.

    The templates of DeepSeek-R1, Qwen3 and QwQ (without a reasoning parser) open the
    block themselves, so the reply is the reasoning, ``</think>`` alone on its line,
    then the answer. Only that closing tag, in lower case; never an opening tag of
    any reasoning tag anywhere in the reply; the closing tag exactly once, alone on
    its line (spaces and tabs around it allowed). Text after it starts the answer;
    nothing after it, with reasoning before it, means the call ended inside its
    reasoning, and the whole reply is reasoning (the offset is the end of the text).
    A closing tag in prose or in code, in the middle of a line, repeated, upper case,
    or ending the reasoning's last line is an answer that mentions it, and is kept.
    The tag is found with ``str.find`` and its line checked by slicing, so a long
    reply costs no more than its length (a pattern over the lines before it kept a
    backtracking frame for each).
    """
    if text.count(_CLOSE) != 1 or _OPENING.search(text):
        return None
    at = text.find(_CLOSE)
    line_start = text.rfind("\n", 0, at) + 1
    line_end = text.find("\n", at)
    if line_end == -1:
        line_end = len(text)
    if text[line_start:at].strip(" \t") or text[at + len(_CLOSE) : line_end].strip(" \t\r"):
        return None
    if _BLANK_TO_END.match(text, line_end):
        return len(text) if text[:line_start].strip() else None
    start = line_end + 1
    lines = text[start:].splitlines(keepends=True)
    first = next(
        i for i, line in enumerate(lines) if line.strip()
    )  # one exists: not blank to the end
    return start + sum(len(line) for line in lines[:first])


def carries_reasoning(value: object, *, strict: bool = False) -> bool:
    """Whether reasoning sits anywhere in ``value``: a part, an item, a message's own field, a tag.

    ``strict`` leaves out :data:`REASONING_PLAIN_FIELDS`: in an object no reader
    recognises, ``reasoning`` and ``thinking`` may be the answer's own fields.
    """
    fields = tuple(
        f for f in REASONING_MESSAGE_FIELDS if not (strict and f in REASONING_PLAIN_FIELDS)
    )
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            if _LEADING_BLOCKS.match(item) or _bare_close(item) is not None:
                return True
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, Mapping):
            if is_reasoning(item) or any(item.get(field) for field in fields):
                return True
            pending.extend(item.values())
    return False
