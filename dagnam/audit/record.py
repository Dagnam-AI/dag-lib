"""The canonical trace record every reader produces and every audit step consumes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class Message:
    """One prompt turn; a ``system`` turn is never stored here (it is the record's ``system``).

    An ``assistant`` turn is earlier context -- its text, or the tool calls it
    made as the JSON the student would answer with -- never the reply itself.
    """

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class TraceRecord:
    """One LLM call as the audit sees it, independent of which tool exported it.

    ``ts`` is always UTC-aware; ``messages`` holds the prompt turns in order,
    with the system prompt split out into ``system`` and earlier assistant
    turns kept as context; every string has passed the terminal sanitizer.
    ``prompt_tokens`` counts every prompt token, cached or not. ``response`` is
    the reply's text with any reasoning removed.
    """

    trace_id: str
    ts: datetime
    model: str
    system: str | None
    messages: tuple[Message, ...]
    response: str
    response_tool_calls: tuple[dict[str, Any], ...]
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float | None
    session_id: str | None
    outcome: float | None
    workload_hint: str | None
    cached_prompt_tokens: int = 0
    """How many of ``prompt_tokens`` were cache reads, which vendors bill at a lower rate."""
    has_media: bool = False
    """The prompt carried an image, audio, file or document part the text does not hold."""
    signature: str | None = None
    """The request's output schema or forced tool, as ``schema=<name>``; never its tool set."""
    reasoning_only: bool = False
    """The reply was nothing but the model's reasoning, so ``response`` is empty: the call is
    counted and priced like any other, and gives no training row."""
