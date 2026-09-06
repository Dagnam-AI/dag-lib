"""The canonical trace record every reader produces and every audit step consumes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class Message:
    """One prompt turn (``system`` and ``assistant`` turns are never stored here)."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class TraceRecord:
    """One LLM call as the audit sees it, independent of which tool exported it.

    ``ts`` is always UTC-aware; ``messages`` holds the prompt turns only, in
    order, with the system prompt split out into ``system`` and assistant turns
    dropped; every string has passed the terminal sanitizer.
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
