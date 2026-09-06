"""Builders for hand-made :class:`TraceRecord` values used by the discovery tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from dagnam.audit import Message, TraceRecord

T0 = datetime(2026, 8, 1, 9, tzinfo=UTC)


def make_record(
    *,
    system: str | None = "Label the ticket.",
    response: str = "billing",
    ts: datetime = T0,
    model: str = "gpt-4o-mini",
    tool_calls: tuple[dict[str, Any], ...] = (),
    prompt_tokens: int = 100,
    completion_tokens: int = 2,
    latency_ms: float = 500.0,
    cost_usd: float | None = 0.001,
    trace_id: str = "t",
) -> TraceRecord:
    return TraceRecord(
        trace_id=trace_id,
        ts=ts,
        model=model,
        system=system,
        messages=(Message("user", "hi"),),
        response=response,
        response_tool_calls=tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        session_id=None,
        outcome=None,
        workload_hint=None,
    )


def make_records(
    count: int,
    *,
    system: str | None = "Label the ticket.",
    response: str | None = None,
    model: str = "gpt-4o-mini",
    cost_usd: float | None = 0.001,
) -> list[TraceRecord]:
    """``count`` records one minute apart; ``response`` cycles ``a``/``b`` unless given."""
    return [
        make_record(
            system=system,
            response="ab"[i % 2] if response is None else response,
            ts=T0 + timedelta(minutes=i),
            model=model,
            cost_usd=cost_usd,
            trace_id=f"t{i}",
        )
        for i in range(count)
    ]
