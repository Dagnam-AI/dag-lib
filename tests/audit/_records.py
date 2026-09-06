"""Builders for hand-made :class:`TraceRecord` and :class:`Workload` values."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from dagnam.audit import Message, TraceRecord, Workload
from dagnam.audit.structure import StructureClass
from dagnam.audit.thresholds import STRUCTURE_SAMPLE

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


def make_workload(
    *,
    calls_per_day: float = 200.0,
    cost_month: float | None = 120.0,
    cls: StructureClass = StructureClass.ENUM_LABEL,
    n: int = 5_000,
    models: tuple[str, ...] = ("gpt-4o-mini",),
    prompt_per_call: int = 100,
    completion_per_call: int = 2,
    confidence: Literal["high", "low"] = "high",
) -> Workload:
    """A discovered workload with ``n`` calls; ``cost_month`` ``None`` means the export had no cost."""
    return Workload(
        id="w1",
        template_hash="w1",
        template_excerpt="Label the ticket.",
        structure_class=cls,
        confidence=confidence,
        calls=n,
        calls_per_day=calls_per_day,
        prompt_tokens=n * prompt_per_call,
        completion_tokens=n * completion_per_call,
        cost_usd_month=cost_month,
        cost_source="export" if cost_month is not None else "unknown",
        latency_p50_ms=400.0,
        latency_p95_ms=900.0,
        distinct_outputs=3,
        entropy_bits=1.5,
        stability=1.0,
        sample_size=min(n, STRUCTURE_SAMPLE),
        models=models,
        record_indices=tuple(range(n)),
    )
