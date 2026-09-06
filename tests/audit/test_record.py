"""The canonical trace record is an immutable value."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from dagnam.audit import Message, TraceRecord


def _record() -> TraceRecord:
    return TraceRecord(
        trace_id="t1",
        ts=datetime(2026, 8, 1, tzinfo=UTC),
        model="gpt-4o-mini",
        system="be brief",
        messages=(Message("user", "hi"),),
        response="hello",
        response_tool_calls=(),
        prompt_tokens=3,
        completion_tokens=1,
        latency_ms=12.5,
        cost_usd=None,
        session_id=None,
        outcome=None,
        workload_hint=None,
    )


def test_record_is_frozen_and_slotted() -> None:
    record = _record()
    with pytest.raises(FrozenInstanceError):
        record.__setattr__("response", "changed")
    assert not hasattr(record, "__dict__")
    assert record == _record()


def test_message_is_frozen() -> None:
    message = Message("user", "hi")
    with pytest.raises(FrozenInstanceError):
        message.__setattr__("content", "x")
