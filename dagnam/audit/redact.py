"""Redact every PII class the shared contract detects: before a row is cut, and again on the cut row.

A training row is redacted twice, and the second pass is the one that counts. The first
(:func:`redact_records`) runs on the whole record before anything is cut, so a cut to the
character budget can never land inside an identifier and ship the tail of it. The second
(:func:`redact_rows`) runs on the row exactly as it will be uploaded -- cut to length, case
folded -- and is never followed by a cut: whatever the first pass left, and any text a cut joined,
is redacted last, so the uploaded row scans clean. A row already redacted is returned unchanged.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
import functools
from typing import Any

from dagnam_contracts.hygiene import PII_CODES, PiiAction, redact_rows as contract_redact_rows

from dagnam.audit.readers.messages import effective_response
from dagnam.audit.record import Message, TraceRecord

# Every class the contract detects is redacted -- the classes come from the contract, so a
# detector added there is redacted here.
PII_POLICY: dict[str, PiiAction] = dict.fromkeys(PII_CODES, "redact")


@dataclass(frozen=True, slots=True)
class RedactStats:
    """Findings per class, the classes looked for, and how many rows changed."""

    counts: dict[str, int]
    pass_list: tuple[str, ...]
    rows_changed: int


def redact_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], RedactStats]:
    """Return redacted copies of ``rows`` (never dropped) with the per-class counts.

    Call it on rows already in their final shape, and never cut them afterwards. The counts hold
    every class, zero included, and are the number of placeholders of each class written.
    """
    redacted, counts = contract_redact_rows(rows)
    changed = sum(new is not old for new, old in zip(redacted, rows, strict=True))
    return redacted, RedactStats(counts=counts, pass_list=PII_CODES, rows_changed=changed)


def redact_records(records: Sequence[TraceRecord]) -> tuple[list[TraceRecord], RedactStats]:
    """Redact every string a training row is built from, before any of it is cut to length.

    A record's answer is its :func:`effective_response` -- its tool calls when
    it made some -- so the redacted copy carries that as its ``response`` and no
    tool calls: what is trained on is exactly what was redacted. A cut made
    after this pass can never expose part of an identifier; the row is redacted again
    once it is cut. The system prompt an agent repeats on every call is
    redacted once and its findings counted on every record.
    """
    rows = [
        {
            "messages": [m.content for m in r.messages],
            "response": effective_response(r.response, r.response_tool_calls),
        }
        for r in records
    ]
    redacted, stats = redact_rows(rows)
    counts = Counter(stats.counts)
    out: list[TraceRecord] = []
    changed = 0
    for record, row, original in zip(records, redacted, rows, strict=True):
        system, found = _system(record.system) if record.system else (record.system, ())
        counts.update(dict(found))
        changed += row != original or system != record.system
        messages = zip(record.messages, row["messages"], strict=True)
        out.append(
            replace(
                record,
                system=system,
                messages=tuple(Message(m.role, content) for m, content in messages),
                response=row["response"],
                response_tool_calls=(),
            )
        )
    return out, RedactStats(counts=dict(counts), pass_list=stats.pass_list, rows_changed=changed)


@functools.lru_cache(maxsize=1_024)
def _system(system: str) -> tuple[str, tuple[tuple[str, int], ...]]:
    """One system prompt redacted, and what was found in it."""
    (row,), stats = redact_rows([{"system": system}])
    return str(row["system"]), tuple(stats.counts.items())


__all__ = ["PII_POLICY", "RedactStats", "redact_records", "redact_rows"]
