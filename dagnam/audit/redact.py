"""Redact every PII class the shared contract detects before a row touches disk (spec §10)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
import functools
from typing import Any

from dagnam_contracts.hygiene import PII_CODES, PiiAction, apply_pii_policy, scan_rows

from dagnam.audit.readers.messages import effective_response
from dagnam.audit.record import Message, TraceRecord

# Spec U3: ``apply_pii_policy(rows, {every class: "redact"})`` -- the classes
# come from the contract, so a detector added there is redacted here.
PII_POLICY: dict[str, PiiAction] = dict.fromkeys(PII_CODES, "redact")


@dataclass(frozen=True, slots=True)
class RedactStats:
    """Findings per class, the classes looked for, and how many rows changed."""

    counts: dict[str, int]
    pass_list: tuple[str, ...]
    rows_changed: int


def redact_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], RedactStats]:
    """Return redacted copies of ``rows`` (never dropped) with the per-class counts."""
    scan = scan_rows(rows, max_issues=0)
    redacted, rows_changed, _removed = apply_pii_policy(rows, PII_POLICY)
    return redacted, RedactStats(
        counts=scan.counts_by_code, pass_list=scan.pass_list, rows_changed=rows_changed
    )


def redact_records(records: Sequence[TraceRecord]) -> tuple[list[TraceRecord], RedactStats]:
    """Redact every string a training row is built from, before any of it is cut to length.

    A record's answer is its :func:`effective_response` -- its tool calls when
    it made some -- so the redacted copy carries that as its ``response`` and no
    tool calls: what is trained on is exactly what was redacted. A cut made
    after redaction can only split a placeholder, never leave part of an
    identifier behind. The system prompt an agent repeats on every call is
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
