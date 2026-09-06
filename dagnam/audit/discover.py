"""Workload discovery: group traces by template hash, classify their outputs, summarize.

Deterministic and pure: no embeddings, no I/O, no randomness. The same
records in any order give the same workloads in the same order.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import math
import statistics
from typing import Any, Literal

from dagnam_contracts.hygiene import PII_CODES, apply_pii_policy

from dagnam.audit.normalize import UNSTRUCTURED, normalize_template, template_hash
from dagnam.audit.record import TraceRecord
from dagnam.audit.structure import (
    StructureClass,
    classify_outputs,
    effective_response,
    normalize_response,
)
from dagnam.audit.thresholds import DAYS_PER_MONTH, EXCERPT_CHARS, STRUCTURE_SAMPLE, WINDOW_DAYS

_SECONDS_PER_DAY = 86_400
_REDACT_EVERYTHING: dict[str, Literal["redact"]] = dict.fromkeys(PII_CODES, "redact")
_QUANTILE_CUTS = 20  # p50 and p95 are the 10th and 19th of 19 cut points


@dataclass(frozen=True, slots=True)
class Workload:
    """One discovered workload with the statistics the economics and the report need.

    ``record_indices`` are positions in the sequence given to
    :func:`discover_workloads`, in time order.
    """

    id: str
    template_hash: str
    template_excerpt: str
    structure_class: StructureClass
    confidence: Literal["high", "low"]
    calls: int
    calls_per_day: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd_month: float | None
    cost_source: Literal["export", "price_table", "unknown"]
    latency_p50_ms: float
    latency_p95_ms: float
    distinct_outputs: int
    entropy_bits: float
    stability: float
    sample_size: int
    models: tuple[str, ...]
    record_indices: tuple[int, ...]

    def to_json(self) -> dict[str, Any]:
        """The scan report's ``Workload`` object (its discovery-owned fields, in contract order)."""
        return {
            "id": self.id,
            "template_hash": self.template_hash,
            "template_excerpt": self.template_excerpt,
            "structure_class": self.structure_class.value,
            "confidence": self.confidence,
            "calls": self.calls,
            "calls_per_day": self.calls_per_day,
            "tokens": {"prompt": self.prompt_tokens, "completion": self.completion_tokens},
            "cost_usd_month": self.cost_usd_month,
            "cost_source": self.cost_source,
            "latency_ms": {"p50": self.latency_p50_ms, "p95": self.latency_p95_ms},
            "distinct_outputs": self.distinct_outputs,
            "entropy": self.entropy_bits,
            "models": list(self.models),
        }


def template_excerpt(template: str) -> str:
    """The first :data:`EXCERPT_CHARS` of a normalized template after PII redaction.

    Normalization already masks emails and numbers; the redaction pass is the
    contract's second line of defence before any prompt text reaches a report.
    """
    (row,), _, _ = apply_pii_policy([{"template": template}], _REDACT_EVERYTHING)
    return str(row["template"])[:EXCERPT_CHARS]


def _entropy_bits(counts: Counter[str]) -> float:
    total = sum(counts.values())
    return -sum(n / total * math.log2(n / total) for n in counts.values())


def _quantiles(values: Sequence[float]) -> tuple[float, float]:
    if len(values) < 2:
        return values[0], values[0]
    cuts = statistics.quantiles(values, n=_QUANTILE_CUTS)
    return cuts[_QUANTILE_CUTS // 2 - 1], cuts[-1]


def _summarize(
    records: Sequence[TraceRecord], indices: Sequence[int], key: str, days: float
) -> Workload:
    rows = [records[i] for i in indices]
    sample = rows[:STRUCTURE_SAMPLE]
    structure_class = classify_outputs(
        [r.response for r in sample], [r.response_tool_calls for r in sample]
    )
    outputs = Counter(
        normalize_response(effective_response(r.response, r.response_tool_calls)) for r in rows
    )
    templates = Counter(normalize_template(r.system or "") for r in rows)
    modal_template, modal_count = templates.most_common(1)[0]
    priced = [r.cost_usd for r in rows if r.cost_usd is not None]
    p50, p95 = _quantiles([r.latency_ms for r in rows])
    return Workload(
        id=key,
        template_hash=key,
        template_excerpt=template_excerpt(modal_template),
        structure_class=structure_class,
        confidence="low" if key == UNSTRUCTURED else "high",
        calls=len(rows),
        calls_per_day=len(rows) / days,
        prompt_tokens=sum(r.prompt_tokens for r in rows),
        completion_tokens=sum(r.completion_tokens for r in rows),
        # The mean priced call times every call, so a partially priced export
        # still extrapolates; scaled from the export span to a month.
        cost_usd_month=(
            statistics.fmean(priced) * len(rows) * (DAYS_PER_MONTH / days) if priced else None
        ),
        cost_source="export" if priced else "unknown",
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        distinct_outputs=len(outputs),
        entropy_bits=_entropy_bits(outputs),
        stability=modal_count / len(rows),
        sample_size=len(sample),
        models=tuple(sorted({r.model for r in rows})),
        record_indices=tuple(indices),
    )


def discover_workloads(
    records: Sequence[TraceRecord], *, window_days: int = WINDOW_DAYS
) -> tuple[Workload, ...]:
    """Group ``records`` into workloads, most monthly spend first.

    Workload key is the system prompt's template hash (``"unstructured"`` for
    traces without one); the structure class is decided on the group's first
    :data:`STRUCTURE_SAMPLE` responses in time order. Rates are per day over
    the longer of ``window_days`` and the export's own timestamp span, so a
    partial export is never mistaken for a quiet month; ``cost_usd_month`` is
    the export's own cost per call, extrapolated to every call and scaled to
    :data:`DAYS_PER_MONTH`. Ties in spend break by calls, then by hash.
    """
    if not records:
        return ()
    order = sorted(range(len(records)), key=lambda i: (records[i].ts, i))
    span = (records[order[-1]].ts - records[order[0]].ts).total_seconds() / _SECONDS_PER_DAY
    days = max(float(window_days), span)
    groups: dict[str, list[int]] = {}
    for i in order:
        groups.setdefault(template_hash(records[i].system), []).append(i)
    workloads = [_summarize(records, indices, key, days) for key, indices in groups.items()]
    workloads.sort(key=lambda w: w.template_hash)  # stable sorts: hash breaks the ties below
    workloads.sort(
        key=lambda w: (
            w.cost_usd_month if w.cost_usd_month is not None else -math.inf,
            w.calls,
        ),
        reverse=True,
    )
    return tuple(workloads)
