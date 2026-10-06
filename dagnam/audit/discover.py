"""Workload discovery: group traces by their key, classify their outputs, summarize.

Deterministic and pure: no embeddings, no I/O, no randomness. The same
records in any order give the same workloads in the same order.

Discovery reads its records once, as a stream, and keeps per record only its
time and position: what a workload needs from its calls -- the first
:data:`STRUCTURE_SAMPLE` responses in time order, the digests of its outputs,
its latencies, sums -- is accumulated as the records go by, so an export larger
than memory can be discovered.
"""

from __future__ import annotations

from array import array
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import heapq
import math
import statistics
from typing import Any, Literal

from dagnam_contracts.hygiene import PII_CODES, apply_pii_policy

from dagnam.audit.normalize import UNSTRUCTURED, normalize_template, template_digest
from dagnam.audit.readers.messages import effective_response
from dagnam.audit.record import TraceRecord
from dagnam.audit.structure import (
    StructureClass,
    classify_outputs,
    normalize_response,
    output_shape,
)
from dagnam.audit.thresholds import DAYS_PER_MONTH, EXCERPT_CHARS, STRUCTURE_SAMPLE, WINDOW_DAYS

_SECONDS_PER_DAY = 86_400
_REDACT_EVERYTHING: dict[str, Literal["redact"]] = dict.fromkeys(PII_CODES, "redact")
_QUANTILE_CUTS = 20  # p50 and p95 are the 10th and 19th of 19 cut points
_OUTPUT_DIGEST_BYTES = 8
_EXCERPT_TEMPLATES = 64
"""Distinct templates per workload whose excerpt is kept."""


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """One model id's share of a workload, as the export spelled it.

    ``calls`` counts the calls that carried token counts, the ones its tokens
    price: a stream without usage adds none, so pricing extrapolates the priced
    calls to it instead of spreading their tokens over it.
    """

    model: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int = 0


@dataclass(frozen=True, slots=True)
class Workload:
    """One discovered workload with the statistics the economics and the report need.

    ``record_indices`` are positions in the sequence given to
    :func:`discover_workloads`, in time order. ``response_mode`` is
    ``"tool_call"`` when most calls answered with a tool call, whose first
    call's name and arguments are then what the audit trains and scores on.
    ``usage_by_model`` keeps the tokens per model spelling so a price table can
    price each one. ``name`` is the workload's own name when the export gave
    one (a ``workload_hint`` or a registered prompt's name), which then keys it.
    ``media_calls`` counts calls whose prompt carried an image, audio or file
    part. ``first_ts`` / ``last_ts`` bound its calls. ``answerless_calls`` counts the
    calls whose reply was nothing but the model's reasoning (``reasoning_only``):
    they are among ``calls`` and in the cost and token counts, as the customer paid
    for them, but give no training row and take no part in the structure class or
    the output counts; ``answerless_spend_share`` is their share of the spend (by the
    export's own cost when it has one, else by completion tokens, else by calls).
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
    response_mode: Literal["text", "tool_call"] = "text"
    usage_by_model: tuple[ModelUsage, ...] = ()
    name: str | None = None
    media_calls: int = 0
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    answerless_calls: int = 0
    answerless_spend_share: float = 0.0

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
            "response_mode": self.response_mode,
            "name": self.name,
            "media_calls": self.media_calls,
        }


def template_excerpt(system: str) -> str:
    """The first :data:`EXCERPT_CHARS` of a system prompt, PII-redacted and then normalized.

    Redaction runs on the raw prompt, before normalization turns the digits of
    a card or a key into ``<NUM>`` and leaves fragments no detector recognises.
    """
    (row,), _, _ = apply_pii_policy([{"template": system}], _REDACT_EVERYTHING)
    return normalize_template(str(row["template"]))[:EXCERPT_CHARS]


def by_spend(workloads: Iterable[Workload]) -> tuple[Workload, ...]:
    """Most monthly spend first (an unknown cost last), then most calls, then id."""
    ordered = sorted(workloads, key=lambda w: w.id)  # stable sorts: the id breaks the ties below
    ordered.sort(
        key=lambda w: (
            w.cost_usd_month if w.cost_usd_month is not None else -math.inf,
            w.calls,
        ),
        reverse=True,
    )
    return tuple(ordered)


def _entropy_bits(counts: Counter[bytes]) -> float:
    total = sum(counts.values())
    return -sum(n / total * math.log2(n / total) for n in counts.values())


def _quantiles(values: Sequence[float]) -> tuple[float, float]:
    if len(values) < 2:
        return values[0], values[0]
    cuts = statistics.quantiles(values, n=_QUANTILE_CUTS)
    return cuts[_QUANTILE_CUTS // 2 - 1], cuts[-1]


type _Moment = tuple[float, int]
"""A record's place in time order: ``(epoch seconds, position in the stream)``."""


@dataclass(slots=True)
class _Group:
    """What one workload keeps of its records while the stream goes by."""

    key: str
    name: str | None
    stamps: array[float] = field(default_factory=lambda: array("d"))
    positions: array[int] = field(default_factory=lambda: array("q"))
    first: datetime | None = None
    last: datetime | None = None
    # A max-heap on the moment, bounded at STRUCTURE_SAMPLE: the earliest responses.
    sample: list[tuple[float, int, str, tuple[dict[str, Any], ...]]] = field(
        default_factory=list[tuple[float, int, str, tuple[dict[str, Any], ...]]]
    )
    outputs: Counter[bytes] = field(default_factory=Counter[bytes])
    templates: Counter[str] = field(default_factory=Counter[str])
    earliest: dict[str, _Moment] = field(default_factory=dict[str, _Moment])
    # template -> (digest, excerpt) of its prompt with the smallest digest; never the prompt.
    excerpts: dict[str, tuple[bytes, str]] = field(default_factory=dict[str, tuple[bytes, str]])
    latencies: array[float] = field(default_factory=lambda: array("d"))
    usage: dict[str, list[int]] = field(default_factory=dict[str, list[int]])
    priced_sum: float = 0.0
    priced: int = 0
    tool_calls: int = 0
    media: int = 0
    answerless: int = 0
    answerless_cost: float = 0.0
    answerless_completion: int = 0

    def add(self, index: int, record: TraceRecord, template: str) -> None:
        stamp = record.ts.timestamp()
        moment = (stamp, index)
        self.stamps.append(stamp)
        self.positions.append(index)
        self.first = record.ts if self.first is None else min(self.first, record.ts)
        self.last = record.ts if self.last is None else max(self.last, record.ts)
        if record.reasoning_only:
            # A call, and a billed one; but it has no answer to classify or to count.
            self.answerless += 1
            self.answerless_completion += record.completion_tokens
            self.answerless_cost += record.cost_usd or 0.0
        else:
            heapq.heappush(
                self.sample, (-stamp, -index, record.response, record.response_tool_calls)
            )
            if len(self.sample) > STRUCTURE_SAMPLE:
                heapq.heappop(self.sample)  # the latest of the kept responses
            effective = effective_response(record.response, record.response_tool_calls)
            answer = normalize_response(effective)
            digest = hashlib.blake2b(answer.encode("utf-8"), digest_size=_OUTPUT_DIGEST_BYTES)
            self.outputs[digest.digest()] += 1
        self.templates[template] += 1
        if moment < self.earliest.get(template, (math.inf, 0)):
            self.earliest[template] = moment
        # ponytail: the excerpts of the first 64 distinct templates only, so a per-call
        # prompt under one name stays bounded; the modal one is almost always among
        # them, else the excerpt is the earliest kept.
        if template in self.excerpts or len(self.excerpts) < _EXCERPT_TEMPLATES:
            self._keep_excerpt(template, record.system or "")
        self.latencies.append(record.latency_ms)
        tokens = self.usage.setdefault(record.model, [0, 0, 0, 0])
        tokens[0] += record.prompt_tokens + record.completion_tokens > 0
        tokens[1] += record.prompt_tokens
        tokens[2] += record.completion_tokens
        tokens[3] += record.cached_prompt_tokens
        if record.cost_usd is not None:
            self.priced_sum += record.cost_usd
            self.priced += 1
        self.tool_calls += bool(record.response_tool_calls)
        self.media += record.has_media

    def _keep_excerpt(self, template: str, system: str) -> None:
        """Keep the excerpt of the template's prompt whose digest is smallest, not the prompt.

        A raw prompt per template was the export held again for per-call
        prompts. The smallest digest is the same whatever the records'
        order, and a stream of distinct prompts beats it only about ``ln n``
        times, so redaction runs a handful of times per template.
        """
        digest = hashlib.blake2b(system.encode("utf-8"), digest_size=_OUTPUT_DIGEST_BYTES).digest()
        kept = self.excerpts.get(template)
        if kept is None or digest < kept[0]:
            self.excerpts[template] = (digest, template_excerpt(system))

    def summarize(self, days: float, sample_rate: float) -> Workload:
        calls = len(self.positions)
        order = sorted(range(calls), key=lambda j: (self.stamps[j], self.positions[j]))
        sample = sorted(self.sample, reverse=True)  # back to time order
        modal = max(self.templates, key=lambda t: (self.templates[t], _later(self.earliest[t])))
        excerpt_of = (
            modal if modal in self.excerpts else min(self.excerpts, key=lambda t: self.earliest[t])
        )
        p50, p95 = _quantiles(self.latencies)
        usage = tuple(
            ModelUsage(model, calls=n, prompt_tokens=p, completion_tokens=c, cached_prompt_tokens=k)
            for model, (n, p, c, k) in sorted(self.usage.items())
        )
        scale = DAYS_PER_MONTH / days / sample_rate
        completion = sum(u.completion_tokens for u in usage)
        if self.priced_sum:
            share = self.answerless_cost / self.priced_sum
        elif completion:
            share = self.answerless_completion / completion
        else:
            share = self.answerless / calls
        return Workload(
            id=self.key,
            template_hash=modal,
            template_excerpt=self.excerpts[excerpt_of][1],
            structure_class=classify_outputs([s[2] for s in sample], [s[3] for s in sample]),
            confidence="low" if self.key.startswith(f"{UNSTRUCTURED}-") else "high",
            calls=calls,
            calls_per_day=calls / days / sample_rate,
            prompt_tokens=sum(u.prompt_tokens for u in usage),
            completion_tokens=sum(u.completion_tokens for u in usage),
            # The mean priced call times every call, so a partially priced export
            # still extrapolates; scaled from the export span to a month.
            cost_usd_month=self.priced_sum / self.priced * calls * scale if self.priced else None,
            cost_source="export" if self.priced else "unknown",
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            distinct_outputs=len(self.outputs),
            entropy_bits=_entropy_bits(self.outputs),
            stability=self.templates[modal] / calls,
            sample_size=len(sample),
            models=tuple(model for model, _ in sorted(self.usage.items())),
            record_indices=tuple(self.positions[j] for j in order),
            response_mode="tool_call" if 2 * self.tool_calls > calls else "text",
            usage_by_model=usage,
            name=self.name,
            media_calls=self.media,
            first_ts=self.first,
            last_ts=self.last,
            answerless_calls=self.answerless,
            answerless_spend_share=share if self.answerless else 0.0,
        )


def _later(moment: _Moment) -> _Moment:
    """Sorts an earlier moment higher, so a tie in count goes to the template seen first."""
    return (-moment[0], -moment[1])


def _key(record: TraceRecord, template: str) -> tuple[str, str | None]:
    """The workload a record belongs to, and the name that keys it when it has one.

    In order: the customer's own name for the
    step (``workload_hint``, a registered prompt's name); else the system
    prompt's template joined with the request's schema or forced tool; else, with
    neither, the shape of its answer.
    """
    if record.workload_hint:
        return template_digest(f"workload_hint:{record.workload_hint}"), record.workload_hint
    if record.signature:
        return template_digest(f"{template}\n{record.signature}"), None
    if template:
        return template_digest(template), None
    return f"{UNSTRUCTURED}-{output_shape(record.response, record.response_tool_calls)}", None


def discover_workloads(
    records: Iterable[TraceRecord],
    *,
    window_days: int = WINDOW_DAYS,
    sample_rate: float = 1.0,
) -> tuple[Workload, ...]:
    """Group ``records`` (read once, in any order) into workloads, most monthly spend first.

    The key is :func:`_key`'s: a named step, else the template (with the
    request's signature), else an ``"unstructured-<shape>"`` bucket at low
    confidence. The structure class is decided on the group's first
    :data:`STRUCTURE_SAMPLE` responses in time order. Rates are per day over
    the longer of ``window_days`` and the export's own timestamp span, so a
    partial export is never mistaken for a quiet month, and divided by
    ``sample_rate`` when the export holds only that share of the traffic;
    ``cost_usd_month`` is the export's own cost per call, extrapolated to every
    call and scaled to :data:`DAYS_PER_MONTH`. Ties in spend break by calls,
    then by id.
    """
    if not 0 < sample_rate <= 1:
        raise ValueError(f"sample_rate must be in (0, 1]; got {sample_rate}")
    # ponytail: one _Group per key, a few KB plus its first responses: unnamed per-call
    # prompts (RAG in the system prompt) still make one per call, until a later change folds them.
    groups: dict[str, _Group] = {}
    for index, record in enumerate(records):
        template = normalize_template(record.system or "")
        key, name = _key(record, template)
        group = groups.get(key)
        if group is None:
            group = groups[key] = _Group(key, name)
        group.add(index, record, template_digest(template))
    if not groups:
        return ()
    first = min(g.first or datetime.max.replace(tzinfo=UTC) for g in groups.values())
    last = max(g.last or first for g in groups.values())
    days = max(float(window_days), (last - first).total_seconds() / _SECONDS_PER_DAY)
    return by_spend(group.summarize(days, sample_rate) for group in groups.values())
