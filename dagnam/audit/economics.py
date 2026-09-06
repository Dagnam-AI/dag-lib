"""Break-even economics: is a workload worth replacing with a small model?

Every rule here is the design's economics table (:mod:`dagnam.audit.thresholds`):
``ratio = teacher $/month ÷ (student $/month + maintenance)``, below
:data:`RATIO_NOT_WORTH_IT` is ``not_worth_it``, at or above
:data:`RATIO_CANDIDATE` is ``candidate``, between is ``marginal``. Free text is
never audited, too few traces or an unknown cost are reported as such.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

from dagnam.audit.candidates import StudentKind
from dagnam.audit.discover import Workload
from dagnam.audit.prices import SERVING_RATES, PriceTable
from dagnam.audit.structure import StructureClass
from dagnam.audit.thresholds import (
    DAYS_PER_MONTH,
    MAINTENANCE_USD_MONTH,
    MIN_TRACES_PER_WORKLOAD,
    RATIO_CANDIDATE,
    RATIO_NOT_WORTH_IT,
)

VerdictStatus = Literal[
    "candidate", "marginal", "not_worth_it", "not_audited", "too_few_samples", "unknown_cost"
]
CustomerVerdict = Literal["REPLACE", "NOT YET", "KEEP"]

STUDENT_KIND: Mapping[StructureClass, StudentKind] = {
    StructureClass.ENUM_LABEL: "cpu-classifier",
    StructureClass.SHORT_SPAN: "cpu-classifier",
    StructureClass.JSON_OBJECT: "gpu-small-llm",
}
"""The student that serves each structure class; a class absent here is not audited."""

_AUDITED: frozenset[VerdictStatus] = frozenset({"candidate", "marginal"})


@dataclass(frozen=True, slots=True)
class Verdict:
    """The economics decision for one workload; ``ratio``/``savings`` only when computed."""

    status: VerdictStatus
    ratio: float | None
    savings_usd_month: float | None
    reason: str


def serving_cost_usd_month(
    kind: StudentKind, *, calls_per_day: float, completion_tokens: int, calls: int
) -> float:
    """Monthly serving cost of a student at this volume, from :data:`SERVING_RATES`."""
    calls_month = calls_per_day * DAYS_PER_MONTH
    if kind == "cpu-classifier":
        return calls_month / 1_000 * SERVING_RATES[kind]["usd_per_1k_requests"]
    output_tokens_month = calls_month * completion_tokens / calls
    return output_tokens_month / 1_000_000 * SERVING_RATES[kind]["usd_per_m_output_tokens"]


def student_cost_usd_month(w: Workload, kind: StudentKind) -> float:
    """:func:`serving_cost_usd_month` at the workload's own volume."""
    return serving_cost_usd_month(
        kind,
        calls_per_day=w.calls_per_day,
        completion_tokens=w.completion_tokens,
        calls=w.calls,
    )


def replaceability(w: Workload) -> Verdict:
    """Apply the design's economics rules to one workload."""
    kind = STUDENT_KIND.get(w.structure_class)
    if kind is None:
        return Verdict("not_audited", None, None, "free-text output is not a small-model job")
    if w.calls < MIN_TRACES_PER_WORKLOAD:
        reason = f"{w.calls:,} traces in the window; {MIN_TRACES_PER_WORKLOAD:,} needed"
        return Verdict("too_few_samples", None, None, reason)
    if w.cost_usd_month is None:
        reason = f"no cost in the export and no price-table row for {', '.join(w.models)}"
        return Verdict("unknown_cost", None, None, reason)
    student = student_cost_usd_month(w, kind)
    ratio = w.cost_usd_month / (student + MAINTENANCE_USD_MONTH)
    # calls_per_month * (teacher_per_call - student_per_call) - maintenance, as monthly sums.
    savings = w.cost_usd_month - student - MAINTENANCE_USD_MONTH
    if ratio < RATIO_NOT_WORTH_IT:
        status: VerdictStatus = "not_worth_it"
    elif ratio < RATIO_CANDIDATE:
        status = "marginal"
    else:
        status = "candidate"
    reason = (
        f"teacher ${w.cost_usd_month:,.2f}/month vs {kind} ${student:,.2f}/month"
        f" + ${MAINTENANCE_USD_MONTH:,.0f} maintenance = {ratio:.1f}x"
    )
    return Verdict(status, ratio, savings, reason)


def customer_verdict(status: VerdictStatus, *, winner: bool) -> CustomerVerdict:
    """The three words the customer reads, a view over ``status`` and whether a candidate won."""
    if status not in _AUDITED:
        return "KEEP"
    return "REPLACE" if winner else "NOT YET"


def price_workload(w: Workload, table: PriceTable) -> Workload:
    """Fill a missing ``cost_usd_month`` from ``table`` when the workload ran on one priced model."""
    # ponytail: a workload spanning several models keeps cost unknown, since the
    # export's token totals cannot be split by model; price per model if that bites.
    if w.cost_usd_month is not None or len(w.models) != 1:
        return w
    window_cost = table.cost(w.models[0], w.prompt_tokens, w.completion_tokens)
    if window_cost is None:
        return w
    window_days = w.calls / w.calls_per_day
    month = window_cost * DAYS_PER_MONTH / window_days
    return replace(w, cost_usd_month=month, cost_source="price_table")
