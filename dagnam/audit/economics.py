"""Break-even economics: is a workload worth replacing with a small model?

Every rule here is the design's economics table, which the contract owns
(``dagnam_contracts.audit.verdict``) so the platform reaches the same verdict
from the same numbers: ``ratio = teacher $/month ÷ (student $/month +
maintenance)``, below :data:`RATIO_NOT_WORTH_IT` is ``not_worth_it``, at or
above :data:`RATIO_CANDIDATE` is ``candidate``, between is ``marginal``. Free
text is never audited, too few traces or an unknown cost are reported as such.
This module applies them to a :class:`~dagnam.audit.discover.Workload`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from dagnam_contracts.audit.serving import StudentKind, serving_cost_usd_month
from dagnam_contracts.audit.verdict import (
    CustomerVerdict,
    VerdictStatus,
    customer_verdict,
    ratio_status,
)

from dagnam.audit.discover import Workload
from dagnam.audit.prices import PriceTable
from dagnam.audit.structure import StructureClass
from dagnam.audit.thresholds import (
    DAYS_PER_MONTH,
    MAINTENANCE_USD_MONTH,
    MIN_TRACES_PER_WORKLOAD,
)

STUDENT_KIND: Mapping[StructureClass, StudentKind] = {
    StructureClass.ENUM_LABEL: "cpu-classifier",
    StructureClass.SHORT_SPAN: "cpu-classifier",
    StructureClass.JSON_OBJECT: "gpu-small-llm",
}
"""The student that serves each structure class; a class absent here is not audited."""


@dataclass(frozen=True, slots=True)
class Verdict:
    """The economics decision for one workload; ``ratio``/``savings`` only when computed."""

    status: VerdictStatus
    ratio: float | None
    savings_usd_month: float | None
    reason: str


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
    status = ratio_status(ratio)
    reason = (
        f"teacher ${w.cost_usd_month:,.2f}/month vs {kind} ${student:,.2f}/month"
        f" + ${MAINTENANCE_USD_MONTH:,.0f} maintenance = {ratio:.1f}x"
    )
    return Verdict(status, ratio, savings, reason)


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


__all__ = [
    "STUDENT_KIND",
    "CustomerVerdict",
    "Verdict",
    "VerdictStatus",
    "customer_verdict",
    "price_workload",
    "ratio_status",
    "replaceability",
    "serving_cost_usd_month",
    "student_cost_usd_month",
]
