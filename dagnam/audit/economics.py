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

from dagnam_contracts.audit.serving import SERVING_RATES, StudentKind, serving_cost_usd_month
from dagnam_contracts.audit.verdict import (
    CustomerVerdict,
    VerdictStatus,
    customer_verdict,
    ratio_status,
)

from dagnam.audit.candidates import CANDIDATES
from dagnam.audit.discover import Workload
from dagnam.audit.prices import PriceTable
from dagnam.audit.structure import StructureClass
from dagnam.audit.thresholds import (
    DAYS_PER_MONTH,
    MAINTENANCE_USD_MONTH,
    MAX_MEDIA_SHARE,
    MIN_TRACES_PER_WORKLOAD,
)

STUDENT_KIND: Mapping[StructureClass, StudentKind] = {
    cls: spec.serving_rate_key
    for cls, specs in CANDIDATES.items()
    for spec in specs
    if spec.serving_rate_key is not None
}
"""The student that serves each structure class; a class absent here is not audited.

Read off the candidate registry, so the scan prices the very student the run
trains (``short_span`` trains ``sft_small``, a GPU LLM) and the two never drift.
"""


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
    if w.media_calls > MAX_MEDIA_SHARE * w.calls:
        # N9: the answers depend on parts the text a student is trained on does not hold.
        reason = (
            f"multimodal input: {w.media_calls / w.calls:.0%} of calls carry image, audio"
            " or file parts a text student cannot see"
        )
        return Verdict("not_audited", None, None, reason)
    kind = STUDENT_KIND.get(w.structure_class)
    if kind is None:
        return Verdict("not_audited", None, None, "free-text output is not a small-model job")
    if w.calls < MIN_TRACES_PER_WORKLOAD:
        reason = f"{w.calls:,} traces in the window; {MIN_TRACES_PER_WORKLOAD:,} needed"
        return Verdict("too_few_samples", None, None, reason)
    if w.cost_usd_month is None and w.prompt_tokens + w.completion_tokens == 0:
        reason = "the export carries neither a cost nor token counts (a stream without usage?)"
        return Verdict("unknown_cost", None, None, reason)
    if w.cost_usd_month is None:
        reason = f"no cost in the export and no price-table row for {', '.join(w.models)}"
        return Verdict("unknown_cost", None, None, reason)
    if w.completion_tokens == 0 and "usd_per_m_output_tokens" in SERVING_RATES[kind]:
        # A student billed per output token would price at $0.00 and win every ratio.
        reason = f"the export has no completion token counts, so the {kind} student is unpriced"
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
    """Fill a missing ``cost_usd_month`` from ``table``, pricing each model spelling on its own.

    Calls on a model the table does not list (a row that fell back to
    ``"unknown"``), or that carried no token counts -- a whole model's or some
    of its calls (m1) -- are extrapolated from the priced ones, the way the
    export's own partial costs are; with no priced call at all the cost stays
    unknown -- never $0 for a stream without usage (N10).
    """
    if w.cost_usd_month is not None:
        return w
    priced = [
        (usage.calls, cost)
        for usage in w.usage_by_model
        if usage.prompt_tokens + usage.completion_tokens > 0
        and (
            cost := table.cost(
                usage.model,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.cached_prompt_tokens,
            )
        )
        is not None
    ]
    if not priced:
        return w
    window_cost = sum(cost for _, cost in priced) * w.calls / sum(calls for calls, _ in priced)
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
