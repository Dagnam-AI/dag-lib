"""Break-even economics: the verdict rules of the design's economics table."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from hypothesis import given, settings, strategies as st
import pytest
from tests.audit._records import make_workload

from dagnam.audit.discover import Workload
from dagnam.audit.economics import (
    STUDENT_KIND,
    CustomerVerdict,
    Verdict,
    VerdictStatus,
    customer_verdict,
    price_workload,
    replaceability,
    student_cost_usd_month,
)
from dagnam.audit.prices import SERVING_RATES, PriceRow, PriceTable
from dagnam.audit.structure import StructureClass
from dagnam.audit.thresholds import (
    DAYS_PER_MONTH,
    MAINTENANCE_USD_MONTH,
    MIN_TRACES_PER_WORKLOAD,
    RATIO_CANDIDATE,
    RATIO_NOT_WORTH_IT,
)


def test_sub_three_x_is_always_not_worth_it() -> None:
    w = make_workload(
        calls_per_day=200, cost_month=120.0
    )  # student ≈ $ + 50 maintenance → ratio < 3
    verdict = replaceability(w)

    assert verdict.status == "not_worth_it"
    assert verdict.ratio is not None
    assert verdict.ratio < RATIO_NOT_WORTH_IT
    assert "maintenance" in verdict.reason


def test_free_text_is_never_a_candidate() -> None:
    verdict = replaceability(
        make_workload(calls_per_day=10_000, cost_month=5_000.0, cls=StructureClass.FREE_TEXT)
    )

    assert verdict == Verdict(
        "not_audited", None, None, "free-text output is not a small-model job"
    )


def test_too_few_samples() -> None:
    verdict = replaceability(make_workload(calls_per_day=10, cost_month=5_000.0, n=999))

    assert verdict.status == "too_few_samples"
    assert verdict.ratio is None
    assert verdict.savings_usd_month is None
    assert f"{MIN_TRACES_PER_WORKLOAD:,}" in verdict.reason


def test_unknown_cost() -> None:
    w = replace(
        make_workload(calls_per_day=10_000, cost_month=5_000.0),
        cost_usd_month=None,
        cost_source="unknown",
    )

    verdict = replaceability(w)

    assert verdict.status == "unknown_cost"
    assert "gpt-4o-mini" in verdict.reason


def test_candidate_and_marginal_boundaries() -> None:
    def status(cost_month: float) -> str:
        return replaceability(make_workload(calls_per_day=200, cost_month=cost_month)).status

    student = student_cost_usd_month(make_workload(calls_per_day=200), "cpu-classifier")
    base = student + MAINTENANCE_USD_MONTH
    assert status(RATIO_NOT_WORTH_IT * base - 0.01) == "not_worth_it"
    assert status(RATIO_NOT_WORTH_IT * base) == "marginal"
    assert status(RATIO_CANDIDATE * base - 0.01) == "marginal"
    assert status(RATIO_CANDIDATE * base) == "candidate"


def test_savings_is_the_design_formula() -> None:
    w = make_workload(calls_per_day=1_000, cost_month=3_000.0, cls=StructureClass.JSON_OBJECT)
    verdict = replaceability(w)
    calls_month = 1_000 * DAYS_PER_MONTH
    teacher_per_call = 3_000.0 / calls_month
    student_per_call = student_cost_usd_month(w, "gpu-small-llm") / calls_month

    assert verdict.status == "candidate"
    assert verdict.savings_usd_month == pytest.approx(
        calls_month * (teacher_per_call - student_per_call) - MAINTENANCE_USD_MONTH
    )


@given(v1=st.floats(1, 1e6), v2=st.floats(1, 1e6))
@settings(max_examples=200, deadline=None)
def test_savings_monotone_in_volume(v1: float, v2: float) -> None:
    lo, hi = sorted((v1, v2))
    per_call = 0.002  # teacher $/call held fixed

    def savings(calls_per_day: float) -> float:
        cost_month = calls_per_day * DAYS_PER_MONTH * per_call
        result = replaceability(
            make_workload(calls_per_day=calls_per_day, cost_month=cost_month)
        ).savings_usd_month
        assert result is not None
        return result

    assert savings(hi) >= savings(lo) - 1e-9 * max(1.0, abs(savings(lo)))


@given(cost_month=st.floats(0, 1e6), calls_per_day=st.floats(1, 1e6))
@settings(max_examples=200, deadline=None)
def test_ratio_below_three_is_never_audited(cost_month: float, calls_per_day: float) -> None:
    verdict = replaceability(make_workload(calls_per_day=calls_per_day, cost_month=cost_month))

    assert verdict.ratio is not None
    if verdict.ratio < RATIO_NOT_WORTH_IT:
        assert verdict.status == "not_worth_it"
    else:
        assert verdict.status in {"marginal", "candidate"}


def test_student_kind_is_a_registry_over_the_structure_class() -> None:
    assert dict(STUDENT_KIND) == {
        StructureClass.ENUM_LABEL: "cpu-classifier",
        StructureClass.SHORT_SPAN: "cpu-classifier",
        StructureClass.JSON_OBJECT: "gpu-small-llm",
    }
    assert StructureClass.FREE_TEXT not in STUDENT_KIND


def test_student_cost_per_kind() -> None:
    w = make_workload(calls_per_day=1_000, completion_per_call=40)
    calls_month = 1_000 * DAYS_PER_MONTH

    cpu = SERVING_RATES["cpu-classifier"]["usd_per_1k_requests"]
    assert student_cost_usd_month(w, "cpu-classifier") == pytest.approx(calls_month / 1_000 * cpu)
    gpu = SERVING_RATES["gpu-small-llm"]["usd_per_m_output_tokens"]
    assert student_cost_usd_month(w, "gpu-small-llm") == pytest.approx(calls_month * 40 / 1e6 * gpu)


def test_short_span_uses_the_classifier_rate() -> None:
    w = make_workload(calls_per_day=5_000, cost_month=4_000.0, cls=StructureClass.SHORT_SPAN)
    assert replaceability(w).status == "candidate"


TABLE = PriceTable(
    version="t",
    as_of=date(2026, 9, 6),
    rows={"m": PriceRow("m", 1.0, 2.0, None, None)},
)


def test_price_workload_fills_the_gap_from_the_table() -> None:
    w = make_workload(calls_per_day=100, cost_month=None, n=3_000, models=("m",))  # 30-day window

    priced = price_workload(w, TABLE)

    tokens_cost = (3_000 * 100 * 1.0 + 3_000 * 2 * 2.0) / 1e6
    assert priced.cost_source == "price_table"
    assert priced.cost_usd_month == pytest.approx(tokens_cost)  # the window is one month already
    assert replaceability(priced).status != "unknown_cost"


def test_price_workload_scales_the_window_to_a_month() -> None:
    w = make_workload(calls_per_day=200, cost_month=None, n=3_000, models=("m",))  # 15-day window

    assert price_workload(w, TABLE).cost_usd_month == pytest.approx(2 * (300_000 + 12_000) / 1e6)


@pytest.mark.parametrize(
    "w",
    [
        make_workload(cost_month=12.0, models=("m",)),  # the export priced it
        make_workload(cost_month=None, models=("other",)),  # no row
        make_workload(cost_month=None, models=("m", "m2")),  # tokens cannot be split by model
    ],
)
def test_price_workload_leaves_what_it_cannot_price(w: Workload) -> None:
    assert price_workload(w, TABLE) is w


CUSTOMER_TABLE: dict[tuple[VerdictStatus, bool], CustomerVerdict] = {
    ("candidate", True): "REPLACE",
    ("marginal", True): "REPLACE",
    ("candidate", False): "NOT YET",
    ("marginal", False): "NOT YET",
    ("not_worth_it", False): "KEEP",
    ("not_audited", False): "KEEP",
    ("too_few_samples", False): "KEEP",
    ("unknown_cost", False): "KEEP",
}


def test_customer_verdict_table_verbatim() -> None:
    for (status, winner), expected in CUSTOMER_TABLE.items():
        assert customer_verdict(status, winner=winner) == expected, (status, winner)
    assert (
        customer_verdict("not_worth_it", winner=True) == "KEEP"
    )  # a winner never overrides a KEEP
