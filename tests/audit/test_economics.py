"""Break-even economics: the verdict rules of the design's economics table."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import date, timedelta

from dagnam_contracts.audit.serving import SERVING_RATES
from hypothesis import given, settings, strategies as st
import pytest
from tests.audit._records import T0, make_record, make_workload

from dagnam.audit import TraceRecord
from dagnam.audit.candidates import CANDIDATES
from dagnam.audit.discover import ModelUsage, Workload, discover_workloads
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
from dagnam.audit.prices import PriceRow, PriceTable
from dagnam.audit.structure import StructureClass
from dagnam.audit.thresholds import (
    DAYS_PER_MONTH,
    MAINTENANCE_USD_MONTH,
    MAX_MEDIA_SHARE,
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
        StructureClass.SHORT_SPAN: "gpu-small-llm",
        StructureClass.JSON_OBJECT: "gpu-small-llm",
    }
    assert StructureClass.FREE_TEXT not in STUDENT_KIND


def test_every_student_kind_is_the_rate_of_the_candidate_the_run_trains() -> None:
    # The scan prices the same student the run trains and its report prices.
    for cls, kind in STUDENT_KIND.items():
        assert {spec.serving_rate_key for spec in CANDIDATES[cls]} - {None} == {kind}, cls


def test_student_cost_per_kind() -> None:
    w = make_workload(calls_per_day=1_000, completion_per_call=40)
    calls_month = 1_000 * DAYS_PER_MONTH

    cpu = SERVING_RATES["cpu-classifier"]["usd_per_1k_requests"]
    assert student_cost_usd_month(w, "cpu-classifier") == pytest.approx(calls_month / 1_000 * cpu)
    gpu = SERVING_RATES["gpu-small-llm"]["usd_per_m_output_tokens"]
    assert student_cost_usd_month(w, "gpu-small-llm") == pytest.approx(calls_month * 40 / 1e6 * gpu)


def test_short_span_is_priced_as_the_gpu_student_it_trains() -> None:
    # 1M calls/day of 6-token spans on $1,000/month read marginal (8.4x) at the CPU
    # rate; the sft_small it trains serves 180M output tokens a month at the GPU rate
    # ($351 at the old $1.95/M, $734 at the A10G's $4.08/M), so it is not worth it.
    w = make_workload(
        calls_per_day=1_000_000,
        cost_month=1_000.0,
        cls=StructureClass.SHORT_SPAN,
        n=30_000_000,
        completion_per_call=6,
    )
    verdict = replaceability(w)
    assert verdict.status == "not_worth_it"
    gpu = SERVING_RATES["gpu-small-llm"]["usd_per_m_output_tokens"]
    assert verdict.ratio == pytest.approx(1_000 / (180 * gpu + MAINTENANCE_USD_MONTH))
    assert "gpu-small-llm" in verdict.reason


def test_a_token_priced_student_with_no_completion_tokens_has_an_unknown_cost() -> None:
    # An export with a cost but no token usage priced the GPU student at $0.00.
    for cls in (StructureClass.JSON_OBJECT, StructureClass.SHORT_SPAN):
        w = make_workload(calls_per_day=1_000, cost_month=3_000.0, cls=cls, completion_per_call=0)
        verdict = replaceability(w)
        assert verdict.status == "unknown_cost", cls
        assert (verdict.ratio, verdict.savings_usd_month) == (None, None)
        assert "completion token" in verdict.reason
    labels = make_workload(calls_per_day=1_000, cost_month=3_000.0, completion_per_call=0)
    assert replaceability(labels).status == "candidate"  # priced per request, not per token


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
        make_workload(cost_month=None, models=("other", "unknown")),  # no row for either
    ],
)
def test_price_workload_leaves_what_it_cannot_price(w: Workload) -> None:
    assert price_workload(w, TABLE) is w


def test_two_spellings_of_one_model_are_each_priced() -> None:
    # An alias mixed with its snapshot was unknown_cost, "no price-table row for
    # gpt-4o-mini, gpt-4o-mini-2024-07-18" -- although both price on their own.
    table = PriceTable.load(None)
    models = ("gpt-4o-mini", "gpt-4o-mini-2024-07-18")
    w = make_workload(calls_per_day=100, cost_month=None, n=3_000, models=models)

    priced = price_workload(w, table)

    whole = table.cost("gpt-4o-mini", w.prompt_tokens, w.completion_tokens)
    assert priced.cost_source == "price_table"
    assert priced.cost_usd_month == pytest.approx(whole)  # a 30-day window
    assert replaceability(priced).status != "unknown_cost"


def test_calls_on_an_unpriced_model_are_extrapolated_from_the_priced_ones() -> None:
    # One row that fell back to "unknown" used to make the whole workload unknown_cost.
    w = make_workload(calls_per_day=100, cost_month=None, n=3_000, models=("m", "unknown"))

    priced = price_workload(w, TABLE)

    assert priced.cost_usd_month == pytest.approx((3_000 * 100 * 1.0 + 3_000 * 2 * 2.0) / 1e6)


def test_cached_prompt_tokens_are_priced_at_the_cached_rate() -> None:
    # 10 gpt-5.4 calls of 10,000 prompt tokens, 9,900 cached: $0.25/month before, $0.027 now.
    usage = ModelUsage("gpt-5.4", 10, 100_000, 10, cached_prompt_tokens=99_000)
    w = replace(
        make_workload(calls_per_day=10 / 30, cost_month=None, n=10, models=("gpt-5.4",)),
        usage_by_model=(usage,),
    )

    priced = price_workload(w, PriceTable.load(None))

    assert priced.cost_usd_month == pytest.approx(0.02740)


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


def test_a_workload_whose_calls_carry_images_is_not_audited() -> None:
    # Receipts classified from the image, with a constant text prompt: the run
    # would train a text classifier on 1,200 identical inputs.
    w = replace(make_workload(calls_per_day=5_000, cost_month=4_000.0), media_calls=1_000)
    verdict = replaceability(w)
    assert verdict.status == "not_audited"
    assert verdict.reason.startswith("multimodal input")
    assert (verdict.ratio, verdict.savings_usd_month) == (None, None)
    few = replace(w, media_calls=int(w.calls * MAX_MEDIA_SHARE))
    assert replaceability(few).status == "candidate"  # at the share, still text


def test_no_usage_and_no_cost_is_an_unknown_cost_never_free() -> None:
    # A stream without include_usage has 0 tokens; the table priced the teacher
    # at $0.00 and the verdict read "not_worth_it: teacher $0.00/month".
    w = make_workload(
        calls_per_day=100, cost_month=None, n=3_000, prompt_per_call=0, completion_per_call=0
    )
    priced = price_workload(w, PriceTable.load(None))
    assert priced.cost_usd_month is None
    verdict = replaceability(priced)
    assert verdict.status == "unknown_cost"
    assert "neither a cost nor token counts" in verdict.reason
    # A model whose calls carried no usage is extrapolated from the ones that did.
    mixed = replace(
        make_workload(calls_per_day=100, cost_month=None, n=3_000, models=("m",)),
        usage_by_model=(
            ModelUsage("m", calls=1_500, prompt_tokens=150_000, completion_tokens=3_000),
            ModelUsage("m-2024-01-01", calls=1_500, prompt_tokens=0, completion_tokens=0),
        ),
    )
    assert price_workload(mixed, TABLE).cost_usd_month == pytest.approx(
        2 * (150_000 * 1.0 + 3_000 * 2.0) / 1e6
    )


def test_a_model_s_calls_without_usage_are_extrapolated_from_its_calls_with_it() -> None:
    # half of one model's calls streamed without usage priced the teacher at half
    # ($0.3012 -> $0.1506 a month): the tokens of 1,000 calls were spread over 2,000.
    def records(with_usage: Callable[[int], bool]) -> list[TraceRecord]:
        return [
            make_record(
                ts=T0 + timedelta(minutes=i),
                trace_id=f"c{i}",
                prompt_tokens=1_000 if with_usage(i) else 0,
                completion_tokens=1 if with_usage(i) else 0,
                cost_usd=None,
            )
            for i in range(2_000)
        ]

    def monthly(with_usage: Callable[[int], bool]) -> float | None:
        (workload,) = discover_workloads(records(with_usage), window_days=30)
        return price_workload(workload, PriceTable.load(None)).cost_usd_month

    everything = monthly(lambda i: True)
    assert everything == pytest.approx(0.3012, abs=1e-4)
    assert monthly(lambda i: i % 2 == 0) == pytest.approx(everything)
