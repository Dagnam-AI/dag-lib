"""Workload discovery: template hash x structure class, with per-workload statistics."""

from __future__ import annotations

from collections import Counter
from dataclasses import fields
from datetime import timedelta
import json
import math

from hypothesis import given, settings, strategies as st
import pytest
from tests.audit._records import T0, make_record, make_records

from dagnam.audit import TraceRecord
from dagnam.audit.discover import Workload, discover_workloads, template_excerpt
from dagnam.audit.normalize import UNSTRUCTURED, template_hash
from dagnam.audit.structure import StructureClass
from dagnam.audit.thresholds import EXCERPT_CHARS, STRUCTURE_SAMPLE, WINDOW_DAYS

# The four LLM steps of the fixtures' support-ticket flow (fixtures/README.md)
# and the output structure each one produces.
GROUND_TRUTH = {
    "intent": "enum_label",
    "urgency": "enum_label",
    "extract": "json_object",
    "reply": "free_text",
}


def test_discovery_recovers_the_four_dogfood_workloads(
    langfuse_records: list[TraceRecord],
) -> None:
    found = discover_workloads(langfuse_records, window_days=30)

    by_hash = {template_hash(r.system): r.workload_hint or "?" for r in langfuse_records}
    assert len(by_hash) == len(GROUND_TRUTH)  # one template per step, no split templates
    assert {w.template_hash for w in found} == set(by_hash)  # recall = 1, precision = 1
    assert all(w.structure_class.value == GROUND_TRUTH[by_hash[w.template_hash]] for w in found)
    assert all(w.confidence == "high" for w in found)
    assert all(w.id == w.template_hash for w in found)
    for workload in found:
        assert workload.calls == 3
        assert {langfuse_records[i].workload_hint for i in workload.record_indices} == {
            by_hash[workload.template_hash]
        }
        assert workload.calls_per_day == 3 / WINDOW_DAYS
        assert workload.cost_source == "export"
        assert workload.models == ("gpt-4o-mini",)
        assert workload.sample_size == 3
        assert workload.stability == 1.0


def test_empty_input() -> None:
    assert discover_workloads([], window_days=30) == ()


def test_single_record_statistics() -> None:
    record = make_record(latency_ms=420.0, cost_usd=0.002, prompt_tokens=7, completion_tokens=3)
    (workload,) = discover_workloads([record], window_days=30)

    assert workload.calls == 1
    assert workload.record_indices == (0,)
    assert workload.latency_p50_ms == workload.latency_p95_ms == 420.0
    assert workload.calls_per_day == 1 / 30
    assert workload.cost_usd_month == 0.002
    assert (workload.prompt_tokens, workload.completion_tokens) == (7, 3)
    assert workload.distinct_outputs == 1
    assert workload.entropy_bits == 0.0
    assert workload.sample_size == 1
    assert workload.structure_class is StructureClass.ENUM_LABEL


def test_unstructured_bucket_is_low_confidence() -> None:
    records = make_records(4, system=None) + make_records(2, cost_usd=0.5)
    found = discover_workloads(records, window_days=30)

    assert [(w.template_hash, w.confidence) for w in found] == [
        (template_hash("Label the ticket."), "high"),
        (UNSTRUCTURED, "low"),
    ]
    unstructured = found[1]
    assert unstructured.id == UNSTRUCTURED
    assert unstructured.template_excerpt == ""
    assert unstructured.record_indices == (0, 1, 2, 3)
    assert unstructured.stability == 1.0


def test_sorted_by_monthly_spend_then_calls_then_hash() -> None:
    cheap = make_records(5, system="A", cost_usd=0.001)
    dear = make_records(2, system="B", cost_usd=0.5)
    unknown_many = make_records(9, system="C", cost_usd=None)
    unknown_few = make_records(3, system="D", cost_usd=None)
    found = discover_workloads(unknown_few + cheap + unknown_many + dear, window_days=30)

    assert [w.cost_source for w in found] == ["export", "export", "unknown", "unknown"]
    assert [w.calls for w in found] == [2, 5, 9, 3]
    assert [w.cost_usd_month for w in found] == pytest.approx([1.0, 0.005, None, None])

    tied = make_records(2, system="Y", cost_usd=None) + make_records(2, system="X", cost_usd=None)
    hashes = [w.template_hash for w in discover_workloads(tied, window_days=30)]
    assert hashes == sorted(hashes)


def test_rates_use_the_longer_of_window_and_export_span() -> None:
    a = make_records(3, system="A")
    spread = [*a, make_record(system="B", ts=T0 + timedelta(days=60), cost_usd=0.003)]
    by_hash = {w.template_hash: w for w in discover_workloads(spread, window_days=30)}

    assert by_hash[template_hash("A")].calls_per_day == 3 / 60
    assert by_hash[template_hash("B")].cost_usd_month == pytest.approx(0.003 / 60 * 30)
    assert discover_workloads(a, window_days=90)[0].calls_per_day == 3 / 90


def test_partial_export_cost_is_extrapolated_from_the_priced_calls() -> None:
    records = make_records(4, cost_usd=0.01) + make_records(4, cost_usd=None)
    (workload,) = discover_workloads(records, window_days=30)
    assert workload.cost_source == "export"
    assert workload.cost_usd_month == pytest.approx(8 * 0.01)


def test_latency_quantiles_and_token_sums() -> None:
    records = [make_record(latency_ms=float(ms), trace_id=str(ms)) for ms in range(1, 101)]
    (workload,) = discover_workloads(records, window_days=30)
    assert workload.latency_p50_ms == 50.5
    assert 95.0 <= workload.latency_p95_ms <= 96.0
    assert workload.prompt_tokens == 100 * 100
    assert workload.completion_tokens == 100 * 2


def test_entropy_and_distinct_outputs_over_normalized_responses() -> None:
    records = make_records(8)  # a/b alternating: two equiprobable outputs
    records += [make_record(response=" A ", trace_id="x"), make_record(response="B", trace_id="y")]
    (workload,) = discover_workloads(records, window_days=30)
    assert workload.distinct_outputs == 2
    assert math.isclose(workload.entropy_bits, 1.0)


def test_structure_class_comes_from_the_first_sample_in_time_order() -> None:
    labels = make_records(STRUCTURE_SAMPLE)
    prose = [
        make_record(
            response=" ".join(f"w{i}{j}" for j in range(30)),
            ts=T0 + timedelta(days=1, minutes=i),
            trace_id=f"p{i}",
        )
        for i in range(50)
    ]
    (workload,) = discover_workloads(prose + labels, window_days=30)

    assert workload.structure_class is StructureClass.ENUM_LABEL
    assert workload.sample_size == STRUCTURE_SAMPLE
    assert workload.calls == STRUCTURE_SAMPLE + 50
    assert workload.record_indices[:2] == (50, 51)  # time order, not input order
    assert set(workload.record_indices) == set(range(STRUCTURE_SAMPLE + 50))


def test_models_are_sorted_and_unique() -> None:
    records = make_records(3, model="z-model") + make_records(2, model="a-model")
    assert discover_workloads(records, window_days=30)[0].models == ("a-model", "z-model")


def test_template_excerpt_is_redacted_and_capped() -> None:
    assert template_excerpt("mail ops@example.com") == "mail [REDACTED:PII_EMAIL]"
    assert template_excerpt("x" * (EXCERPT_CHARS + 50)) == "x" * EXCERPT_CHARS
    assert template_excerpt("") == ""
    (workload,) = discover_workloads([make_record(system="Order 42, email a@b.co")], window_days=30)
    assert workload.template_excerpt == "Order <NUM>, email <EMAIL>"


def test_to_json_uses_the_report_field_names() -> None:
    (workload,) = discover_workloads(make_records(2), window_days=30)
    payload = workload.to_json()
    assert json.loads(json.dumps(payload)) == payload
    assert list(payload) == [
        "id",
        "template_hash",
        "template_excerpt",
        "structure_class",
        "confidence",
        "calls",
        "calls_per_day",
        "tokens",
        "cost_usd_month",
        "cost_source",
        "latency_ms",
        "distinct_outputs",
        "entropy",
    ]
    assert payload["structure_class"] == "enum_label"
    assert payload["tokens"] == {"prompt": 200, "completion": 4}
    assert payload["latency_ms"] == {"p50": 500.0, "p95": 500.0}


def test_workload_is_frozen_with_the_planned_fields() -> None:
    assert [f.name for f in fields(Workload)] == [
        "id",
        "template_hash",
        "template_excerpt",
        "structure_class",
        "confidence",
        "calls",
        "calls_per_day",
        "prompt_tokens",
        "completion_tokens",
        "cost_usd_month",
        "cost_source",
        "latency_p50_ms",
        "latency_p95_ms",
        "distinct_outputs",
        "entropy_bits",
        "stability",
        "sample_size",
        "models",
        "record_indices",
    ]
    assert not hasattr(discover_workloads(make_records(1), window_days=30)[0], "__dict__")


def _mixed_records() -> list[TraceRecord]:
    return (
        make_records(6, system="Label {{ticket}}", cost_usd=0.002)
        + make_records(3, system="Draft a reply", response="Thanks, we shipped it out today.")
        + make_records(2, system=None, response='{"a": 1}', cost_usd=None)
    )


def test_same_input_twice_is_identical() -> None:
    records = _mixed_records()
    assert discover_workloads(records, window_days=30) == discover_workloads(
        records, window_days=30
    )


@given(order=st.permutations(list(range(11))))
@settings(max_examples=50, deadline=None)
def test_discovery_is_order_independent(order: list[int]) -> None:
    records = _mixed_records()
    shuffled = [records[i] for i in order]
    baseline = discover_workloads(records, window_days=30)
    found = discover_workloads(shuffled, window_days=30)

    assert [(w.template_hash, w.structure_class) for w in found] == [
        (w.template_hash, w.structure_class) for w in baseline
    ]
    assert all(w.calls == len(w.record_indices) for w in found)
    assert Counter(len(w.record_indices) for w in found) == Counter({6: 1, 3: 1, 2: 1})
    for a, b in zip(found, baseline, strict=True):
        assert [shuffled[i] for i in a.record_indices] == [records[i] for i in b.record_indices]
