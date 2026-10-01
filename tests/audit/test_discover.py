"""Workload discovery: template hash x structure class, with per-workload statistics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import fields, replace
from datetime import timedelta
import json
import math
from pathlib import Path
import tracemalloc

from hypothesis import given, settings, strategies as st
import pytest
from tests.audit._records import T0, make_record, make_records

from dagnam.audit import TraceRecord, discover, read_traces
from dagnam.audit.discover import (
    ModelUsage,
    Workload,
    by_spend,
    discover_workloads,
    template_excerpt,
)
from dagnam.audit.normalize import UNSTRUCTURED, normalize_template, template_hash
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
    # The fixture names its steps (``metadata.workload``), and the name keys them (N11).
    assert {w.name for w in found} == set(GROUND_TRUTH)
    assert all(w.name == by_hash[w.template_hash] for w in found)
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
    assert unstructured.id == f"{UNSTRUCTURED}-short"
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
    assert workload.template_excerpt == "Order <NUM>, email [REDACTED:PII_EMAIL]"


def test_the_excerpt_is_redacted_before_it_is_normalized() -> None:
    # Normalization turns the digits of an identifier into <NUM>, after which no
    # detector can recognise what is left of it; redaction must see the raw text.
    card = "4111 1111 1111 1111"
    assert template_excerpt(f"Refund card {card} today") == (
        "Refund card [REDACTED:PII_PAYMENT_CARD] today"
    )
    phone = "Escalate to Jane (+1 415 555 0100)."
    assert template_excerpt(phone) == "Escalate to Jane ([REDACTED:PII_PHONE])."


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
        "models",
        "response_mode",
        "name",
        "media_calls",
    ]
    assert payload["models"] == ["gpt-4o-mini"]
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
        "response_mode",
        "usage_by_model",
        "name",
        "media_calls",
        "first_ts",
        "last_ts",
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


def test_traces_without_a_system_prompt_split_into_per_shape_buckets() -> None:
    # 3,000 calls with no system prompt -- yes/no labels, then JSON, then prose --
    # were one enum_label candidate with 2,002 distinct outputs.
    records = [
        make_record(
            system=None,
            response=(
                "yn"[i % 2]
                if i < 1_000
                else json.dumps({"a": i, "b": "x"})
                if i < 2_000
                else f"Here is a long free text answer number {i} with many words in it."
            ),
            ts=T0 + timedelta(minutes=i),
            trace_id=f"t{i}",
        )
        for i in range(3_000)
    ]
    found = {w.id: w for w in discover_workloads(records, window_days=30)}

    assert {i: w.structure_class for i, w in found.items()} == {
        f"{UNSTRUCTURED}-short": StructureClass.ENUM_LABEL,
        f"{UNSTRUCTURED}-json": StructureClass.JSON_OBJECT,
        f"{UNSTRUCTURED}-long": StructureClass.FREE_TEXT,
    }
    assert {w.calls for w in found.values()} == {1_000}
    assert {(w.template_hash, w.confidence) for w in found.values()} == {(UNSTRUCTURED, "low")}
    assert found[f"{UNSTRUCTURED}-short"].distinct_outputs == 2


def test_usage_is_kept_per_model_spelling() -> None:
    records = [
        make_record(model=model, prompt_tokens=p, completion_tokens=c, trace_id=f"t{i}")
        for i, (model, p, c) in enumerate(
            [("gpt-4o-mini", 10, 1), ("gpt-4o-mini-2024-07-18", 20, 2), ("gpt-4o-mini", 30, 3)]
        )
    ]
    (workload,) = discover_workloads(records, window_days=30)

    assert workload.models == ("gpt-4o-mini", "gpt-4o-mini-2024-07-18")
    assert workload.usage_by_model == (
        ModelUsage("gpt-4o-mini", calls=2, prompt_tokens=40, completion_tokens=4),
        ModelUsage("gpt-4o-mini-2024-07-18", calls=1, prompt_tokens=20, completion_tokens=2),
    )


def test_a_workload_answering_with_tool_calls_is_in_tool_call_mode() -> None:
    call = {"function": {"name": "record", "arguments": '{"product": "p1"}'}}
    tooled = [make_record(response="", tool_calls=(call,), trace_id=f"t{i}") for i in range(3)]
    (workload,) = discover_workloads([*tooled, make_record(trace_id="x")], window_days=30)
    assert workload.response_mode == "tool_call"
    assert workload.to_json()["response_mode"] == "tool_call"
    (plain,) = discover_workloads([*tooled[:2], *make_records(2)], window_days=30)
    assert plain.response_mode == "text"


def test_by_spend_orders_most_spend_first_then_calls_then_id() -> None:
    found = discover_workloads(
        make_records(2, system="A", cost_usd=None) + make_records(3, system="B", cost_usd=None),
        window_days=30,
    )
    priced = [replace(w, cost_usd_month=float(w.calls)) for w in found]
    assert [w.calls for w in by_spend(reversed(priced))] == [3, 2]
    assert by_spend([]) == ()


def test_a_secret_in_the_system_prompt_never_reaches_the_excerpt() -> None:
    # A-11: the published excerpt read "... Bearer sk-live-AbCdEfGhIjKlMnOpQrStUvWx ...".
    system = (
        "You are Acme's extraction bot. Escalations go to Jane Doe (jane.doe@acme.com,"
        " +1 415 555 0100). Auth header: Bearer sk-live-AbCdEfGhIjKlMnOpQrStUvWx."
        " Extract the customer's contact fields as JSON."
    )
    (workload,) = discover_workloads([make_record(system=system)], window_days=30)

    assert "<SECRET>" in workload.template_excerpt
    assert "sk-live" not in workload.template_excerpt
    assert "AbCdEf" not in workload.template_excerpt
    # A key with digits in it: normalizing first left "sk-proj-a<NUM>B<NUM>...".
    digits = template_excerpt("Use Bearer sk-proj-a1B2c3D4e5F6g7H8i9J0kLmN for the API.")
    assert digits == "Use Bearer <SECRET> for the API."
    # The id is the hash of the normalized prompt, whatever redaction does to the excerpt.
    assert workload.id == template_hash(system)


def test_a_workload_hint_overrides_the_template_hash() -> None:
    # N11 / spec section 14: the customer's own name for a step is the most reliable key.
    # One system prompt, two named steps: two workloads; a per-call RAG prompt under
    # one name: one workload.
    records = [
        make_record(workload_hint="intent" if i % 2 else "urgency", trace_id=f"t{i}")
        for i in range(6)
    ] + [
        make_record(system=f"Answer using context {i}: ...", workload_hint="rag", trace_id=f"r{i}")
        for i in range(5)
    ]
    found = {w.name: w for w in discover_workloads(records, window_days=30)}

    assert {name: w.calls for name, w in found.items()} == {"intent": 3, "urgency": 3, "rag": 5}
    assert all(len(w.id) == 16 and w.id != w.template_hash for w in found.values())
    assert found["rag"].confidence == "high"
    assert found["intent"].template_hash == template_hash("Label the ticket.")
    assert found["intent"].to_json()["name"] == "intent"
    (plain,) = discover_workloads(make_records(2), window_days=30)
    assert plain.name is None
    assert plain.id == plain.template_hash


def test_the_request_signature_joins_the_key() -> None:
    # N12: one system prompt, two response schemas -- two extraction tasks merged.
    records = [
        make_record(signature=f"schema={'invoice' if i % 2 else 'resume'}", trace_id=f"t{i}")
        for i in range(6)
    ]
    found = discover_workloads(records, window_days=30)

    assert sorted(w.calls for w in found) == [3, 3]
    assert len({w.id for w in found}) == 2
    assert {w.template_hash for w in found} == {template_hash("Label the ticket.")}
    no_system = discover_workloads(
        [make_record(system=None, signature="schema=invoice")], window_days=30
    )
    assert no_system[0].id != UNSTRUCTURED
    assert no_system[0].confidence == "high"


def _router_line(
    i: int,
    tools: list[str],
    *,
    system: str = "Route the ticket to the right team.",
    call: str = "route",
    tool_choice: object = None,
) -> dict[str, object]:
    """One call answered with a ``call`` tool call (the B2-2 router, the RR-2 forced tools)."""
    offered = [{"type": "function", "function": {"name": t, "parameters": {}}} for t in tools]
    arguments = json.dumps({"team": ["billing", "shipping"][i % 2]})
    system_turn = [{"role": "system", "content": system}] if system else []
    request: dict[str, object] = {
        "model": "gpt-4o-mini",
        "messages": [*system_turn, {"role": "user", "content": f"ticket {i}"}],
        "tools": offered,
    }
    if tool_choice is not None:
        request["tool_choice"] = tool_choice
    return {
        "request": request,
        "response": {
            "id": f"c{i}",
            "created": int((T0 + timedelta(minutes=i)).timestamp()),
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "x",
                                "type": "function",
                                "function": {"name": call, "arguments": arguments},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 400, "completion_tokens": 10},
        },
    }


@pytest.mark.parametrize(
    "offered",
    [
        # a deploy adds a tool mid-window: it was 2 x 1,200
        lambda i: ["route", "escalate"] + (["refund"] if i >= 1_200 else []),
        # permission-gated or retrieved tools: 4 workloads, three under the 1,000-call floor
        lambda i: ["route"] + (["escalate"] if i % 3 else []) + (["refund"] if i % 5 == 0 else []),
    ],
    ids=["tool-added-mid-window", "per-call-tool-subsets"],
)
def test_a_router_offered_varying_tool_sets_is_one_workload(
    tmp_path: Path, offered: Callable[[int], list[str]]
) -> None:
    # B2-2: the offered tool set keyed the workload, so one 2,400-call router split
    # into pieces too small to audit. What a call OFFERS is not what it does.
    path = tmp_path / "router.jsonl"
    path.write_text("".join(json.dumps(_router_line(i, offered(i))) + "\n" for i in range(2_400)))

    (workload,) = discover_workloads(read_traces(path, source="openai")[0], window_days=30)

    assert workload.calls == 2_400
    assert workload.response_mode == "tool_call"
    assert workload.id == template_hash("Route the ticket to the right team.")


@pytest.mark.parametrize("system", ["Extract the fields.", ""], ids=["one-system", "no-system"])
def test_two_forced_tools_are_two_workloads_and_auto_or_required_is_one(
    tmp_path: Path, system: str
) -> None:
    # RR-2: a forced tool IS the output schema (OpenAI ``tool_choice.function.name``,
    # LangChain ``with_structured_output``), and B2-2 merged two extraction tasks into
    # one 2,400-call workload -- without a system prompt, into ``unstructured-json``.
    def forced(i: int) -> dict[str, object]:
        tool = ["extract_invoice", "extract_receipt"][i % 2]
        choice = {"type": "function", "function": {"name": tool}}
        return _router_line(i, [tool], system=system, call=tool, tool_choice=choice)

    path = tmp_path / "forced.jsonl"
    path.write_text("".join(json.dumps(forced(i)) + "\n" for i in range(2_400)))
    found = discover_workloads(read_traces(path, source="openai")[0], window_days=30)
    assert sorted(w.calls for w in found) == [1_200, 1_200]
    assert {w.confidence for w in found} == {"high"}

    free = [
        _router_line(i, ["route", "escalate"], tool_choice=["auto", "required"][i % 2])
        for i in range(2_400)
    ]
    path.write_text("".join(json.dumps(line) + "\n" for line in free))
    (router,) = discover_workloads(read_traces(path, source="openai")[0], window_days=30)
    assert router.calls == 2_400


def test_calls_carrying_media_are_counted() -> None:
    records = [*make_records(3), make_record(has_media=True, trace_id="m")]
    (workload,) = discover_workloads(records, window_days=30)
    assert workload.media_calls == 1


def test_a_sample_rate_scales_volume_and_spend_back_up() -> None:
    # N18: an export that holds 10% of the traffic understated calls and spend tenfold.
    records = make_records(4, cost_usd=0.01)
    (full,) = discover_workloads(records, window_days=30)
    (sampled,) = discover_workloads(records, window_days=30, sample_rate=0.1)

    assert sampled.calls == full.calls == 4  # what the export holds, for the sample floors
    assert sampled.calls_per_day == pytest.approx(full.calls_per_day * 10)
    assert sampled.cost_usd_month == pytest.approx((full.cost_usd_month or 0) * 10)
    with pytest.raises(ValueError, match="sample_rate"):
        discover_workloads(records, window_days=30, sample_rate=0)


def test_discovery_reads_a_stream_once() -> None:
    # R3-16: the scan no longer holds every record: discovery takes an iterator.
    records = make_records(5)
    stream = iter(records)
    (workload,) = discover_workloads(stream, window_days=30)
    assert workload.calls == 5
    assert list(stream) == []
    assert (workload.first_ts, workload.last_ts) == (records[0].ts, records[-1].ts)


def test_a_named_workload_keeps_the_text_of_a_bounded_number_of_templates() -> None:
    # A per-call prompt under one name (RAG) must not hold every system prompt.
    one_offs = [
        make_record(
            system=f"Context {chr(65 + i // 26)}{chr(65 + i % 26)}: answer from it.",
            workload_hint="rag",
            ts=T0 + timedelta(minutes=i),
            trace_id=f"o{i}",
        )
        for i in range(70)
    ]
    repeated = [
        make_record(
            system="The common prompt.",
            workload_hint="rag",
            ts=T0 + timedelta(hours=5, minutes=i),
            trace_id=f"c{i}",
        )
        for i in range(3)
    ]
    (workload,) = discover_workloads(one_offs + repeated, window_days=30)

    # The modal template came after the kept ones: the excerpt is the earliest kept.
    assert workload.template_hash == template_hash("The common prompt.")
    assert workload.template_excerpt == "Context AA: answer from it."
    assert workload.stability == 3 / 73


def test_discovery_holds_no_system_prompt_per_template(monkeypatch: pytest.MonkeyPatch) -> None:
    # m6: every distinct template kept its whole raw system prompt until the end, so an
    # export of per-call (RAG) prompts was held in memory again, as in batch 1.
    monkeypatch.setattr(discover, "normalize_template", normalize_template.__wrapped__)
    prompt_chars, templates = 20_000, 30
    held: list[int] = []

    def stream() -> Iterator[TraceRecord]:
        for i in range(templates):
            system = f"Context {chr(65 + i // 26)}{chr(65 + i % 26)}: " + "passage " * 2_500
            yield make_record(system=system, ts=T0 + timedelta(minutes=i), trace_id=f"r{i}")
        held.append(tracemalloc.get_traced_memory()[0])  # every group is still alive here

    tracemalloc.start()
    try:
        found = discover_workloads(stream(), window_days=30)
    finally:
        tracemalloc.stop()

    assert len(found) == templates
    assert found[0].template_excerpt.startswith("Context A")
    # 690 KB held before (more than the 600 KB of prompts); about 160 KB of bookkeeping now.
    assert held[0] < prompt_chars * templates / 2
