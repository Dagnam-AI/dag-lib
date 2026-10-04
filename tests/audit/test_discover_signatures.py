"""Discovery keys: a router's tool sets, forced tools, media, sampling, and bounded memory."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import timedelta
import json
from pathlib import Path
import tracemalloc

import pytest
from tests.audit._records import T0, make_record, make_records

from dagnam.audit import TraceRecord, discover, read_traces
from dagnam.audit.discover import (
    discover_workloads,
)
from dagnam.audit.normalize import normalize_template, template_hash


def _router_line(
    i: int,
    tools: list[str],
    *,
    system: str = "Route the ticket to the right team.",
    call: str = "route",
    tool_choice: object = None,
) -> dict[str, object]:
    """One call answered with a ``call`` tool call (a router, forced tools)."""
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
    # The offered tool set keyed the workload, so one 2,400-call router split
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
    # A forced tool IS the output schema (OpenAI ``tool_choice.function.name``,
    # LangChain ``with_structured_output``), and keying on the offered tools merged two extraction tasks into
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
    # An export that holds 10% of the traffic understated calls and spend tenfold.
    records = make_records(4, cost_usd=0.01)
    (full,) = discover_workloads(records, window_days=30)
    (sampled,) = discover_workloads(records, window_days=30, sample_rate=0.1)

    assert sampled.calls == full.calls == 4  # what the export holds, for the sample floors
    assert sampled.calls_per_day == pytest.approx(full.calls_per_day * 10)
    assert sampled.cost_usd_month == pytest.approx((full.cost_usd_month or 0) * 10)
    with pytest.raises(ValueError, match="sample_rate"):
        discover_workloads(records, window_days=30, sample_rate=0)


def test_discovery_reads_a_stream_once() -> None:
    # The scan no longer holds every record: discovery takes an iterator.
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
