"""Derived training rows: format by structure class, truncation, dedup, the pipeline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

from dagnam_contracts.prompts import render_chat_prompt
import pytest

from dagnam.audit import Message, TraceRecord, discover_workloads, read_traces
from dagnam.audit.derive import (
    FORMAT_BY_STRUCTURE,
    DedupStats,
    DeriveStats,
    build_dataset,
    dedup_rows,
    derive_rows,
    derive_workloads,
    normalize_label,
)
from dagnam.audit.thresholds import ENUM_MAX_DISTINCT, MAX_TRAIN_ROWS

FIXTURE = Path(__file__).parent / "fixtures" / "langfuse_sample.jsonl"
TICKETS = 3


def _records(hint: str) -> list[TraceRecord]:
    records, _ = read_traces(FIXTURE, source="langfuse")
    return [r for r in records if r.workload_hint == hint]


def _rec(
    *,
    response: str = "returns",
    system: str | None = "classify",
    turns: tuple[str, ...] = ("hello",),
    ts: int = 0,
    session: str | None = None,
) -> TraceRecord:
    return TraceRecord(
        trace_id=f"t{ts}",
        ts=datetime(2026, 8, 1, tzinfo=UTC) + timedelta(seconds=ts),
        model="m",
        system=system,
        messages=tuple(Message("user", t) for t in turns),
        response=response,
        response_tool_calls=(),
        prompt_tokens=1,
        completion_tokens=1,
        latency_ms=1.0,
        cost_usd=None,
        session_id=session,
        outcome=None,
        workload_hint=None,
    )


@pytest.mark.parametrize(
    ("raw", "label"),
    [
        ("Returns.", "returns"),
        ("  BILLING!!  ", "billing"),
        ("high", "high"),
        ("...", ""),
        ('"Refund"', "refund"),
    ],
)
def test_normalize_label(raw: str, label: str) -> None:
    assert normalize_label(raw) == label


def test_format_registry_covers_every_structure_class() -> None:
    assert FORMAT_BY_STRUCTURE == {
        "enum_label": "labeled-example",
        "json_object": "chat-messages",
        "short_span": "chat-messages",
        "free_text": "chat-messages",
    }
    with pytest.raises(ValueError, match="unknown structure class"):
        derive_rows([_rec()], structure_class="table", max_seq_length=10)


def test_enum_workload_yields_labeled_example_rows() -> None:
    records = _records("intent")
    rows, stats = derive_rows(records, structure_class="enum_label", max_seq_length=4096)

    assert stats == DeriveStats(
        rows=TICKETS,
        truncated=0,
        truncation_rate=0.0,
        format_key="labeled-example",
        skipped=0,
        record_indices=(0, 1, 2),
    )
    assert [r["label"] for r in rows] == ["returns", "billing", "shipping"]
    for record, row in zip(records, rows, strict=True):
        assert set(row) == {"input", "label"}
        assert row["input"]
        assert row["label"]
        expected = render_chat_prompt(
            [{"role": m.role, "content": m.content} for m in record.messages],
            system=record.system,
        )
        assert row["input"] == expected


def test_free_text_workload_yields_chat_messages_rows() -> None:
    records = _records("reply")
    rows, stats = derive_rows(records, structure_class="free_text", max_seq_length=4096)

    assert stats.format_key == "chat-messages"
    assert stats.rows == TICKETS
    for record, row in zip(records, rows, strict=True):
        assert set(row) == {"messages"}
        roles = [m["role"] for m in row["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert row["messages"][0]["content"] == record.system
        assert row["messages"][-1]["content"] == record.response


def test_chat_messages_without_system_prompt_omits_the_system_turn() -> None:
    rows, _ = derive_rows(
        [_rec(system=None, response=" hi ")], structure_class="short_span", max_seq_length=100
    )
    assert rows[0]["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_front_truncation_keeps_the_last_user_turn() -> None:
    record = _rec(system="s" * 40, turns=("first " * 10, "second " * 10, "LAST TURN"))
    rendered = render_chat_prompt(
        [{"role": "user", "content": t} for t in ("first " * 10, "second " * 10, "LAST TURN")],
        system="s" * 40,
    )
    limit = 40
    rows, stats = derive_rows([record], structure_class="enum_label", max_seq_length=limit)

    assert rows[0]["input"] == rendered[-limit:]
    assert rows[0]["input"].endswith("<|user|>\nLAST TURN\n")
    assert len(rows[0]["input"]) == limit
    assert (stats.truncated, stats.truncation_rate) == (1, 1.0)


def test_chat_messages_truncation_drops_the_oldest_turns_first() -> None:
    record = _rec(system="sys", turns=("a" * 30, "b" * 30, "c" * 30))
    rows, stats = derive_rows([record], structure_class="json_object", max_seq_length=40)

    contents = [m["content"] for m in rows[0]["messages"]]
    assert contents == ["sys", "c" * 30, "returns"]
    assert stats.truncated == 1


def test_chat_messages_never_drops_the_last_turn() -> None:
    record = _rec(system="sys", turns=("a" * 30, "b" * 300))
    rows, stats = derive_rows([record], structure_class="json_object", max_seq_length=70)

    contents = [m["content"] for m in rows[0]["messages"]]
    assert contents == ["sys", "b" * 300, "returns"]
    assert stats.truncated == 1


def test_records_with_an_empty_response_or_prompt_are_skipped_and_indexed() -> None:
    records = [_rec(response="..."), _rec(ts=1), _rec(ts=2, system=None, turns=())]
    rows, stats = derive_rows(records, structure_class="enum_label", max_seq_length=100)

    assert len(rows) == 1
    assert stats.skipped == 2
    assert stats.record_indices == (1,)


def test_chat_messages_records_without_a_prompt_or_response_are_skipped() -> None:
    records = [_rec(system=None, turns=()), _rec(ts=1, response="  "), _rec(ts=2)]
    rows, stats = derive_rows(records, structure_class="free_text", max_seq_length=100)

    assert len(rows) == 1
    assert stats.record_indices == (2,)


def test_empty_input_has_a_zero_truncation_rate() -> None:
    rows, stats = derive_rows([], structure_class="enum_label", max_seq_length=10)
    assert rows == []
    assert stats == DeriveStats(0, 0, 0.0, "labeled-example", 0, ())


def test_dedup_keeps_first_and_reports_count() -> None:
    rows = [
        {"input": "a", "label": "x"},
        {"label": "x", "input": "a"},
        {"input": "b", "label": "y"},
    ]
    kept, stats = dedup_rows(rows)

    assert kept == [rows[0], rows[2]]
    assert stats == DedupStats(removed=1, kept_indices=(0, 2))
    assert dedup_rows(kept) == (kept, DedupStats(removed=0, kept_indices=(0, 1)))


def test_derivation_is_deterministic() -> None:
    records = _records("intent")
    first = derive_rows(records, structure_class="enum_label", max_seq_length=64)
    second = derive_rows(records, structure_class="enum_label", max_seq_length=64)
    assert first == second


def test_build_dataset_aligns_split_indices_with_the_kept_rows() -> None:
    # Ten sessions of two records; one exact duplicate and one PII row inside.
    records: list[TraceRecord] = []
    for i in range(20):
        turn = "same ticket" if i in (3, 5) else f"ticket {i} from a{i}@x.io"
        records.append(_rec(turns=(turn,), ts=i, session=f"s{i // 2}", response="returns"))

    dataset = build_dataset(records, structure_class="enum_label", max_seq_length=4096)

    assert len(dataset.rows) == 19
    assert all("@" not in row["input"] for row in dataset.rows)
    assert dataset.stats["format_key"] == "labeled-example"
    assert dataset.stats["derive"]["rows"] == 20
    assert dataset.stats["redact"]["counts"]["PII_EMAIL"] == 18
    assert dataset.stats["dedup"]["removed"] == 1
    indices = sorted(dataset.split["train"] + dataset.split["eval_holdout"])
    assert indices == list(range(19))
    # Row 5 was the duplicate of row 3, so record 6 sits at row 5: the split
    # must be computed over the records that survived, not the original list.
    assert dataset.rows[5]["input"].endswith("ticket 6 from [REDACTED:PII_EMAIL]\n")
    assert dataset.stats["boundary_ts"] == records[16].ts.isoformat()
    assert dataset.split["eval_holdout"] == [15, 16, 17, 18]


def _tool_call(arguments: Mapping[str, object]) -> dict[str, object]:
    function = {"name": "record", "arguments": json.dumps(arguments)}
    return {"id": "c", "type": "function", "function": function}


def test_a_tool_call_is_trained_on_its_name_and_arguments() -> None:
    # 1,200 function-calling extractions with ``content: null`` were all skipped
    # (0 holdout rows); with a preamble, the preamble became the target.
    args = {"product": "p1", "sentiment": "positive"}
    silent = replace(_rec(response=""), response_tool_calls=(_tool_call(args),))
    chatty = replace(
        _rec(response="I'll record that.", ts=1), response_tool_calls=(_tool_call(args),)
    )

    rows, stats = derive_rows([silent, chatty], structure_class="json_object", max_seq_length=100)

    assert stats.skipped == 0
    call = {"arguments": args, "name": "record"}
    assert [json.loads(r["messages"][-1]["content"]) for r in rows] == [call, call]
    labels, _ = derive_rows([silent], structure_class="enum_label", max_seq_length=100)
    assert labels[0]["label"] == normalize_label(json.dumps(call, sort_keys=True))


def test_redaction_runs_before_the_front_cut() -> None:
    # A cut landing inside an SSN shipped "45-6789 ..." -- 6 of its 9 digits, with
    # no redaction counted, because redaction ran on the already-cut text.
    def rendered(filler: int) -> str:
        turn = "SSN 123-45-6789 " + "x" * filler
        return render_chat_prompt([{"role": "user", "content": turn}], system="classify")

    start = rendered(0).index("123-45-6789")
    filler = 2_048 - len(rendered(0)) + start + len("123-")
    assert rendered(filler)[-2_048:].startswith("45-6789")  # where the cut lands
    record = _rec(system="classify", turns=("SSN 123-45-6789 " + "x" * filler,))

    dataset = build_dataset([record], structure_class="enum_label", max_seq_length=2_048)

    assert "6789" not in dataset.rows[0]["input"]
    assert dataset.stats["redact"]["counts"]["PII_NATIONAL_ID"] == 1


def test_build_dataset_counts_the_targets_redaction_changed() -> None:
    # A student that learned to emit "[REDACTED:PII_EMAIL]" scores as agreeing with a
    # redacted truth, although it cannot return the email: the scan says so.
    records = [
        _rec(response=json.dumps({"email": f"m{i}@example.org", "n": i}), ts=i, turns=(f"t{i}",))
        for i in range(3)
    ] + [_rec(response=json.dumps({"email": None, "n": 9}), ts=3, turns=("t3",))]

    dataset = build_dataset(records, structure_class="json_object", max_seq_length=4_096)

    assert dataset.stats["redact"]["truths_changed"] == 3
    assert dataset.stats["redact"]["rows_changed"] == 3
    assert "example.org" not in json.dumps(dataset.rows)


def test_build_dataset_caps_the_training_rows_and_keeps_the_holdout() -> None:
    # Every train row past the cap would be paid GPU time past a hard ceiling.
    records = [
        _rec(response="billing" if i % 5 else "refund", ts=i, turns=(f"ticket {i}",))
        for i in range(50)
    ]
    whole = build_dataset(records, structure_class="enum_label", max_seq_length=4_096)
    capped = build_dataset(
        records, structure_class="enum_label", max_seq_length=4_096, max_train_rows=10
    )

    assert len(whole.split["train"]) == 40
    assert len(capped.split["train"]) == 10
    assert len(capped.rows) == 10 + len(whole.split["eval_holdout"])
    holdout = [capped.rows[i] for i in capped.split["eval_holdout"]]
    assert holdout == [whole.rows[i] for i in whole.split["eval_holdout"]]
    assert sorted(capped.split["train"] + capped.split["eval_holdout"]) == list(
        range(len(capped.rows))
    )
    labels = [capped.rows[i]["label"] for i in capped.split["train"]]
    assert labels.count("refund") == 2  # a fifth of the rows, as in the whole
    assert capped.stats["cap"] == {"limit": 10, "train_rows": 40, "dropped": 30}
    assert whole.stats["cap"] == {"limit": MAX_TRAIN_ROWS, "train_rows": 40, "dropped": 0}


def test_the_cap_keeps_a_router_s_rare_route() -> None:
    # Tool-call and JSON rows all shared one stratum, so the cap sampled a router's
    # routes at random: a rare route could fall to none while the holdout kept it.
    def routed(i: int) -> TraceRecord:
        call = _tool_call({"team": "legal" if i % 10 == 3 else "billing"})
        return replace(_rec(response="", ts=i, turns=(f"ticket {i}",)), response_tool_calls=(call,))

    records = [routed(i) for i in range(50)]
    capped = build_dataset(
        records, structure_class="json_object", max_seq_length=4_096, max_train_rows=10
    )

    teams = [
        json.loads(capped.rows[i]["messages"][-1]["content"])["arguments"]["team"]
        for i in capped.split["train"]
    ]
    assert Counter(teams) == {"billing": 9, "legal": 1}  # ceil(10 x 36/40), ceil(10 x 4/40)


INVOICE_LINES = "Invoice line 17: widget, 12 x 3.50 EUR, net 42.00 EUR\n" * 220


def test_rows_over_the_student_s_context_leave_the_training_split_only() -> None:
    # qlora-sft-chat@1.2 drops, never cuts, a row over its 2,048 tokens -- at
    # train time, after the credits are spent. A 12k-character invoice in the user turn
    # is such a row (the last turn is kept whole); the scan now leaves it out of
    # training and counts it. The holdout keeps it, as serving will see it.
    records = [
        _rec(
            system="Extract the invoice total as JSON.",
            response=json.dumps({"total": i}),
            turns=(INVOICE_LINES if i % 2 else f"invoice {i}: 1 line, total {i} EUR",),
            ts=i,
        )
        for i in range(50)
    ]

    dataset = build_dataset(records, structure_class="json_object", max_seq_length=2_048)

    assert dataset.stats["budget"] == {"max_tokens": 2_048, "dropped": 20}
    assert len(dataset.split["train"]) == 20
    assert all(len(json.dumps(dataset.rows[i])) < 500 for i in dataset.split["train"])
    holdout = [dataset.rows[i]["messages"][1]["content"] for i in dataset.split["eval_holdout"]]
    assert len(holdout) == 10
    assert holdout.count(INVOICE_LINES) == 5
    assert dataset.stats["cap"] == {"limit": MAX_TRAIN_ROWS, "train_rows": 20, "dropped": 0}
    # A head-tune classifier truncates its input by design: nothing leaves it.
    labels = build_dataset(records, structure_class="enum_label", max_seq_length=2_048)
    assert labels.stats["budget"] == {"max_tokens": None, "dropped": 0}
    assert len(labels.split["train"]) == 40


def test_the_cap_still_caps_an_extraction_whose_answers_are_all_distinct() -> None:
    # A stratum per distinct answer would keep a row of each and disable the
    # cap, so past ENUM_MAX_DISTINCT distinct chat targets they are one stratum.
    records = [
        _rec(response=json.dumps({"total": i}), ts=i, turns=(f"invoice {i}",)) for i in range(100)
    ]
    capped = build_dataset(
        records, structure_class="json_object", max_seq_length=4_096, max_train_rows=10
    )
    assert len(capped.split["train"]) == 10
    assert capped.stats["cap"] == {"limit": 10, "train_rows": 80, "dropped": 70}


def test_the_cap_keeps_every_label_however_many_there_are() -> None:
    # The "too many strata" fallback meant for extraction answers also caught a
    # label workload with over 50 labels in training, and dropped a one-row label.
    labels = [f"label{i % (ENUM_MAX_DISTINCT + 5)}" for i in range(300)]
    labels[7] = "tail"  # one early row, so it trains
    records = [_rec(response=label, ts=i, turns=(f"ticket {i}",)) for i, label in enumerate(labels)]
    capped = build_dataset(
        records, structure_class="enum_label", max_seq_length=4_096, max_train_rows=100
    )

    kept = {capped.rows[i]["label"] for i in capped.split["train"]}
    assert "tail" in kept
    assert len(kept) == ENUM_MAX_DISTINCT + 6


def test_derive_workloads_builds_the_same_datasets_from_two_streamed_passes() -> None:
    # The scan held every record of the export; now two passes plan and then
    # derive the rows each workload keeps, and nothing else is held.
    records = (
        [
            _rec(system="labels", ts=i, turns=(f"t{i} from a{i}@x.io",), response="ab"[i % 2])
            for i in range(12)
        ]
        + [
            _rec(system="labels", ts=20, turns=("t0 from a0@x.io",), response="a"),  # a duplicate
            _rec(system=None, ts=30, turns=(), response="skipped"),
        ]
        + [_rec(system="prose", ts=40 + i, turns=(f"p{i}",)) for i in range(3)]
    )
    workloads = discover_workloads(records[:14], window_days=30)
    opened: list[int] = []

    def stream() -> Iterator[TraceRecord]:
        opened.append(1)
        return iter(records)

    datasets = derive_workloads(stream, workloads, max_seq_length=4_096)

    assert len(opened) == 2
    for w in workloads:
        whole = build_dataset(
            [records[i] for i in w.record_indices],
            structure_class=w.structure_class.value,
            max_seq_length=4_096,
        )
        assert datasets[w.id] == whole
    assert derive_workloads(stream, (), max_seq_length=4_096) == {}
    assert len(opened) == 2  # nothing to derive, nothing read


def test_an_agent_step_s_row_keeps_the_call_its_tool_result_answers() -> None:
    # A mid-loop step whose label depends on a tool result must show the student
    # the call that produced it, in both row formats.
    call = '{"arguments": {"order": 3}, "name": "lookup"}'
    record = replace(
        _rec(response="shipped"),
        messages=(
            Message("user", "where is order 3?"),
            Message("assistant", call),
            Message("tool", "status=shipped"),
        ),
    )
    (labeled,), _ = derive_rows([record], structure_class="enum_label", max_seq_length=4_096)
    assert f"<|assistant|>\n{call}\n<|tool|>\nstatus=shipped\n" in labeled["input"]
    (chat,), _ = derive_rows([record], structure_class="json_object", max_seq_length=4_096)
    assert [m["role"] for m in chat["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
