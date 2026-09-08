"""Derived training rows: format by structure class, truncation, dedup, the pipeline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from dagnam_contracts.prompts import render_chat_prompt
import pytest

from dagnam.audit import Message, TraceRecord, read_traces
from dagnam.audit.derive import (
    FORMAT_BY_STRUCTURE,
    DedupStats,
    DeriveStats,
    build_dataset,
    dedup_rows,
    derive_rows,
    normalize_label,
)

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
