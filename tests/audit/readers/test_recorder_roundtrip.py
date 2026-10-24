"""The recorder's real output, read back: raw against norm, the round trip, media and the contract.

The fixtures ``recorder_*.jsonl`` are the lines ``dagnam_record`` wrote while the real ``openai`` and
``anthropic`` clients called a local mock server (``record/tests/gen_fixtures.py`` makes them), so
nothing here is a hand-built shape.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from dagnam_contracts.trace import parse_envelope
import pytest

from dagnam.audit import Message
from dagnam.audit.readers import read_traces
from dagnam.audit.readers.messages import effective_response
from dagnam.audit.readers.recorder import to_record
from dagnam.audit.record import TraceRecord
from dagnam.audit.scan import scan_export
from dagnam.audit.scan_report import write_scan

FIXTURES = Path(__file__).parents[1] / "fixtures"
CHAT = "recorder_openai_chat.jsonl"
BATCH = "openai_roundtrip_batch.jsonl"
RECORDER_FILES = sorted(path.name for path in FIXTURES.glob("recorder_*.jsonl"))
PAYLOADS = ("iVBORw0KGgo", "JVBERi0x", "UklGRiQA")  # the starts of the base64 the calls carried
EXPECTED_FILES = [
    "recorder_anthropic.jsonl",
    "recorder_anthropic_stream.jsonl",
    "recorder_media.jsonl",
    "recorder_mixed_bad.jsonl",
    "recorder_norm_only.jsonl",
    "recorder_openai_chat.jsonl",
    "recorder_openai_parse.jsonl",
    "recorder_openai_responses.jsonl",
    "recorder_openai_stream.jsonl",
    "recorder_other.jsonl",
]

COMPARED = (
    "model",
    "system",
    "messages",
    "response",
    "prompt_tokens",
    "completion_tokens",
    "cached_prompt_tokens",
    "has_media",
    "reasoning_only",
)
"""The record fields that must not depend on whether the reader began from raw or from norm.

``response_tool_calls`` is compared too, as the canonical JSON a student answers (raw keeps each
vendor's call object, norm the OpenAI shape without ids). ``reasoning_only`` is on the list
beyond the plan's nine fields, because a call that is only reasoning must not read as an empty
answer on one path and not the other."""

EXPECTED_DIFFERENCES: dict[tuple[str, int], dict[str, str]] = {
    ("recorder_anthropic.jsonl", 7): {
        "reasoning_only": (
            "A reply made only of thinking blocks is reasoning_only when read from raw; the norm"
            " has no field for it (its response is empty and it has no tool call), so it reads as"
            " an empty answer. Either way the call is counted and priced and gives no training row."
        ),
    },
}
"""``(file, 1-based line)`` -> ``{field: why raw and norm still differ}``. Nothing else may differ,
and an entry that no longer differs fails the agreement test, so the table cannot go stale."""


def load(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines()]


def both_forms() -> list[tuple[str, int, dict[str, Any]]]:
    """Every fixture line that carries raw bodies and a norm."""
    return [
        (name, number, row)
        for name in RECORDER_FILES
        for number, row in enumerate(load(name), 1)
        if "request" in row and "norm" in row
    ]


def differing_fields(row: dict[str, Any]) -> set[str]:
    """The compared fields on which the record from raw and the record from norm differ."""
    from_raw = to_record(row)
    from_norm = to_record(row | {"request": None, "response": None})
    if row["status"] != 200:
        assert from_raw is None
        assert from_norm is None
        return set()
    assert from_raw is not None
    assert from_norm is not None
    found = {name for name in COMPARED if getattr(from_raw, name) != getattr(from_norm, name)}
    if effective_response("", from_raw.response_tool_calls) != effective_response(
        "", from_norm.response_tool_calls
    ):
        found.add("response_tool_calls")
    return found


def test_the_fixture_files_are_the_ones_the_generator_writes() -> None:
    assert RECORDER_FILES == EXPECTED_FILES
    assert (FIXTURES / BATCH).is_file()


@pytest.mark.parametrize("name", RECORDER_FILES)
def test_the_contract_validator_accepts_every_recorded_line(name: str) -> None:
    lines = (FIXTURES / name).read_text(encoding="utf-8").splitlines()
    assert lines
    for line in lines:
        parse_envelope(line)


def check_line(
    name: str,
    number: int,
    row: dict[str, Any],
    table: dict[tuple[str, int], dict[str, str]] = EXPECTED_DIFFERENCES,
) -> None:
    """Fail unless the line differs on exactly the fields the table pins for it."""
    assert differing_fields(row) == set(table.get((name, number), {})), (name, number)


@pytest.mark.parametrize(
    ("name", "number", "row"),
    [pytest.param(*case, id=f"{case[0]}:{case[1]}") for case in both_forms()],
)
def test_raw_and_norm_agree_except_for_the_pinned_differences(
    name: str, number: int, row: dict[str, Any]
) -> None:
    check_line(name, number, row)


NORM_CHANGES: dict[str, Any] = {  # what to put in the norm so that this field no longer agrees
    "model": "other-model",
    "system": "another system",
    "messages": [{"role": "user", "content": "another prompt"}],
    "response": "another answer",
    "prompt_tokens": 1,
    "completion_tokens": 1,
    "cached_prompt_tokens": 1,
    "has_media": True,
}


@pytest.mark.parametrize(
    "field", [*(f for f in COMPARED if f != "reasoning_only"), "response_tool_calls"]
)
def test_a_difference_in_any_compared_field_is_noticed(field: str) -> None:
    row = load(CHAT)[0]
    norm = row["norm"]
    if field == "response_tool_calls":
        norm["tool_calls"] = [{"type": "function", "function": {"name": "x", "arguments": "{}"}}]
    else:
        norm[field] = NORM_CHANGES[field]
    assert differing_fields(row) == {field}


def test_the_table_fails_a_line_it_does_not_pin_and_one_it_pins_in_vain() -> None:
    thinking_only = load("recorder_anthropic.jsonl")[6]
    agreeing = load(CHAT)[0]
    check_line("recorder_anthropic.jsonl", 7, thinking_only)
    pin = {("recorder_anthropic.jsonl", 7): {"reasoning_only": "x"}}
    with pytest.raises(AssertionError):  # a difference nobody pinned
        check_line("recorder_anthropic.jsonl", 7, thinking_only, {})
    with pytest.raises(AssertionError):  # a pinned difference that is gone
        check_line("recorder_anthropic.jsonl", 7, agreeing, pin)
    with pytest.raises(AssertionError):  # pinned, but a different field differs
        check_line(
            "recorder_anthropic.jsonl",
            7,
            thinking_only,
            {("recorder_anthropic.jsonl", 7): {"model": "x"}},
        )


def test_every_pinned_difference_is_a_line_with_both_forms_and_has_a_reason() -> None:
    lines = {(name, number) for name, number, _ in both_forms()}
    assert set(EXPECTED_DIFFERENCES) <= lines
    for fields in EXPECTED_DIFFERENCES.values():
        assert fields
        assert all(len(reason) > 40 for reason in fields.values())
        assert set(fields) <= {*COMPARED, "response_tool_calls"}


def test_a_failed_call_is_skipped_on_both_paths() -> None:
    rows = load("recorder_mixed_bad.jsonl")
    assert [row["status"] for row in rows] == [200, 429, 200, 500, 200, 599, 529]
    for row in rows:
        assert differing_fields(row) == set()
    kept = [row for row in rows if row["status"] == 200]
    assert all(to_record(row) is not None for row in kept)
    assert not any(to_record(row) for row in rows if row["status"] != 200)


def test_a_norm_only_line_keeps_the_tool_call_of_an_earlier_turn() -> None:
    norm_only = load("recorder_norm_only.jsonl")[3]
    assert "request" not in norm_only
    record = to_record(norm_only)
    assert record is not None
    assert record.messages == (
        Message("user", "Where is order W-1001?"),
        Message("assistant", '{"arguments": {"order_id": "W-1001"}, "name": "lookup_order"}'),
        Message("tool", "shipped"),
        Message("user", "Thanks."),
    )


def test_tool_calls_cut_off_at_max_tokens_read_the_same_from_both_forms() -> None:
    row = load("recorder_anthropic_stream.jsonl")[3]
    assert row["response"]["stop_reason"] == "max_tokens"
    assert row["response"]["content"][0]["input"] == '{"city": "Pa'
    record = to_record(row)
    assert record is not None
    assert effective_response("", record.response_tool_calls) == (
        '{"arguments": "{\\"city\\": \\"Pa", "name": "weather"}'
    )
    assert differing_fields(row) == set()


def test_a_parse_call_keeps_its_schema_name_for_the_workload_key() -> None:
    chat, responses = load("recorder_openai_parse.jsonl")
    assert chat["request"]["response_format"]["name"] == "Ticket"
    assert responses["request"]["text_format"]["name"] == "Ticket"
    records = [to_record(chat), to_record(responses)]
    assert [r.signature for r in records if r] == ["schema=Ticket", "schema=Ticket"]
    assert "parsed" in chat["response"]["choices"][0]["message"]


def test_a_cached_anthropic_prompt_counts_every_token_and_the_cache_reads() -> None:
    row = load("recorder_anthropic.jsonl")[1]
    usage = row["response"]["usage"]
    assert (usage["input_tokens"], usage["cache_creation_input_tokens"]) == (12, 100)
    assert usage["cache_read_input_tokens"] == 900
    for record in (to_record(row), to_record(row | {"request": None, "response": None})):
        assert record is not None
        assert (record.prompt_tokens, record.cached_prompt_tokens) == (1012, 900)


def without_markers(value: Any) -> Any:
    """``value`` with each omission marker replaced by an empty text part."""
    if isinstance(value, dict):
        if value.get("type") == "media_omitted":
            return {"type": "text", "text": ""}
        return {key: without_markers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [without_markers(item) for item in value]
    return value


def test_media_is_marked_by_the_omission_marker_alone() -> None:
    rows = load("recorder_media.jsonl")
    assert len(rows) == 5
    for row in rows:
        text = json.dumps(row)
        assert "media_omitted" in json.dumps(row["request"])
        assert not any(payload in text for payload in PAYLOADS)  # only the size was kept
        raw_only = {key: value for key, value in row.items() if key not in ("norm", "norm_v")}
        marked = to_record(raw_only)
        unmarked = to_record(raw_only | {"request": without_markers(row["request"])})
        assert marked is not None
        assert unmarked is not None
        assert marked.has_media
        assert not unmarked.has_media
        from_norm = to_record(row | {"request": None, "response": None})
        assert from_norm is not None
        assert from_norm.has_media


def drop_ids(records: list[TraceRecord]) -> list[TraceRecord]:
    return [replace(record, trace_id="") for record in records]


def test_the_recorder_file_and_the_batch_export_read_to_the_same_records() -> None:
    recorded = drop_ids(list(read_traces(FIXTURES / CHAT, source="dagnam")[0]))
    exported = drop_ids(list(read_traces(FIXTURES / BATCH, source="openai")[0]))
    assert len(recorded) == len(exported) == 13
    for number, (mine, theirs) in enumerate(zip(recorded, exported, strict=True), 1):
        assert mine == theirs, number


def scan_json(tmp_path: Path, name: str, source: Any) -> dict[str, Any]:
    result = scan_export(FIXTURES / name, source=source)
    write_scan(
        tmp_path,
        result.workloads,
        result.datasets,
        source=result.source,
        window=result.window,
        price_table=result.table,
    )
    return json.loads((tmp_path / "scan-report.json").read_text(encoding="utf-8"))


def test_scanning_the_recorder_file_equals_scanning_the_same_calls_as_an_openai_batch(
    tmp_path: Path,
) -> None:
    recorded = scan_json(tmp_path / "recorder", CHAT, "dagnam")
    exported = scan_json(tmp_path / "batch", BATCH, "openai")
    # Only what must differ: when the report was written, and the name of the format read.
    assert (recorded.pop("source"), exported.pop("source")) == ("dagnam", "openai")
    recorded.pop("generated_at")
    exported.pop("generated_at")
    assert recorded == exported
    assert recorded["totals"]["calls"] == 13
