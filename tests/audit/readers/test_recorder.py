"""The local recorder's trace file (envelope v1) -> TraceRecord."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import pytest

from dagnam.audit.readers.base import (
    MalformedExportError,
    MalformedRowError,
    UnsupportedExportError,
    read_records,
)
from dagnam.audit.readers.messages import effective_response
from dagnam.audit.readers.recorder import READER, REQUIRED_FIELDS, to_record
from dagnam.audit.record import Message, TraceRecord

AGREEING = (
    "recorder_openai_chat.jsonl",
    "recorder_openai_responses.jsonl",
    "recorder_anthropic.jsonl",
)


def read(path: Path) -> list[TraceRecord]:
    records, _ = read_records(path, "dagnam", READER)
    return list(records)


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_the_reader_needs_the_envelope_header_fields() -> None:
    assert REQUIRED_FIELDS == ("v", "trace_id", "ts")
    assert READER.required_fields == REQUIRED_FIELDS
    assert READER.to_record is to_record


def test_openai_chat_line_reads_through_the_openai_parser(fixtures_dir: Path) -> None:
    out = read(fixtures_dir / "recorder_openai_chat.jsonl")
    assert len(out) == 13
    assert all(
        r.trace_id.startswith("rec_chat_") for r in out
    )  # the envelope's id, not the completion's
    first = out[0]
    assert first.ts == datetime(2026, 10, 12, 0, 0, 1, tzinfo=UTC)  # the envelope's, not `created`
    assert first.latency_ms == 301
    assert (first.model, first.response) == ("gpt-4o-mini-2024-07-18", "returns")
    assert first.system == "You are a support triage assistant. Reply with one intent label."
    assert (first.prompt_tokens, first.completion_tokens, first.cached_prompt_tokens) == (40, 2, 0)
    assert out[1].cached_prompt_tokens == 1920
    assert effective_response(out[2].response, out[2].response_tool_calls) == (
        '{"arguments": {"order_id": "W-1001"}, "name": "lookup_order"}'
    )
    assert out[4].system == "Be brief."
    assert [m.role for m in out[4].messages] == ["user", "assistant", "user"]


def test_openai_responses_line_reads_through_the_openai_parser(fixtures_dir: Path) -> None:
    text, call, media, *_ = read(fixtures_dir / "recorder_openai_responses.jsonl")
    assert (text.system, text.response, text.prompt_tokens) == (
        "Reply with one urgency label.",
        "high",
        30,
    )
    assert text.messages == (Message("user", "Refund me today!"),)
    assert effective_response(call.response, call.response_tool_calls) == (
        '{"arguments": {"priority": "high"}, "name": "open_ticket"}'
    )
    assert media.has_media
    assert not text.has_media


def test_anthropic_line_maps_system_blocks_and_usage(fixtures_dir: Path) -> None:
    plain, cached, thinking, tool, document, *_ = read(fixtures_dir / "recorder_anthropic.jsonl")
    assert plain.system == "You are a triage bot. Answer in one word."
    assert plain.response == "returns"
    assert plain.model == "claude-sonnet-4-5-20250929"
    assert (plain.prompt_tokens, plain.completion_tokens, plain.cached_prompt_tokens) == (25, 4, 0)
    assert (plain.trace_id, plain.latency_ms, plain.signature) == ("rec_anth_1", 501, None)
    # Anthropic's input_tokens excludes the cache; the record counts every prompt token.
    assert (cached.prompt_tokens, cached.cached_prompt_tokens) == (1012, 900)
    assert thinking.response == "returns"
    assert not thinking.reasoning_only
    assert tool.response == ""
    assert tool.response_tool_calls[0]["name"] == "lookup_order"
    assert document.system == "Plain string system."
    assert document.has_media


def test_a_media_omitted_part_marks_the_call_and_reads_as_no_text(fixtures_dir: Path) -> None:
    media = read(fixtures_dir / "recorder_openai_chat.jsonl")[3]
    assert media.has_media
    assert media.messages == (Message("user", "What is wrong here?"),)
    assert not any(r.has_media for r in read(fixtures_dir / "recorder_openai_chat.jsonl")[:3])


def test_non_200_is_skipped_not_malformed(fixtures_dir: Path) -> None:
    records, stats = read_records(fixtures_dir / "recorder_mixed_bad.jsonl", "dagnam", READER)
    assert len(list(records)) == 3
    assert (stats.rows_seen, stats.rows_kept, stats.rows_malformed) == (7, 3, 0)


def test_norm_only_line_is_read_without_raw(fixtures_dir: Path) -> None:
    chat, anthropic, *_ = read(fixtures_dir / "recorder_norm_only.jsonl")
    assert (chat.response, chat.model, chat.system) == (
        "shipping",
        "gpt-4o-mini-2024-07-18",
        "Label the intent.",
    )
    assert chat.messages == (Message("user", "Where is my order?"),)
    assert (chat.prompt_tokens, chat.completion_tokens) == (20, 2)
    assert chat.ts == datetime(2026, 10, 12, 0, 5, 1, tzinfo=UTC)
    assert chat.latency_ms == 201
    assert anthropic.response_tool_calls[0]["function"]["name"] == "close_account"
    assert (
        effective_response("", anthropic.response_tool_calls)
        == '{"arguments": {}, "name": "close_account"}'
    )


def other_with_norm(fixtures_dir: Path) -> dict[str, Any]:
    """A ``provider: "other"`` line that carries a norm.

    The recorder writes no norm for ``other`` (it knows no such shape), so this is the recorded
    raw-only line with a norm added by hand, for a recorder that may one day write one.
    """
    row = lines(fixtures_dir / "recorder_other.jsonl")[0]
    norm = {
        "model": "my-local-model",
        "system": None,
        "messages": [{"role": "user", "content": "Where is my order?"}],
        "response": "shipping",
        "tool_calls": [],
        "prompt_tokens": 5,
        "completion_tokens": 1,
        "cached_prompt_tokens": 0,
        "has_media": False,
    }
    return row | {"norm": norm, "norm_v": 1}


def test_norm_wins_over_raw_for_the_other_provider(fixtures_dir: Path) -> None:
    other = other_with_norm(fixtures_dir)
    assert other["request"]
    assert other["response"]  # raw is there, and has no rules the reader knows
    record = to_record(other)
    assert record is not None
    assert (record.model, record.system, record.response) == ("my-local-model", None, "shipping")
    assert record.latency_ms == 51.0


def test_other_provider_without_norm_is_malformed(fixtures_dir: Path) -> None:
    other, openai_custom = lines(fixtures_dir / "recorder_other.jsonl")  # record_call: no norm
    assert "norm" not in other
    with pytest.raises(MalformedRowError, match="no norm"):
        to_record(other)
    # The same call handed over as OpenAI's has raw bodies the OpenAI parser reads.
    record = to_record(openai_custom)
    assert record is not None
    assert (record.messages, record.response) == ((Message("user", "x"),), "y")


def test_norm_with_no_user_turn_is_malformed(fixtures_dir: Path) -> None:
    row = other_with_norm(fixtures_dir)
    row["norm"]["messages"] = [{"role": "assistant", "content": "hi"}]
    with pytest.raises(MalformedRowError, match="no user turn"):
        to_record(row)


def test_norm_reply_is_cleaned_like_a_raw_one(fixtures_dir: Path) -> None:
    row = other_with_norm(fixtures_dir)
    row["norm"]["response"] = "<think>hmm</think>shipping"
    record = to_record(row)
    assert record is not None
    assert record.response == "shipping"


def test_an_openai_body_that_reports_a_failure_status_is_skipped(fixtures_dir: Path) -> None:
    row = lines(fixtures_dir / "recorder_openai_chat.jsonl")[0]
    row["response"]["status_code"] = 500
    assert to_record(row) is None


def test_anthropic_without_a_model_is_unknown(fixtures_dir: Path) -> None:
    row = lines(fixtures_dir / "recorder_anthropic.jsonl")[0]
    del row["request"]["model"], row["response"]["model"]
    record = to_record(row)
    assert record is not None
    assert record.model == "unknown"


def test_anthropic_falls_back_to_the_request_model(fixtures_dir: Path) -> None:
    row = lines(fixtures_dir / "recorder_anthropic.jsonl")[0]
    del row["response"]["model"]
    record = to_record(row)
    assert record is not None
    assert record.model == "claude-sonnet-4-5"


AGREEING_FIELDS = (
    "model",
    "system",
    "messages",
    "response",
    "prompt_tokens",
    "completion_tokens",
    "cached_prompt_tokens",
    "has_media",
    "trace_id",
    "ts",
    "latency_ms",
)


def assert_raw_and_norm_agree(row: dict[str, Any]) -> None:
    trace_id = row["trace_id"]
    from_raw = to_record(row)
    from_norm = to_record(row | {"request": None, "response": None})
    assert from_raw is not None
    assert from_norm is not None
    for f in AGREEING_FIELDS:
        assert getattr(from_raw, f) == getattr(from_norm, f), (trace_id, f)
    # Raw keeps each vendor's call object (an id, Anthropic's tool_use block); the norm is the
    # OpenAI shape without ids. Both are judged and trained as the one canonical JSON.
    assert effective_response(
        from_raw.response, from_raw.response_tool_calls
    ) == effective_response(from_norm.response, from_norm.response_tool_calls), trace_id


@pytest.mark.parametrize("name", AGREEING)
def test_raw_and_norm_agree(fixtures_dir: Path, name: str) -> None:
    for row in lines(fixtures_dir / name):
        assert_raw_and_norm_agree(row)


@pytest.mark.parametrize("name", AGREEING[:2])
def test_an_openai_body_without_an_id_or_creation_time_still_reads(
    fixtures_dir: Path, name: str
) -> None:
    # The envelope has both; the completion's own are never used, so their absence is no fault.
    for row in lines(fixtures_dir / name):
        for key in ("id", "created", "created_at"):
            row["response"].pop(key, None)
        assert_raw_and_norm_agree(row)
        record = to_record(row)
        assert record is not None
        assert record.trace_id == row["trace_id"]


def test_a_null_openai_id_falls_back_to_the_envelopes(fixtures_dir: Path) -> None:
    row = lines(fixtures_dir / "recorder_openai_chat.jsonl")[0]
    row["response"]["id"] = None
    assert_raw_and_norm_agree(row)


def test_the_completions_creation_time_is_read_but_never_wins(fixtures_dir: Path) -> None:
    row = lines(fixtures_dir / "recorder_openai_responses.jsonl")[0]
    assert "created_at" in row["response"]
    record = to_record(row)
    assert record is not None
    assert record.ts == datetime(2026, 10, 12, 0, 1, 41, tzinfo=UTC)


def test_raw_wins_over_a_norm_that_differs(fixtures_dir: Path) -> None:
    for name in AGREEING:
        for row in lines(fixtures_dir / name):
            truth = to_record(row)
            row["norm"] |= {
                "model": "norm-model",
                "system": "norm system",
                "messages": [{"role": "user", "content": "norm prompt"}],
                "response": "norm reply",
                "tool_calls": [],
                "prompt_tokens": 999,
            }
            assert to_record(row) == truth


def test_anthropic_without_messages_is_malformed(fixtures_dir: Path) -> None:
    row = lines(fixtures_dir / "recorder_anthropic.jsonl")[0]
    del row["request"]["messages"]
    with pytest.raises(MalformedRowError, match="missing messages"):
        to_record(row)


def test_anthropic_without_content_is_malformed(fixtures_dir: Path) -> None:
    row = lines(fixtures_dir / "recorder_anthropic.jsonl")[0]
    del row["response"]["content"]
    with pytest.raises(MalformedRowError, match="missing content"):
        to_record(row)


def test_anthropic_with_empty_content_is_still_a_record(fixtures_dir: Path) -> None:
    row = lines(fixtures_dir / "recorder_anthropic.jsonl")[0]
    row["response"]["content"] = []
    record = to_record(row)
    assert record is not None
    assert record.response == ""


def test_a_newer_envelope_version_counts_as_malformed() -> None:
    with pytest.raises(MalformedRowError, match="v=2"):
        to_record({"v": 2, "trace_id": "x", "ts": "2026-10-12T00:00:00Z"})


def test_an_invalid_envelope_is_a_value_error_the_reader_counts() -> None:
    with pytest.raises(ValueError, match="provider"):
        to_record({"v": 1, "trace_id": "x", "ts": "2026-10-12T00:00:00Z"})


def test_a_few_newer_lines_are_counted_and_the_rest_read(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    good = (fixtures_dir / "recorder_openai_chat.jsonl").read_text(encoding="utf-8").splitlines()
    newer = json.dumps({"v": 2, "trace_id": "x", "ts": "2026-10-12T00:00:00Z"})
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join([*good * 5, newer]) + "\n", encoding="utf-8")
    records, stats = read_records(path, "dagnam", READER)
    assert len(list(records)) == 65
    assert (stats.rows_kept, stats.rows_malformed, stats.first_malformed) == (65, 1, (65,))


NEWER = json.dumps({"v": 2, "trace_id": "x", "ts": "2026-10-12T00:00:00Z"})


def test_a_file_of_only_newer_lines_is_an_unsupported_export(tmp_path: Path) -> None:
    path = tmp_path / "newer.jsonl"
    path.write_text(NEWER + "\n" + NEWER + "\n", encoding="utf-8")
    with pytest.raises(UnsupportedExportError, match=r"newer recorder.*v=2"):
        read(path)


def test_a_file_whose_first_data_line_is_newer_is_an_unsupported_export(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    good = (fixtures_dir / "recorder_openai_chat.jsonl").read_text(encoding="utf-8")
    path = tmp_path / "first-newer.jsonl"
    path.write_text("\n" + NEWER + "\n" + good, encoding="utf-8")  # a blank line is no row
    with pytest.raises(UnsupportedExportError, match="newer recorder"):
        read(path)


def test_a_newer_norm_version_on_the_first_line_is_an_unsupported_export(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    row = lines(fixtures_dir / "recorder_norm_only.jsonl")[0]
    row["norm_v"] = 2
    path = tmp_path / "norm-newer.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(UnsupportedExportError, match="norm_v=2"):
        read(path)


def test_a_first_line_that_is_only_invalid_leaves_the_judgement_to_the_malformed_share(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(
        json.dumps({"v": 1, "trace_id": "x", "ts": "2026-10-12T00:00:00Z"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(MalformedExportError):
        read(path)


def test_a_file_that_is_not_the_recorders_is_unsupported(fixtures_dir: Path) -> None:
    with pytest.raises(UnsupportedExportError, match="v, trace_id, ts"):
        read(fixtures_dir / "openai_sample.jsonl")


def test_a_norm_assistant_turn_is_read_like_a_raw_one(fixtures_dir: Path) -> None:
    row = lines(fixtures_dir / "recorder_norm_only.jsonl")[0]
    row["norm"]["messages"] = [
        {"role": "user", "content": "Where is it?"},
        {"role": "assistant", "content": "<think>status</think>Sure, one moment."},
        {"role": "user", "content": "<think>kept</think>Well?"},
    ]
    record = to_record(row)
    assert record is not None
    assert [m.content for m in record.messages] == [
        "Where is it?",
        "Sure, one moment.",
        "<think>kept</think>Well?",
    ]
