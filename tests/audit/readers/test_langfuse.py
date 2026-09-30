"""Langfuse observation export -> TraceRecord."""

from __future__ import annotations

from collections import Counter
from datetime import UTC
import json
from pathlib import Path

import pytest

from dagnam.audit import Message, TraceRecord, read_traces
from dagnam.audit.derive import build_dataset
from dagnam.audit.prices import PriceTable
from dagnam.audit.readers import langfuse

# Row counts of ``fixtures/langfuse_sample.jsonl`` (see fixtures/README.md).
LINES = 15
RECORDS = 12
PER_SESSION = 4
SESSIONS = 3


def test_reader_yields_generations(fixtures_dir: Path) -> None:
    records, stats = read_traces(fixtures_dir / "langfuse_sample.jsonl", source="langfuse")
    rows = list(records)

    assert len(rows) == RECORDS
    assert (stats.rows_seen, stats.rows_kept, stats.rows_malformed) == (LINES, RECORDS, 0)
    assert stats.first_malformed == ()
    for row in rows:
        assert row.system
        assert row.messages
        assert all(m.role == "user" for m in row.messages)
        assert row.response
        assert row.latency_ms > 0
        assert row.prompt_tokens > 0
        assert row.ts.tzinfo is UTC
        assert row.model == "gpt-4o-mini"
        assert row.workload_hint in {"intent", "urgency", "extract", "reply"}
    assert sum(r.cost_usd is not None for r in rows) / len(rows) >= 0.99
    by_session = Counter(r.session_id for r in rows)
    assert len(by_session) == SESSIONS
    assert set(by_session.values()) == {PER_SESSION}
    assert len({r.trace_id for r in rows}) == RECORDS


def test_workload_hint_is_none_without_metadata(fixtures_dir: Path, tmp_path: Path) -> None:
    stripped = tmp_path / "stripped.jsonl"
    with (fixtures_dir / "langfuse_sample.jsonl").open() as src, stripped.open("w") as dst:
        for line in src:
            row = json.loads(line)
            row.pop("metadata", None)
            dst.write(json.dumps(row) + "\n")

    records, _ = read_traces(stripped, source="langfuse")
    assert {r.workload_hint for r in records} == {None}


def _generation() -> dict[str, object]:
    return {
        "id": "gen-1",
        "traceId": "trace-1",
        "type": "GENERATION",
        "startTime": "2026-08-01T09:00:00.000Z",
        "endTime": "2026-08-01T09:00:00.250Z",
        "model": "gpt-4o-mini",
        "input": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        "output": {"role": "assistant", "content": "hello"},
        "usage": {"input": 5, "output": 1},
        "calculatedTotalCost": 0.001,
    }


def test_skips_non_generation_and_empty_output() -> None:
    assert langfuse.to_record({**_generation(), "type": "SPAN"}) is None
    assert langfuse.to_record({**_generation(), "output": None}) is None


def test_legacy_token_and_latency_spellings() -> None:
    row = _generation()
    del row["usage"], row["endTime"]
    row.update({"promptTokens": 7, "completionTokens": 2, "latency": 1.5, "sessionId": "s-9"})

    record = langfuse.to_record(row)

    assert record is not None
    assert (record.prompt_tokens, record.completion_tokens) == (7, 2)
    assert record.latency_ms == 1500.0
    assert record.session_id == "s-9"


def test_usage_details_and_cost_details_spellings() -> None:
    row = _generation()
    del row["usage"], row["calculatedTotalCost"]
    row.update({"usageDetails": {"input": 9, "output": 3}, "costDetails": {"total": 0.5}})

    record = langfuse.to_record(row)

    assert record is not None
    assert (record.prompt_tokens, record.completion_tokens, record.cost_usd) == (9, 3, 0.5)


def test_string_output_and_tool_calls() -> None:
    plain = langfuse.to_record({**_generation(), "output": "just text"})
    assert plain is not None
    assert plain.response == "just text"
    assert plain.response_tool_calls == ()

    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    tooled = langfuse.to_record(
        {**_generation(), "output": {"role": "assistant", "content": None, "tool_calls": [call]}}
    )
    assert tooled is not None
    assert tooled.response == ""
    assert tooled.response_tool_calls == (call,)


def test_missing_tokens_and_cost_are_defaulted_not_malformed() -> None:
    row = _generation()
    del row["usage"], row["calculatedTotalCost"]

    record = langfuse.to_record(row)

    assert record is not None
    assert (record.prompt_tokens, record.completion_tokens) == (0, 0)
    assert record.cost_usd is None
    assert record.outcome is None


def test_outcome_from_metadata() -> None:
    record = langfuse.to_record({**_generation(), "metadata": {"outcome": "0.5"}})
    assert record is not None
    assert record.outcome == 0.5


def test_anthropic_usage_shape_counts_cached_prompt_tokens_and_prices() -> None:
    row = _generation()
    del row["usage"], row["calculatedTotalCost"]
    row.update(
        {
            "model": "anthropic/claude-sonnet-4-5",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 40,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 100,
            },
        }
    )

    record = langfuse.to_record(row)

    assert record is not None
    assert (record.prompt_tokens, record.completion_tokens) == (1000, 40)
    assert PriceTable.load(None).cost(
        record.model, record.prompt_tokens, record.completion_tokens
    ) == pytest.approx(0.0036)


def test_gemini_usage_shape_is_read_and_priced() -> None:
    row = _generation()
    del row["usage"]
    row.update(
        {
            "model": "models/gemini-2.5-flash",
            "usageMetadata": {"promptTokenCount": 1_000, "candidatesTokenCount": 200},
        }
    )

    record = langfuse.to_record(row)

    assert record is not None
    assert (record.prompt_tokens, record.completion_tokens) == (1000, 200)
    assert PriceTable.load(None).cost(
        record.model, record.prompt_tokens, record.completion_tokens
    ) == pytest.approx(0.0008)


def test_snake_case_gemini_usage_details_are_read() -> None:
    row = _generation()
    del row["usage"]
    row["usageDetails"] = {"prompt_token_count": 12, "candidates_token_count": 3}

    record = langfuse.to_record(row)

    assert record is not None
    assert (record.prompt_tokens, record.completion_tokens) == (12, 3)


def test_anthropic_shaped_generation_keeps_its_system_and_flattens_its_blocks() -> None:
    # The Anthropic SDK's ``system`` kwarg sits beside ``messages``, and the reply is
    # content blocks; the LangSmith reader already reads both shapes.
    row = {
        **_generation(),
        "model": "claude-haiku-4-5",
        "input": {
            "system": "Classify the ticket intent.",
            "messages": [{"role": "user", "content": "where is my order"}],
        },
        "output": {"role": "assistant", "content": [{"type": "text", "text": "track_order"}]},
    }

    record = langfuse.to_record(row)

    assert record is not None
    assert record.system == "Classify the ticket intent."
    assert record.response == "track_order"


def test_cached_prompt_tokens_are_read_from_the_anthropic_usage() -> None:
    row = _generation()
    row["usage"] = {"input_tokens": 100, "output_tokens": 4, "cache_read_input_tokens": 800}

    record = langfuse.to_record(row)

    assert record is not None
    assert (record.prompt_tokens, record.cached_prompt_tokens) == (900, 800)


def test_a_responses_generation_with_tools_is_read() -> None:
    # The Langfuse OpenAI integration logs a Responses call with tools as
    # ``{input, tools}`` and a ``function_call`` item: unstructured, empty reply.
    call = {"type": "function_call", "call_id": "c0", "name": "route", "arguments": '{"team": "a"}'}
    row = {
        **_generation(),
        "input": {
            "input": [
                {"role": "system", "content": "Route the ticket to a team."},
                {"role": "user", "content": "ticket 0"},
            ],
            "tools": [{"type": "function", "name": "route", "parameters": {}}],
        },
        "output": call,
    }
    record = langfuse.to_record(row)
    assert record is not None
    assert record.system == "Route the ticket to a team."
    assert record.response_tool_calls == (call,)
    assert record.signature is None  # the offered tools never key a workload (B2-2)


def test_an_agent_step_keeps_the_calls_its_tool_results_answer() -> None:
    step = [
        {"role": "system", "content": "You are a research agent. Use tools, then answer."},
        {"role": "user", "content": "question 1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "search", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c", "content": "tool result 0 for 1"},
    ]
    record = langfuse.to_record({**_generation(), "input": step})
    assert record is not None
    assert [m.role for m in record.messages] == ["user", "assistant", "tool"]
    assert record.messages[1].content == '{"arguments": {}, "name": "search"}'


def test_an_image_part_marks_the_call_multimodal() -> None:
    content = [
        {"type": "text", "text": "What category?"},
        {"type": "image_url", "image_url": {"url": "https://img/1.png"}},
    ]
    row = {**_generation(), "input": [{"role": "user", "content": content}]}
    record = langfuse.to_record(row)
    assert record is not None
    assert record.has_media
    plain = langfuse.to_record(_generation())
    assert plain is not None
    assert not plain.has_media


PROMPT = [
    {"role": "system", "content": "Label urgency: low/high."},
    {"role": "user", "content": "ticket"},
]
REPLY = {"role": "assistant", "content": "low"}


def _blob_v2(i: int) -> dict[str, object]:
    """An ``observations_v2/`` blob-export row: snake_case, the I/O as JSON strings."""
    return {
        "id": f"o{i}",
        "type": "GENERATION",
        "trace_id": "t",
        "start_time": "2026-09-01T00:00:00Z",
        "end_time": "2026-09-01T00:00:01Z",
        "provided_model_name": "gpt-4o-mini",
        "input": json.dumps(PROMPT),
        "output": json.dumps(REPLY),
        "usage_details": {"input": 10, "output": 1, "input_cached_tokens": 30},
        "cost_details": {"total": 0.002},
        "prompt_name": "urgency",
        "prompt_version": 3,
    }


def _assert_urgency(record: object) -> None:
    assert isinstance(record, TraceRecord)
    assert record.system == "Label urgency: low/high."
    assert record.messages == (Message("user", "ticket"),)
    assert record.response == "low"


def test_the_blob_export_shape_is_read(tmp_path: Path) -> None:
    # observations_v2/ (the only blob export after 2026-11-16): "missing startTime".
    path = tmp_path / "observations_v2.jsonl"
    path.write_text("".join(json.dumps(_blob_v2(i)) + "\n" for i in range(5)))

    records, stats = read_traces(path, source="langfuse")
    first, *_ = list(records)

    assert stats.rows_malformed == 0
    _assert_urgency(first)
    assert first.model == "gpt-4o-mini"
    assert first.session_id == "t"
    assert first.latency_ms == 1000.0
    assert first.cost_usd == 0.002
    # Langfuse usage buckets are exclusive: ``input`` excludes the cached tokens.
    assert (first.prompt_tokens, first.cached_prompt_tokens, first.completion_tokens) == (40, 30, 1)
    assert first.workload_hint == "urgency"  # the registered prompt names the workload (N11)


AI_MESSAGE = {
    "content": "billing",
    "additional_kwargs": {},
    "response_metadata": {"model_name": "gpt-4o-mini"},
    "type": "ai",
    "name": None,
    "id": "run-123",
    "tool_calls": [],
}
"""A LangChain ``AIMessage`` dump, as Langfuse stores one an ``@observe`` generation returned."""


@pytest.mark.parametrize(
    ("output", "response"),
    [
        # RR-3: role-less, and read whole as JSON -- its per-call ``id`` made every
        # answer distinct; it is a reply because it carries ``content``.
        (AI_MESSAGE, "billing"),
        # Answers that only look like replies stay the model's JSON answer.
        ({"output": ["a", "b"]}, json.dumps({"output": ["a", "b"]})),
        ({"name": "Bob", "role": "admin"}, json.dumps({"name": "Bob", "role": "admin"})),
        (
            {"type": "message", "priority": "high"},
            json.dumps({"type": "message", "priority": "high"}),
        ),
    ],
    ids=["langchain-ai-message", "output-list-answer", "role-field-answer", "type-field-answer"],
)
def test_a_reply_is_what_carries_a_reply_and_the_rest_is_the_answer(
    output: dict[str, object], response: str
) -> None:
    for exported in (output, json.dumps(output)):
        record = langfuse.to_record({**_generation(), "output": exported})
        assert record is not None
        assert record.response == response


def test_priority_buckets_are_not_added_twice() -> None:
    # m2: Langfuse's own SDK marks ``input_priority*`` / ``output_priority*`` as not
    # exclusive (its CallbackHandler never subtracts them): they count tokens the
    # other buckets already hold. 1,800 prompt tokens were read for 900.
    usage = {
        "input": 100,
        "input_cache_read": 800,
        "input_priority": 900,
        "input_priority_cache_read": 800,
        "output": 5,
        "output_priority": 5,
        "input_cost": 0.1,
    }
    record = langfuse.to_record({**_generation(), "usage_details": usage, "usage": None})
    assert record is not None
    assert (record.prompt_tokens, record.cached_prompt_tokens, record.completion_tokens) == (
        900,
        800,
        5,
    )


def test_the_ui_export_json_strings_are_decoded(tmp_path: Path) -> None:
    # The UI's CSV and JSON exports store I/O as JSON strings: the whole prompt was one
    # user turn with no system prompt, and the reply the JSON of the message dict.
    row = {
        **_generation(),
        "input": json.dumps(PROMPT),
        "output": json.dumps(REPLY),
        "usage": json.dumps({"input": 5, "output": 1}),
        "metadata": json.dumps({"workload": "urgency"}),
    }
    record = langfuse.to_record(row)
    _assert_urgency(record)
    assert record is not None
    assert (record.prompt_tokens, record.workload_hint) == (5, "urgency")
    array = tmp_path / "ui_export.json"
    array.write_text(json.dumps([row]))
    (read,) = list(read_traces(array, source="langfuse")[0])
    _assert_urgency(read)
    # A reply that is itself JSON text stays the model's answer, not a message.
    answer = langfuse.to_record({**_generation(), "output": json.dumps({"a": 1})})
    assert answer is not None
    assert answer.response == '{"a": 1}'
    semconv = [{"role": "assistant", "parts": [{"type": "text", "content": "high"}]}]
    parts = langfuse.to_record({**_generation(), "output": json.dumps(semconv)})
    assert parts is not None
    assert parts.response == "high"


def test_a_json_answer_with_a_type_key_is_the_reply_not_a_message(tmp_path: Path) -> None:
    # B2-1: an answer like {"type": "refund", ...} was taken for a message with no
    # content, so every reply read "", the workload derived no rows at all and its
    # verdict fell to too_few_samples. Blob/UI exports carry it as a JSON string, the
    # API export as an object; both are the model's answer, as JSON text.
    answers = [{"type": "refund", "amount": 12}, {"type": "exchange", "amount": 0}]
    path = tmp_path / "observations_v2.jsonl"
    path.write_text(
        "".join(
            json.dumps({**_blob_v2(i), "id": f"o{i}", "output": json.dumps(answers[i % 2])}) + "\n"
            for i in range(4)
        )
    )
    records = list(read_traces(path, source="langfuse")[0])
    assert [r.response for r in records] == [json.dumps(answers[i % 2]) for i in range(4)]
    dataset = build_dataset(records, structure_class="json_object", max_seq_length=4_096)
    assert dataset.stats["derive"] == {
        "rows": 4,
        "skipped": 0,
        "truncated": 0,
        "truncation_rate": 0.0,
    }
    as_object = langfuse.to_record({**_generation(), "output": answers[0]})
    assert as_object is not None
    assert json.loads(as_object.response) == answers[0]
    # A typed Responses item and a Responses object are still replies to unwrap.
    message = {"type": "message", "content": [{"type": "output_text", "text": "low"}]}
    for output in (message, {"object": "response", "output": [message]}):
        record = langfuse.to_record({**_generation(), "output": json.dumps(output)})
        assert record is not None
        assert record.response == "low"


def test_a_prompt_that_only_looks_like_json_stays_a_prompt() -> None:
    record = langfuse.to_record({**_generation(), "input": "{not json, a plain prompt"})
    assert record is not None
    assert record.messages[0].content == "{not json, a plain prompt"


def test_a_text_outcome_in_the_metadata_is_ignored() -> None:
    record = langfuse.to_record({**_generation(), "metadata": {"outcome": "resolved"}})
    assert record is not None
    assert record.outcome is None
