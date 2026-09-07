"""LangSmith run export -> TraceRecord (JSONL and Parquet)."""

from __future__ import annotations

from collections import Counter
from datetime import UTC
import importlib.util
from pathlib import Path

import polars as pl
import pytest

from dagnam.audit import Message, read_traces
from dagnam.audit.prices import PriceTable
from dagnam.audit.readers import langsmith

# Row counts of ``fixtures/langsmith_sample.jsonl`` (see fixtures/README.md).
LINES = 15
RECORDS = 12
PER_SESSION = 4
SESSIONS = 3


@pytest.fixture
def langsmith_parquet(fixtures_dir: Path, tmp_path: Path) -> Path:
    target = tmp_path / "langsmith_sample.parquet"
    pl.read_ndjson(fixtures_dir / "langsmith_sample.jsonl", infer_schema_length=None).write_parquet(
        target
    )
    return target


def _assert_real_shape(path: Path) -> None:
    records, stats = read_traces(path, source="langsmith")
    rows = list(records)

    assert len(rows) == RECORDS
    assert (stats.rows_seen, stats.rows_kept, stats.rows_malformed) == (LINES, RECORDS, 0)
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


def test_reader_yields_llm_runs_from_jsonl(fixtures_dir: Path) -> None:
    _assert_real_shape(fixtures_dir / "langsmith_sample.jsonl")


def test_reader_yields_llm_runs_from_parquet_without_pyarrow(langsmith_parquet: Path) -> None:
    assert importlib.util.find_spec("pyarrow") is None
    _assert_real_shape(langsmith_parquet)


def _llm_run() -> dict[str, object]:
    return {
        "id": "run-1",
        "trace_id": "trace-1",
        "session_id": "project-uuid",
        "run_type": "llm",
        "start_time": "2026-08-01T09:00:00.000000",
        "end_time": "2026-08-01T09:00:00.400000",
        "inputs": {
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        },
        "outputs": {"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
        "extra": {"invocation_params": {"model": "gpt-4o-mini"}, "metadata": {}},
        "prompt_tokens": 5,
        "completion_tokens": 1,
        "total_cost": 0.001,
    }


def test_skips_non_llm_runs() -> None:
    assert langsmith.to_record({**_llm_run(), "run_type": "chain"}) is None


def test_session_falls_back_to_trace_id_when_no_thread_metadata() -> None:
    record = langsmith.to_record(_llm_run())
    assert record is not None
    assert record.session_id == "trace-1"
    assert record.latency_ms == 400.0

    threaded = langsmith.to_record({**_llm_run(), "extra": {"metadata": {"thread_id": "th-1"}}})
    assert threaded is not None
    assert threaded.session_id == "th-1"
    assert threaded.model == "unknown"


def test_langchain_serialized_messages_and_generations() -> None:
    def lc(kind: str, content: str) -> dict[str, object]:
        return {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain", "schema", kind],
            "kwargs": {"content": content},
        }

    row = {
        **_llm_run(),
        "inputs": {
            "messages": [
                [lc("SystemMessage", "sys"), lc("HumanMessage", "hi"), lc("AIMessage", "prior")]
            ]
        },
        "outputs": {
            "generations": [[{"text": "hello again", "message": lc("AIMessage", "hello again")}]],
            "llm_output": {"token_usage": {"prompt_tokens": 11, "completion_tokens": 4}},
        },
        "extra": {
            "metadata": {"ls_model_name": "gpt-4o-mini", "session_id": "ticket-1", "workload": "w"}
        },
        "prompt_tokens": None,
        "completion_tokens": None,
    }
    del row["total_cost"]

    record = langsmith.to_record(row)

    assert record is not None
    assert record.system == "sys"
    assert [(m.role, m.content) for m in record.messages] == [("user", "hi")]
    assert record.response == "hello again"
    assert (record.prompt_tokens, record.completion_tokens) == (11, 4)
    assert record.cost_usd is None
    assert (record.session_id, record.workload_hint, record.model) == (
        "ticket-1",
        "w",
        "gpt-4o-mini",
    )


def test_message_list_output_shape() -> None:
    row = {
        **_llm_run(),
        "outputs": {"messages": [{"role": "assistant", "content": "from messages"}]},
    }
    record = langsmith.to_record(row)
    assert record is not None
    assert record.response == "from messages"


def test_bare_prompt_input_and_string_output() -> None:
    row = {**_llm_run(), "inputs": {"prompt": "just a prompt"}, "outputs": {"output": "plain"}}
    record = langsmith.to_record(row)
    assert record is not None
    assert record.messages == (Message("user", "just a prompt"),)
    assert record.response == "plain"

    with pytest.raises(langsmith.MalformedRowError, match=r"inputs\.messages is not a list"):
        langsmith.to_record({**_llm_run(), "inputs": {"messages": "oops"}})
    with pytest.raises(langsmith.MalformedRowError, match="missing prompt"):
        langsmith.to_record({**_llm_run(), "inputs": {}})


def test_flat_text_only_generations() -> None:
    row = {**_llm_run(), "outputs": {"generations": [{"text": "flat"}]}}
    record = langsmith.to_record(row)
    assert record is not None
    assert record.response == "flat"


def test_missing_end_time_gives_zero_latency() -> None:
    row = _llm_run()
    del row["end_time"]
    record = langsmith.to_record(row)
    assert record is not None
    assert record.latency_ms == 0.0


def test_anthropic_run_with_usage_metadata_is_read_and_priced() -> None:
    row = _llm_run()
    del row["prompt_tokens"], row["completion_tokens"]
    row.update(
        {
            "extra": {
                "invocation_params": {"model_name": "claude-sonnet-4-5-20250929"},
                "metadata": {"ls_provider": "anthropic"},
            },
            "usage_metadata": {"input_tokens": 900, "output_tokens": 120},
        }
    )

    record = langsmith.to_record(row)

    assert record is not None
    assert record.model == "claude-sonnet-4-5-20250929"
    assert (record.prompt_tokens, record.completion_tokens) == (900, 120)
    assert PriceTable.load(None).cost(
        record.model, record.prompt_tokens, record.completion_tokens
    ) == pytest.approx(0.0045)


def test_usage_metadata_on_the_outputs_is_read() -> None:
    row = _llm_run()
    del row["prompt_tokens"], row["completion_tokens"]
    row["outputs"] = {
        **row["outputs"],  # type: ignore[dict-item]
        "usage_metadata": {"input_tokens": 7, "output_tokens": 2},
    }

    record = langsmith.to_record(row)

    assert record is not None
    assert (record.prompt_tokens, record.completion_tokens) == (7, 2)


def test_wrap_anthropic_run_reads_its_content_blocks() -> None:
    """``wrap_anthropic`` dumps the Message into ``outputs``: content blocks, folded system."""
    row = _llm_run()
    del row["prompt_tokens"], row["completion_tokens"]
    row.update(
        {
            "name": "ChatAnthropic",
            # ``_strip_not_given`` folds the ``system`` kwarg in as a system turn.
            "inputs": {
                "messages": [
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "hi"},
                ]
            },
            "outputs": {
                "role": "assistant",
                "model": "claude-sonnet-4-5-20250929",
                "content": [
                    {"type": "thinking", "thinking": "hmm"},  # non-text block: ignored
                    {"type": "text", "text": "hello "},
                    {"type": "text", "text": "there"},
                ],
                "usage_metadata": {"input_tokens": 900, "output_tokens": 120},
            },
            "extra": {"metadata": {"ls_provider": "anthropic"}},
        }
    )

    record = langsmith.to_record(row)

    assert record is not None
    assert record.system == "be brief"
    assert record.messages == (Message("user", "hi"),)
    assert record.response == "hello there"
    assert (record.prompt_tokens, record.completion_tokens) == (900, 120)
    assert record.model == "claude-sonnet-4-5-20250929"


def test_wrap_gemini_run_reads_its_contents_and_content() -> None:
    """A raw Gemini request/response: ``inputs.contents`` parts, string ``outputs.content``."""
    row = _llm_run()
    del row["prompt_tokens"], row["completion_tokens"]
    row.update(
        {
            "inputs": {"contents": [{"role": "user", "parts": [{"text": "what is AI?"}]}]},
            "outputs": {
                "role": "assistant",
                "content": "a field of study",
                "usage_metadata": {"prompt_token_count": 12, "candidates_token_count": 4},
            },
            "extra": {"metadata": {"ls_model_name": "gemini-2.5-flash"}},
        }
    )

    record = langsmith.to_record(row)

    assert record is not None
    assert record.system is None
    assert record.messages == (Message("user", "what is AI?"),)
    assert record.response == "a field of study"
    assert (record.prompt_tokens, record.completion_tokens) == (12, 4)
    assert record.model == "gemini-2.5-flash"


def test_wrap_gemini_run_lifts_the_system_instruction_into_a_system_turn() -> None:
    """``wrap_gemini`` keeps the system prompt in ``config.system_instruction``; it becomes turn 0."""
    row = _llm_run()
    del row["prompt_tokens"], row["completion_tokens"]
    row.update(
        {
            "inputs": {
                "messages": [{"role": "user", "content": "cancelling order"}],
                "config": {"system_instruction": "Classify the intent."},
            },
            "outputs": {
                "role": "assistant",
                "content": "cancel_order",
                "usage_metadata": {"input_tokens": 85, "output_tokens": 1},
            },
            "extra": {"metadata": {"ls_model_name": "gemini-2.5-flash-lite"}},
        }
    )

    record = langsmith.to_record(row)

    assert record is not None
    assert record.system == "Classify the intent."
    assert record.messages == (Message("user", "cancelling order"),)
    assert record.response == "cancel_order"

    row["inputs"] = {
        "messages": [{"role": "user", "content": "cancelling order"}],
        "config": {"system_instruction": {"parts": [{"text": "Be terse."}]}},
    }
    record = langsmith.to_record(row)
    assert record is not None
    assert record.system == "Be terse."
