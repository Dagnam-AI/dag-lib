"""Reasoning never reaches a prompt or a reply, in any shape, through any reader."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import pytest
from tests.audit.readers._reasoning_shapes import (
    BLOCKS,
    GONE,
    REASONING_ITEM,
    RESIDUAL,
    SHAPES,
    A,
    R,
)

from dagnam.audit import Message, TraceRecord, read_traces
from dagnam.audit.readers import generic, langfuse, langsmith, openai_jsonl
from dagnam.audit.readers.messages import split_prompt
from dagnam.audit.readers.reasoning import (
    REASONING_MESSAGE_FIELDS,
    REASONING_TYPES,
    is_reasoning,
    is_reasoning_type,
)


def _langfuse(value: object) -> TraceRecord | None:
    return langfuse.to_record(
        {
            "id": "gen-1",
            "traceId": "t-1",
            "type": "GENERATION",
            "startTime": "2026-08-01T09:00:00Z",
            "input": [{"role": "user", "content": "hi"}],
            "output": value,
        }
    )


def _langsmith(outputs: object) -> TraceRecord | None:
    return langsmith.to_record(
        {
            "id": "run-1",
            "trace_id": "t-1",
            "run_type": "llm",
            "start_time": "2026-08-01T09:00:00Z",
            "inputs": {"messages": [{"role": "user", "content": "hi"}]},
            "outputs": outputs,
        }
    )


def _openai(completion: dict[str, Any]) -> TraceRecord | None:
    return openai_jsonl.to_record(
        {
            "request": {"messages": [{"role": "user", "content": "hi"}]},
            "response": {"id": "c-1", "created": 1_785_000_000, **completion},
        }
    )


def _generic(value: object) -> TraceRecord | None:
    return generic.bind(None).to_record(
        {
            "trace_id": "1",
            "ts": "2026-08-01T00:00:00Z",
            "messages": [{"role": "user", "content": "hi"}],
            "response": value,
        }
    )


def _readers(where: str, value: Any) -> dict[str, Callable[[], TraceRecord | None]]:
    """Every way each reader can be handed this shape: as an object, and as a JSON string."""
    encoded = value if isinstance(value, str) else json.dumps(value)
    cases: dict[str, Callable[[], TraceRecord | None]] = {
        "langfuse": lambda: _langfuse(value),
        "langfuse-string": lambda: _langfuse(encoded),
        "jsonl": lambda: _generic(value),
        "csv": lambda: _generic(encoded),
    }
    if where == "message":
        cases["openai"] = lambda: _openai({"choices": [{"message": value}]})
        cases["langsmith"] = lambda: _langsmith({"choices": [{"message": value}]})
    elif where == "object":
        cases["langsmith"] = lambda: _langsmith(value)
    elif where in ("items", "lone"):
        items = value if where == "items" else [value]
        cases["openai"] = lambda: _openai({"object": "response", "output": items})
        cases["langsmith"] = lambda: _langsmith({"output": items})
        cases["langsmith-string"] = lambda: _langsmith({"output": json.dumps(items)})
    else:
        cases["openai"] = lambda: _openai({"choices": [{"message": {"content": value}}]})
        cases["langsmith"] = lambda: _langsmith({"output": value})
    return cases


def _cases() -> list[Any]:
    return [
        pytest.param(name, reader, id=f"{name}-{reader}")
        for name, (where, value, _) in SHAPES.items()
        for reader in _readers(where, value)
    ]


def _leaks(*texts: str) -> bool:
    return any(secret in text for text in texts for secret in R)


@pytest.mark.parametrize(("name", "reader"), _cases())
def test_reasoning_never_reaches_the_reply_through_any_reader(name: str, reader: str) -> None:
    where, value, expected = SHAPES[name]
    record = _readers(where, value)[reader]()

    assert record is not None
    assert not _leaks(record.response, json.dumps(record.response_tool_calls))
    if expected == GONE:
        # No answer to learn from: the call stays (it was billed), the row does not.
        assert (record.response, record.response_tool_calls) == ("", ())
        assert record.reasoning_only
    else:
        assert (record.response, record.response_tool_calls) == (expected, ())
        assert not record.reasoning_only


def test_every_reader_is_covered_by_every_kind_of_shape() -> None:
    # A reader that cannot take a kind would silently drop out of the matrix above.
    readers = {"langfuse", "langfuse-string", "jsonl", "csv", "openai", "langsmith"}
    for where in ("message", "object", "items", "lone", "text"):
        value = next(v for w, v, _ in SHAPES.values() if w == where)
        assert set(_readers(where, value)) >= readers - ({"openai"} if where == "object" else set())


def _turn_cases() -> list[Any]:
    return [
        pytest.param(name, id=name)
        for name, (where, _, _) in SHAPES.items()
        if where in ("message", "items", "text")
    ]


@pytest.mark.parametrize("name", _turn_cases())
def test_reasoning_never_reaches_a_context_turn_through_any_reader(name: str) -> None:
    where, value, expected = SHAPES[name]
    turn = value if where == "message" else {"role": "assistant", "content": value}
    prompt = [{"role": "user", "content": "q1"}, turn, {"role": "user", "content": "q2"}]
    answer = "" if expected == GONE else expected
    want = (Message("user", "q1"), Message("assistant", answer), Message("user", "q2"))

    seen = {
        "langfuse": langfuse.to_record(
            {
                "id": "g",
                "traceId": "t",
                "type": "GENERATION",
                "startTime": "2026-08-01T09:00:00Z",
                "input": prompt,
                "output": "ok",
            }
        ),
        "jsonl": generic.bind(None).to_record(
            {"trace_id": "1", "ts": "2026-08-01T00:00:00Z", "messages": prompt, "response": "ok"}
        ),
        "csv": generic.bind(None).to_record(
            {
                "trace_id": "1",
                "ts": "2026-08-01T00:00:00Z",
                "messages": json.dumps(prompt),
                "response": "ok",
            }
        ),
        "openai": openai_jsonl.to_record(
            {
                "request": {"messages": prompt},
                "response": {
                    "id": "c",
                    "created": 1_785_000_000,
                    "choices": [{"message": {"content": "ok"}}],
                },
            }
        ),
        "langsmith": langsmith.to_record(
            {
                "id": "r",
                "trace_id": "t",
                "run_type": "llm",
                "start_time": "2026-08-01T09:00:00Z",
                "inputs": {"messages": prompt},
                "outputs": {"output": "ok"},
            }
        ),
    }
    for reader, record in seen.items():
        assert record is not None, reader
        assert record.messages == want, reader
        assert not _leaks(*(m.content for m in record.messages)), reader


def test_a_user_turn_that_is_json_text_of_blocks_holding_reasoning_is_cleaned_too() -> None:
    _, turns = split_prompt([{"role": "user", "content": json.dumps(BLOCKS)}])
    assert turns == (Message("user", A),)


def test_every_type_on_the_table_is_dropped_wherever_it_sits() -> None:
    for kind in sorted(REASONING_TYPES):
        assert is_reasoning_type(kind)
        assert is_reasoning({"type": kind, "text": R[0]})
        reply = {"role": "assistant", "content": [{"type": kind, "text": R[0]}, A]}
        for read in (_langfuse(reply), _generic(reply)):
            assert read is not None
            assert (read.response, R[0] in read.response) == (A, False)
        _, turns = split_prompt(
            [
                {"role": "user", "content": [{"type": kind, "text": R[0]}, "q"]},
                {"role": "assistant", "content": [{"type": kind, "text": R[0]}, "a"]},
                {"role": "user", "content": "q2"},
            ]
        )
        assert turns == (Message("user", "q"), Message("assistant", "a"), Message("user", "q2"))


def test_a_response_object_no_reader_recognises_is_never_written_out_whole() -> None:
    chunk = {"choices": [{"delta": {"reasoning_content": R[0]}}], "model": "m"}
    for read, value in ((_langfuse, chunk), (_generic, chunk), (_generic, json.dumps(chunk))):
        record = read(value)
        assert record is not None
        assert (record.response, record.reasoning_only) == ("", True)


def test_an_answer_that_is_json_keeps_every_byte_even_with_a_field_named_reasoning() -> None:
    # Chain of thought in a schema is the model's own answer, not an envelope's reasoning.
    for answer in (
        {"label": "x", "score": 3},
        {"reasoning": "because", "label": "x"},
        {"thinking": "hmm", "label": "x"},
        [{"reasoning": "because", "label": "x"}],
    ):
        for read, value in (
            (_langfuse, answer),
            (_generic, answer),
            (_generic, json.dumps(answer)),
        ):
            record = read(value)
            assert record is not None
            assert record.response == json.dumps(answer)
            assert not record.reasoning_only


def test_a_reasoning_only_row_stays_in_the_read_and_is_flagged(tmp_path: Path) -> None:
    def line(i: int, output: object) -> str:
        return json.dumps(
            {
                "id": f"g{i}",
                "traceId": "t",
                "type": "GENERATION",
                "startTime": "2026-08-01T09:00:00Z",
                "input": [{"role": "user", "content": "hi"}],
                "output": output,
            }
        )

    rows = [line(i, {"role": "assistant", "content": "ok"}) for i in range(60)]
    rows += [line(60 + i, REASONING_ITEM) for i in range(3)]
    path = tmp_path / "export.jsonl"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    records, stats = read_traces(path, source="langfuse")
    kept = list(records)

    assert (stats.rows_seen, stats.rows_kept, stats.rows_malformed) == (63, 63, 0)
    assert [r.reasoning_only for r in kept] == [False] * 60 + [True] * 3


@pytest.mark.parametrize(
    "name",
    ["reasoning_content", "reasoning", "reasoning_details", "reasoningContent", "thinking_blocks"],
)
def test_a_message_with_its_reasoning_in_a_field_and_no_content_is_reasoning_only(
    name: str,
) -> None:
    reply = {"role": "assistant", "content": None, name: [{"text": R[0]}]}
    for read in (_langfuse, _generic):
        record = read(reply)
        assert record is not None
        assert (record.response, record.reasoning_only) == ("", True)
    assert name in REASONING_MESSAGE_FIELDS


@pytest.mark.parametrize("name", list(RESIDUAL))
def test_an_unrecognised_object_with_reasoning_under_a_plain_name_is_kept_as_written(
    name: str,
) -> None:
    # The stated residual: it cannot be told from a structured answer, so it is not guessed at.
    obj = RESIDUAL[name]
    for read, value in ((_langfuse, obj), (_generic, obj), (_generic, json.dumps(obj))):
        record = read(value)
        assert record is not None
        assert (record.response, record.reasoning_only) == (json.dumps(obj), False)


def test_a_role_less_object_with_content_and_a_reasoning_summary() -> None:
    # No reader can call it a reply (an answer may have a ``content`` field), so the generic
    # readers keep the call and no answer; LangSmith's ``outputs`` is a reply message by
    # contract, so it reads ``content`` and still never reads the summary.
    obj = {"content": A, "reasoning_summary": R[0]}
    for read in (_langfuse, _generic):
        record = read(obj)
        assert record is not None
        assert (record.response, record.reasoning_only) == ("", True)
    record = _langsmith(obj)
    assert record is not None
    assert (record.response, record.reasoning_only) == (A, False)
