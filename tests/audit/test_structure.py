"""Output-structure classes: enum labels, JSON objects, short spans, free text."""

from __future__ import annotations

import json
from typing import Any

import pytest

from dagnam.audit.structure import StructureClass, classify_outputs, normalize_response
from dagnam.audit.thresholds import (
    ENUM_MAX_DISTINCT,
    ENUM_MAX_MEDIAN_TOKENS,
    JSON_KEY_STABILITY,
    JSON_OBJECT_SHARE,
    SPAN_MAX_MEDIAN_TOKENS,
)

N = 200
NO_CALLS: list[tuple[dict[str, Any], ...]] = [()] * N


def classify(responses: list[str]) -> StructureClass:
    return classify_outputs(responses, [()] * len(responses))


def words(count: int, seed: str = "w") -> str:
    return " ".join(f"{seed}{i}" for i in range(count))


def test_class_values_are_the_design_names() -> None:
    assert [c.value for c in StructureClass] == [
        "enum_label",
        "json_object",
        "short_span",
        "free_text",
    ]
    assert StructureClass("json_object") is StructureClass.JSON_OBJECT


def test_classify_outputs_boundaries() -> None:
    assert classify_outputs(["a"] * 150 + ["b"] * 50, NO_CALLS) is StructureClass.ENUM_LABEL
    assert classify_outputs([json.dumps({"k": i}) for i in range(N)], NO_CALLS) is (
        StructureClass.JSON_OBJECT
    )
    assert classify_outputs([f"span {i}" for i in range(N)], NO_CALLS) is StructureClass.SHORT_SPAN
    assert classify_outputs([words(40)] * N, NO_CALLS) is StructureClass.FREE_TEXT


def test_empty_and_single() -> None:
    assert classify_outputs([], []) is StructureClass.FREE_TEXT
    assert classify(["billing"]) is StructureClass.ENUM_LABEL
    assert classify(['{"a": 1}']) is StructureClass.JSON_OBJECT
    assert classify([words(40)]) is StructureClass.FREE_TEXT


def test_response_normalization_folds_case_and_whitespace() -> None:
    assert normalize_response("  Billing \n") == "billing"
    assert classify(["Billing", "billing", " BILLING "]) is StructureClass.ENUM_LABEL
    assert normalize_response("") == ""


@pytest.mark.parametrize(
    ("distinct", "expected"),
    [
        (ENUM_MAX_DISTINCT - 1, StructureClass.ENUM_LABEL),
        (ENUM_MAX_DISTINCT, StructureClass.ENUM_LABEL),
        (ENUM_MAX_DISTINCT + 1, StructureClass.SHORT_SPAN),
    ],
)
def test_enum_distinct_boundary(distinct: int, expected: StructureClass) -> None:
    responses = [f"label{i}" for i in range(distinct)]
    assert classify(responses) is expected


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (ENUM_MAX_MEDIAN_TOKENS - 1, StructureClass.ENUM_LABEL),
        (ENUM_MAX_MEDIAN_TOKENS, StructureClass.ENUM_LABEL),
        (ENUM_MAX_MEDIAN_TOKENS + 1, StructureClass.FREE_TEXT),
    ],
)
def test_enum_median_length_boundary(tokens: int, expected: StructureClass) -> None:
    # Two distinct outputs over 200 records: distinct ratio 0.01, so a miss on
    # the enum rule falls through the span rule to free text.
    responses = [words(tokens, seed="ab"[i % 2]) for i in range(N)]
    assert classify(responses) is expected


@pytest.mark.parametrize(
    ("objects", "expected"),
    [
        (int(JSON_OBJECT_SHARE * N) + 1, StructureClass.JSON_OBJECT),
        (int(JSON_OBJECT_SHARE * N), StructureClass.JSON_OBJECT),
        (int(JSON_OBJECT_SHARE * N) - 1, StructureClass.FREE_TEXT),
    ],
)
def test_json_share_boundary(objects: int, expected: StructureClass) -> None:
    responses = [json.dumps({"k": i, "v": words(12)}) for i in range(objects)]
    responses += [words(12, seed=f"f{i}") for i in range(N - objects)]
    assert classify(responses) is expected


def _with_keys(count: int, index: int) -> str:
    return json.dumps({f"k{j}": f"{index}-{words(3)}" for j in range(count)})


@pytest.mark.parametrize(
    ("kept_keys", "expected"),
    [
        # Modal key set has five keys; Jaccard to a subset of ``kept_keys`` is kept/5.
        (int(JSON_KEY_STABILITY * 5) + 1, StructureClass.JSON_OBJECT),
        (int(JSON_KEY_STABILITY * 5), StructureClass.JSON_OBJECT),
        (int(JSON_KEY_STABILITY * 5) - 1, StructureClass.FREE_TEXT),
    ],
)
def test_json_key_stability_boundary(kept_keys: int, expected: StructureClass) -> None:
    modal = N // 2 + 10  # an unambiguous modal key set
    responses = [_with_keys(5, i) for i in range(modal)]
    responses += [_with_keys(kept_keys, i) for i in range(modal, N)]
    assert classify(responses) is expected


def test_json_arrays_and_scalars_are_not_objects() -> None:
    assert classify([json.dumps([i, words(12)]) for i in range(N)]) is StructureClass.FREE_TEXT
    assert classify([json.dumps(i) for i in range(N)]) is StructureClass.SHORT_SPAN
    assert classify(["{not json" + words(12) for _ in range(N)]) is StructureClass.FREE_TEXT


def test_empty_objects_are_a_stable_key_set() -> None:
    assert classify(["{}"] * N) is StructureClass.JSON_OBJECT


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (SPAN_MAX_MEDIAN_TOKENS - 1, StructureClass.SHORT_SPAN),
        (SPAN_MAX_MEDIAN_TOKENS, StructureClass.SHORT_SPAN),
        (SPAN_MAX_MEDIAN_TOKENS + 1, StructureClass.FREE_TEXT),
    ],
)
def test_span_median_length_boundary(tokens: int, expected: StructureClass) -> None:
    responses = [words(tokens, seed=f"s{i}") for i in range(N)]
    assert classify(responses) is expected


@pytest.mark.parametrize(
    ("distinct", "expected"),
    [
        # 100 responses: distinct ratio 0.51 is a span, 0.50 is not (strict >).
        (51, StructureClass.SHORT_SPAN),
        (50, StructureClass.FREE_TEXT),
    ],
)
def test_span_distinct_ratio_boundary(distinct: int, expected: StructureClass) -> None:
    responses = [words(6, seed=f"s{i % distinct}") for i in range(100)]
    assert classify(responses) is expected


@pytest.mark.parametrize(
    "call",
    [
        {"function": {"name": "route", "arguments": '{"queue": "billing", "priority": 2}'}},
        {"name": "route", "args": {"queue": "billing", "priority": 2}},
        {"arguments": {"queue": "billing", "priority": 2}},
    ],
)
def test_tool_calls_are_json_objects_over_their_arguments(call: dict[str, Any]) -> None:
    responses = [""] * N
    assert classify_outputs(responses, [(call,)] * N) is StructureClass.JSON_OBJECT


def test_tool_call_without_object_arguments_still_counts_as_an_object() -> None:
    calls = [({"function": {"name": "f", "arguments": "not json"}},)] * N
    assert classify_outputs([""] * N, calls) is StructureClass.JSON_OBJECT
    assert classify_outputs([""] * N, [({"name": "f"},)] * N) is StructureClass.JSON_OBJECT


def test_tool_call_beats_the_text_response() -> None:
    call = {"function": {"arguments": '{"queue": "billing"}'}}
    assert classify_outputs(["billing"] * N, [(call,)] * N) is StructureClass.JSON_OBJECT


def test_lengths_must_match() -> None:
    with pytest.raises(ValueError):
        classify_outputs(["a", "b"], [()])
