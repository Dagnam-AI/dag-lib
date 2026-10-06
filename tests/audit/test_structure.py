"""Output-structure classes: enum labels, JSON objects, short spans, free text."""

from __future__ import annotations

import json
from typing import Any

import pytest

from dagnam.audit.readers.messages import effective_response
from dagnam.audit.scoring import modal_keys, score_json
from dagnam.audit.structure import (
    StructureClass,
    classify_outputs,
    normalize_response,
    output_shape,
    student_tokens,
    token_count,
)
from dagnam.audit.thresholds import (
    ENUM_MAX_DISTINCT,
    ENUM_MAX_MEDIAN_TOKENS,
    FLOOR_JSON,
    JSON_KEY_STABILITY,
    JSON_OBJECT_SHARE,
    SPAN_MAX_MEDIAN_TOKENS,
)
from dagnam.audit.token_estimate import estimate

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


def test_output_shape_sorts_one_response_into_its_unstructured_bucket() -> None:
    assert output_shape('{"a": 1}', ()) == "json"
    assert output_shape("", ({"function": {"arguments": '{"a": 1}'}},)) == "json"
    assert output_shape("yes", ()) == "short"
    assert output_shape(words(SPAN_MAX_MEDIAN_TOKENS), ()) == "short"
    assert output_shape(words(SPAN_MAX_MEDIAN_TOKENS + 1), ()) == "long"
    assert output_shape("[1, 2]", ()) == "short"  # JSON, but not an object


def test_a_router_that_decides_by_tool_name_keeps_its_decision() -> None:
    # A handoff router calls one of four tools, always with ``{}`` arguments.
    # Judged on the arguments alone it had one distinct output, so a student that
    # always emits ``{}`` scored ~100% -- a false REPLACE.
    names = ["transfer_to_billing", "transfer_to_shipping", "transfer_to_returns", "escalate"]
    calls = [({"function": {"name": names[i % 4], "arguments": "{}"}},) for i in range(N)]
    judged = [effective_response("", c) for c in calls]

    assert len({normalize_response(t) for t in judged}) == len(names)
    constant = [judged[0]] * N
    assert score_json(constant, judged, modal_keys(judged)).value < FLOOR_JSON


def test_parallel_tool_calls_are_one_json_answer() -> None:
    # A reply that sets a product and a team is the ordered list of both calls,
    # and a workload of such replies is still a JSON extraction.
    def calls(i: int) -> tuple[dict[str, Any], ...]:
        return (
            {
                "function": {
                    "name": "set_product",
                    "arguments": json.dumps({"product": "AB"[i % 2]}),
                }
            },
            {"function": {"name": "set_team", "arguments": json.dumps({"team": "xyz"[i % 3]})}},
        )

    assert classify_outputs([""] * N, [calls(i) for i in range(N)]) is StructureClass.JSON_OBJECT
    assert output_shape("", calls(0)) == "json"
    one_or_two = [calls(i)[: 1 + i % 2] for i in range(N)]
    assert classify_outputs([""] * N, one_or_two) is StructureClass.JSON_OBJECT


CHINESE_REPLY = (  # \uff0c is the fullwidth comma Chinese prose uses
    "您好\uff0c非常抱歉给您带来不便。您的订单{i}目前正在运输中\uff0c预计三到五个工作日内送达\uff0c"
    "请耐心等待\uff0c如有问题请随时联系我们。"
)


def test_text_without_spaces_is_counted_in_tokens_not_words() -> None:
    # A 60-character Chinese reply was 1 whitespace "token", so a free-text reply
    # workload read as short_span and was audited with an SFT student.
    replies = [CHINESE_REPLY.format(i=i) for i in range(N)]
    assert classify(replies) is StructureClass.FREE_TEXT
    assert output_shape(replies[0], ()) == "long"
    assert classify(["退款", "物流问题", "账户"] * 70) is StructureClass.ENUM_LABEL
    assert classify(["ยกเลิก", "คืนเงิน"] * 100) is StructureClass.ENUM_LABEL
    assert token_count("billing issue") == 2
    assert token_count("物流问题") == 2
    assert token_count("订单123已发货") == 4  # 订单 (1) + 123 (1) + 已发货 (2)
    assert token_count("") == 0


def test_student_tokens_is_the_token_estimate_cached_for_the_rows_of_a_workload() -> None:
    # Every row of a workload carries the same system prompt, so it is counted once.
    text = "You extract the order number and the customer's intent from an email."
    student_tokens.cache_clear()
    assert student_tokens(text) == estimate(text) == 15
    assert student_tokens(text) == 15
    info = student_tokens.cache_info()
    assert (info.hits, info.misses, info.maxsize) == (1, 1, 256)
