"""Scorers against hand-computed values, and the Wilson interval at its edges."""

from __future__ import annotations

import pytest

from dagnam.audit.scoring import Agreement, modal_keys, score_json, score_labels, wilson_interval


def test_score_labels_exact_and_macro_f1_with_ci() -> None:
    a = score_labels(["a", "b", "a", "c"], ["a", "b", "b", "c"])
    assert a.metric == "exact"
    assert a.exact == 0.75
    assert a.value == 0.75
    assert a.n == 4
    # a: P=.5 R=1 F1=.667; b: P=1 R=.5 F1=.667; c: F1=1 -> mean 0.7778
    assert a.macro_f1 is not None
    assert abs(a.macro_f1 - 0.7778) < 1e-3
    assert 0.30 < a.ci95[0] < 0.75 < a.ci95[1]
    assert abs(a.ci95[0] - 0.3006) < 1e-3
    assert abs(a.ci95[1] - 0.9544) < 1e-3
    assert a.field_f1 is None


def test_score_labels_normalizes_both_sides() -> None:
    a = score_labels(["Billing.", " returns"], ["billing", "Returns"])
    assert a.exact == 1.0
    assert a.macro_f1 == 1.0


def test_score_labels_empty_is_zero_with_an_uninformative_interval() -> None:
    a = score_labels([], [])
    assert (a.value, a.n, a.ci95, a.macro_f1) == (0.0, 0, (0.0, 1.0), 0.0)


def test_paired_inputs_must_align() -> None:
    with pytest.raises(ValueError, match="2 predictions but 1 truths"):
        score_labels(["a", "b"], ["a"])
    with pytest.raises(ValueError, match="1 predictions but 0 truths"):
        score_json(["{}"], [], keys=["k"])


def test_wilson_interval_edges() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(10, 10)
    assert abs(lo - 0.7225) < 1e-3
    assert abs(hi - 1.0) < 1e-12
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert abs(hi - 0.2775) < 1e-3


def test_score_json_field_level() -> None:
    a = score_json(
        ['{"product":"x","order_id":"1"}'],
        ['{"product":"x","order_id":"2","sentiment":"neg"}'],
        keys=["product", "order_id", "sentiment"],
    )
    assert a.metric == "field_f1"
    assert a.field_precision == 0.5
    assert a.field_recall is not None
    assert abs(a.field_recall - 1 / 3) < 1e-9
    assert a.field_f1 is not None
    assert abs(a.field_f1 - 0.4) < 1e-9
    assert a.value == a.field_f1
    assert a.n == 1
    assert a.ci95 == wilson_interval(2, 5)  # 2tp of 2tp+fp+fn trials


def test_score_json_treats_unparseable_or_non_object_text_as_no_fields() -> None:
    a = score_json(["not json", "[1, 2]"], ['{"k": 1}', '{"k": 2}'], keys=["k"])
    assert (a.field_precision, a.field_recall, a.field_f1) == (0.0, 0.0, 0.0)
    perfect = score_json(['{"k": {"b": 1, "a": 2}}'], ['{"k": {"a": 2, "b": 1}}'], keys=["k"])
    assert perfect.value == 1.0


def test_score_json_ignores_keys_outside_the_key_set() -> None:
    a = score_json(['{"k": 1, "extra": 9}'], ['{"k": 1, "other": 3}'], keys=["k"])
    assert a.value == 1.0


def test_modal_keys_is_the_majority_key_set_sorted() -> None:
    truths = ['{"b": 1, "a": 2}', '{"a": 1, "b": 3}', '{"a": 1}', "nope"]
    assert modal_keys(truths) == ["a", "b"]
    assert modal_keys(["nope", "[1]"]) == []


def test_agreement_to_json_carries_only_the_populated_extras() -> None:
    a = Agreement(metric="exact", value=0.5, ci95=(0.1, 0.9), n=2, exact=0.5, macro_f1=0.4)
    assert a.to_json() == {
        "metric": "exact",
        "value": 0.5,
        "ci95": [0.1, 0.9],
        "n": 2,
        "exact": 0.5,
        "macro_f1": 0.4,
    }
