"""The publish bodies themselves: one workload row, and the two rules about the cap."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from tests.audit._publish import entry

from dagnam.audit.publish_body import (
    MAX_WORKLOADS,
    capped,
    patch_body,
    too_many_selected,
    workload_body,
)
from dagnam.audit.scoring import score_labels
from dagnam.audit.state import StepState


def test_a_workload_without_numbers_still_publishes_a_valid_body() -> None:
    body = workload_body(
        {"id": "w9", "structure_class": "free_text"}, selected=False, pii_counts={}
    )
    assert body == {
        "workload_id": "w9",
        "structure_class": "free_text",
        "template_excerpt": "",
        "calls_per_day": 0.0,
        "mean_prompt_tokens": None,
        "mean_completion_tokens": None,
        "spend_usd_month": None,
        "export_p50_ms": None,
        "verdict": "unknown_cost",
        "verdict_reason": "",
        "ratio": None,
        "pii_counts": {},
        "selected": False,
        "response_mode": "text",
    }


def test_a_tool_call_workload_publishes_its_response_mode() -> None:
    # P4: the page tells the customer the replacement answers in message.content.
    entry_ = {"id": "w1", "structure_class": "json_object", "response_mode": "tool_call"}
    assert workload_body(entry_, selected=True, pii_counts={})["response_mode"] == "tool_call"
    stray = {"id": "w1", "structure_class": "json_object", "response_mode": "tools"}
    assert workload_body(stray, selected=True, pii_counts={})["response_mode"] == "text"


def test_the_cap_keeps_every_selected_workload_and_drops_the_cheapest_others() -> None:
    """Selected rows sort first at any spend: dropping one would 404 its candidates."""
    entries: list[Mapping[str, Any]] = [
        {**entry(f"w{i}", "enum_label", "candidate"), "cost_usd_month": float(i)}
        for i in range(MAX_WORKLOADS + 10)
    ]
    kept = capped(entries, {"w0", "w1"})

    ids = [str(e["id"]) for e in kept]
    assert len(kept) == MAX_WORKLOADS
    assert ids[:2] == ["w1", "w0"]  # the two the run took, cheapest though they are
    assert "w2" not in ids  # the cheapest unselected rows are what went


def test_a_scan_that_fits_is_left_exactly_as_it_was_scanned() -> None:
    entries: list[Mapping[str, Any]] = [entry("w1", "enum_label", "candidate")]
    assert capped(entries, set()) is entries


def test_only_more_selected_workloads_than_the_page_holds_refuses_a_publish() -> None:
    assert too_many_selected(MAX_WORKLOADS) is None
    refusal = too_many_selected(MAX_WORKLOADS + 1)
    assert refusal is not None
    assert str(MAX_WORKLOADS + 1) in refusal
    assert "--workloads" in refusal  # it says how to make the run publishable


def test_a_label_candidate_publishes_its_weakest_class_recall() -> None:
    # Q4: a constant "ok" student on a 1%-fraud holdout clears the interval (exact 0.99)
    # and misses every fraud row; the published block says so, and the page shows it.
    truth = ["ok"] * 990 + ["fraud"] * 10
    agreement = score_labels(["ok"] * 1_000, truth)
    step = StepState(agreement={**agreement.to_json(), "floor": 0.97}, scored=True)

    published = patch_body("replay_and_score", step, None, 12.5)["agreement"]

    assert isinstance(published, dict)
    assert published["min_class_recall"] == 0.0
    assert published["value"] == pytest.approx(0.99)
