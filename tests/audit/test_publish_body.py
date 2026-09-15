"""The publish bodies themselves: one workload row, and the two rules about the cap."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tests.audit._publish import entry

from dagnam.audit.publish_body import MAX_WORKLOADS, capped, too_many_selected, workload_body


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
    }


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
