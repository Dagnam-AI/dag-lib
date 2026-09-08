"""dag-lib's audit numbers are the contract's numbers -- one implementation, two importers."""

from __future__ import annotations

from typing import Any

from dagnam_contracts.audit import (
    scoring as contract_scoring,
    serving as contract_serving,
    verdict as contract_verdict,
)
from dagnam_contracts.audit.report import (
    DEFAULT_BASE_URL,
    REPORT_SCHEMA,
    render_switch_snippet,
    winner_of,
)

from dagnam.audit import derive, economics, prices, report, scoring, steps_serve, thresholds

# By name, not through ``dagnam.audit``: the package re-exports the ``frontier``
# *function*, which shadows the module of the same name.
from dagnam.audit.frontier import CandidateResult, Winner, frontier


def test_scorers_are_the_contract_objects() -> None:
    assert scoring.score_labels is contract_scoring.score_labels
    assert scoring.score_json is contract_scoring.score_json
    assert scoring.wilson_interval is contract_scoring.wilson_interval
    assert scoring.modal_keys is contract_scoring.modal_keys
    assert scoring.Agreement is contract_scoring.Agreement
    assert scoring.Z95 == contract_scoring.Z95
    assert derive.normalize_label is contract_scoring.normalize_label


def test_thresholds_and_rules_are_the_contract_values() -> None:
    for name in (
        "DAYS_PER_MONTH",
        "FLOOR_JSON",
        "FLOOR_LABEL",
        "MAINTENANCE_USD_MONTH",
        "MIN_HOLDOUT",
        "MIN_TRACES_PER_WORKLOAD",
        "RATIO_CANDIDATE",
        "RATIO_NOT_WORTH_IT",
    ):
        assert getattr(thresholds, name) == getattr(contract_verdict, name), name
    assert steps_serve.UNRELIABLE_ERROR_SHARE == contract_verdict.UNRELIABLE_ERROR_SHARE
    assert economics.customer_verdict is contract_verdict.customer_verdict
    assert economics.ratio_status is contract_verdict.ratio_status
    assert economics.serving_cost_usd_month is contract_serving.serving_cost_usd_month
    assert frontier is contract_verdict.frontier
    assert CandidateResult is contract_verdict.CandidateResult
    assert Winner is contract_verdict.Winner
    assert report.SCHEMA == REPORT_SCHEMA
    assert report.DEFAULT_BASE_URL == DEFAULT_BASE_URL
    assert report.render_switch_snippet is render_switch_snippet


def test_bundled_serving_rates_equal_the_contract_rates() -> None:
    """``serving.json`` still stamps the price table, but its numbers are the contract's."""
    for kind, rates in contract_serving.SERVING_RATES.items():
        for unit, value in rates.items():
            assert prices.SERVING_RATES[kind][unit] == value, f"{kind}.{unit}"


def test_report_winner_matches_the_contract_over_the_same_candidates() -> None:
    candidates: list[dict[str, Any]] = [
        {
            "kind": "head_tune",
            "status": "scored",
            "deployment_id": "d",
            "agreement": {"ci95": [0.99, 1.0]},
            "latency_ms": {"p95": 500.0},
            "serving_cost_usd_month": {"value": 12.0},
        }
    ]
    assert report._winner(candidates, 0.95) == winner_of(candidates, floor=0.95)
