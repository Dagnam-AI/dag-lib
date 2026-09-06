"""The scan report: the JSON contract and the markdown rendered from it."""

from __future__ import annotations

from datetime import UTC, date, datetime
import json
from pathlib import Path

import pytest
from tests.audit._records import make_workload

from dagnam.audit.discover import Workload
from dagnam.audit.prices import PriceRow, PriceTable
from dagnam.audit.scan_report import ScanReport, Window, build_scan_report, write_scan_report
from dagnam.audit.structure import StructureClass
from dagnam.audit.thresholds import MAX_UNSTRUCTURED_SHARE, PRICE_TABLE_STALE_DAYS

WINDOW = Window(
    start=datetime(2026, 8, 1, tzinfo=UTC), end=datetime(2026, 8, 31, tzinfo=UTC), days=30.0
)
FRESH = PriceTable(
    version="2026-09",
    as_of=datetime.now(UTC).date(),
    rows={"m": PriceRow("m", 1.0, 2.0, None, None)},
)
PASS_LIST = ["PII_EMAIL", "PII_PHONE"]
COUNTS = {"PII_EMAIL": 2}


def _workloads() -> list[Workload]:
    return [
        make_workload(calls_per_day=5_000, cost_month=4_000.0),  # candidate
        make_workload(calls_per_day=200, cost_month=120.0),  # not worth it
        make_workload(
            calls_per_day=10_000, cost_month=5_000.0, cls=StructureClass.FREE_TEXT
        ),  # not audited
        make_workload(calls_per_day=10, cost_month=5_000.0, n=999),  # too few samples
        make_workload(
            calls_per_day=100, cost_month=None, n=3_000, models=("m",)
        ),  # priced from the table
        make_workload(calls_per_day=100, cost_month=None, models=("other",)),  # unknown cost
    ]


def _report(workloads: list[Workload] | None = None, table: PriceTable = FRESH) -> ScanReport:
    return build_scan_report(
        workloads if workloads is not None else _workloads(),
        source="langfuse",
        window=WINDOW,
        price_table=table,
        pii_pass_list=PASS_LIST,
        pii_counts=COUNTS,
    )


def test_report_follows_the_contract() -> None:
    report = _report()
    js = report.to_json()

    assert js["schema"] == ScanReport.schema == "dagnam.audit.scan/1"
    assert datetime.fromisoformat(js["generated_at"]).tzinfo is not None
    assert js["source"] == "langfuse"
    assert js["window"] == {
        "start": "2026-08-01T00:00:00+00:00",
        "end": "2026-08-31T00:00:00+00:00",
        "days": 30.0,
    }
    assert js["price_table_version"] == "2026-09"
    assert js["pii"] == {"pass_list": PASS_LIST, "counts": COUNTS}
    assert js["totals"] == {
        "calls": 5_000 * 4 + 999 + 3_000,
        "cost_usd_month": pytest.approx(4_000 + 120 + 5_000 + 5_000 + 0.312),
        "prompt_tokens": sum(w.prompt_tokens for w in _workloads()),
        "completion_tokens": sum(w.completion_tokens for w in _workloads()),
    }
    assert js["warnings"] == []
    assert [w["verdict"]["status"] for w in js["workloads"]] == [
        "candidate",
        "not_worth_it",
        "not_audited",
        "too_few_samples",
        "not_worth_it",
        "unknown_cost",
    ]
    priced = js["workloads"][4]  # priced from the table, then judged like any other
    assert priced["cost_source"] == "price_table"
    assert priced["cost_usd_month"] == pytest.approx(0.312)
    first = js["workloads"][0]
    assert set(first) == set(_workloads()[0].to_json()) | {"verdict", "dataset"}
    assert set(first["verdict"]) == {"status", "ratio", "savings_usd_month", "reason"}
    assert first["dataset"] is None


def test_price_table_stamps_version_and_warns_when_stale() -> None:
    stale = PriceTable(version="2025-01", as_of=date(2025, 1, 6), rows={})

    report = _report(table=stale)

    assert report.price_table_version == "2025-01"
    (warning,) = report.warnings
    assert "2025-01" in warning
    assert str(PRICE_TABLE_STALE_DAYS) in warning
    assert "stale" in warning


def test_unstructured_share_warns_low_confidence() -> None:
    low = make_workload(calls_per_day=100, n=3_000, confidence="low")
    high = make_workload(calls_per_day=100, n=3_000)

    assert _report([low, high, high, high, high]).warnings == ()  # exactly the boundary share
    (warning,) = _report([low, high, high]).warnings
    assert f"{MAX_UNSTRUCTURED_SHARE:.0%}" in warning
    assert "low-confidence" in warning


def test_empty_scan_is_a_valid_report() -> None:
    js = _report([]).to_json()

    assert js["workloads"] == []
    assert js["totals"]["cost_usd_month"] == 0.0


def test_markdown_is_a_view_of_the_json(tmp_path: Path) -> None:
    report = _report()
    write_scan_report(report, tmp_path)
    md = (tmp_path / "scan-report.md").read_text()
    js = json.loads((tmp_path / "scan-report.json").read_text())

    assert js == report.to_json()
    for w in js["workloads"]:
        if w["cost_usd_month"] is not None:
            assert f"{w['cost_usd_month']:.2f}" in md  # every number in md comes from js
        assert w["verdict"]["reason"] in md
    assert f"{js['totals']['cost_usd_month']:.2f}" in md
    assert js["price_table_version"] in md
    assert js["generated_at"] in md
    assert md.index("## Not worth replacing") < md.index("## Candidates")
    assert md.index("## Candidates") < md.index("## PII")


def test_markdown_shows_customer_verdicts_and_never_replace_without_a_winner(
    tmp_path: Path,
) -> None:
    write_scan_report(_report(), tmp_path)
    md = (tmp_path / "scan-report.md").read_text()

    keep, candidates = md.split("## Candidates")
    assert keep.count("| KEEP |") == 5
    assert "NOT YET" not in keep
    assert candidates.count("| NOT YET |") == 1
    assert "KEEP" not in candidates
    assert "REPLACE" not in md  # no winner exists at scan time


def test_markdown_carries_the_warnings_and_unknown_costs(tmp_path: Path) -> None:
    stale = PriceTable(version="2025-01", as_of=date(2025, 1, 6), rows={})
    write_scan_report(_report(table=stale), tmp_path)
    md = (tmp_path / "scan-report.md").read_text()

    assert "## Warnings" in md
    assert "stale" in md
    assert "| unknown |" in md


def test_dataset_numbers_ride_along_and_a_thin_holdout_is_too_few_samples() -> None:
    dataset = {
        "rows": 4_000,
        "dedup_removed": 10,
        "redactions": 3,
        "truncated": 0,
        "split": {"train": 3_200, "eval_holdout": 800},
    }
    thin = {**dataset, "split": {"train": 3_900, "eval_holdout": 100}}
    workloads = [make_workload(calls_per_day=5_000, cost_month=4_000.0)] * 2
    js = build_scan_report(
        workloads,
        source="langfuse",
        window=WINDOW,
        price_table=FRESH,
        pii_pass_list=PASS_LIST,
        pii_counts=COUNTS,
        datasets={"w1": dataset},
    ).to_json()
    assert js["workloads"][0]["dataset"] == dataset
    assert js["workloads"][0]["verdict"]["status"] == "candidate"

    js = build_scan_report(
        workloads,
        source="langfuse",
        window=WINDOW,
        price_table=FRESH,
        pii_pass_list=PASS_LIST,
        pii_counts=COUNTS,
        datasets={"w1": thin},
    ).to_json()
    assert js["workloads"][0]["verdict"] == {
        "status": "too_few_samples",
        "ratio": None,
        "savings_usd_month": None,
        "reason": "100 holdout rows after the split; 200 needed",
    }
