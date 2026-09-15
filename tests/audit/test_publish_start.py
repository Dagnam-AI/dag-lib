"""``Publisher.start``: the ``AuditCreate`` body, the account's workload cap, the watch link."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from tests.audit._platform import FakePlatform
from tests.audit._publish import AUDIT_ID, SCAN, entry, start

from dagnam._core.exceptions import APIError
from dagnam.audit.publish import MAX_WORKLOADS, Publisher, audit_url, workload_body
from dagnam.audit.state import AuditState

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------- start


def test_start_publishes_every_workload_and_records_the_audit_id(
    publisher: Publisher, platform: FakePlatform, state: AuditState, audit_dir: Path
) -> None:
    start(publisher, audit_dir)

    assert state.audit_id == "audit-1"
    (body,) = platform.audits
    assert body["project_id"] == "proj-1"
    assert body["source"] == "cli"
    assert body["floor"] == 0.97
    assert body["max_credits"] == 500
    assert body["price_table_version"] == "2026-09"
    assert body["scan_generated_at"] == "2026-09-06T10:00:00+00:00"
    assert body["sdk_version"] == "9.9.9"
    assert body["local_dir_name"] == "audit"
    w1, w2 = body["workloads"]
    assert w1 == {
        "workload_id": "w1",
        "structure_class": "enum_label",
        "template_excerpt": "Classify the ticket",
        "calls_per_day": 100.0,
        "mean_prompt_tokens": 100,
        "mean_completion_tokens": 2,
        "spend_usd_month": 900.0,
        "export_p50_ms": 400.0,
        "verdict": "candidate",
        "verdict_reason": "12.0x over the floor",
        "ratio": 12.0,
        "pii_counts": {},
        "selected": True,
    }
    assert w2["selected"] is False  # scanned, but not one of the workloads this run took


def test_start_carries_the_redaction_counts_the_scan_derived(
    publisher: Publisher, platform: FakePlatform, audit_dir: Path
) -> None:
    meta_path = audit_dir / "workloads" / "w1" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["stats"]["redact"]["counts"] = {"PII_EMAIL": 7}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    start(publisher, audit_dir)

    w1, w2 = platform.audits[0]["workloads"]
    assert w1["pii_counts"] == {"PII_EMAIL": 7}
    assert w2["pii_counts"] == {}


def test_start_on_a_resumed_run_creates_no_second_audit(
    publisher: Publisher, platform: FakePlatform, state: AuditState, audit_dir: Path
) -> None:
    state.audit_id = AUDIT_ID
    start(publisher, audit_dir)
    assert platform.audits == []
    assert state.audit_id == AUDIT_ID


def test_start_publishes_nothing_when_the_scan_found_no_workload(
    publisher: Publisher, platform: FakePlatform, state: AuditState, audit_dir: Path
) -> None:
    publisher.start(
        audit_dir, {"workloads": []}, [], floor=0.97, max_credits=500, sdk_version="9.9.9"
    )
    assert platform.audits == []
    assert state.audit_id is None


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


# ------------------------------------------------ the account's own workload cap


def test_a_scan_bigger_than_the_account_holds_publishes_the_run_s_own_first(
    publisher: Publisher, platform: FakePlatform, audit_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """201 workloads would be a dropped 422; the run's own and the biggest spenders go."""
    cheap = [
        {**entry(f"w{i}", "enum_label", "candidate"), "cost_usd_month": float(i)}
        for i in range(200)
    ]
    mine = {**entry("mine", "enum_label", "candidate"), "cost_usd_month": 0.0}
    scan = {**SCAN, "workloads": [mine, *cheap]}

    publisher.start(audit_dir, scan, ["mine"], floor=0.97, max_credits=500, sdk_version="9.9.9")

    published = platform.audits[0]["workloads"]
    assert len(published) == MAX_WORKLOADS
    assert published[0]["workload_id"] == "mine"  # selected first, despite spending nothing
    assert published[0]["selected"] is True
    assert [w["workload_id"] for w in published[1:4]] == ["w199", "w198", "w197"]
    assert "w0" not in {w["workload_id"] for w in published}
    assert "the scan found 201 workloads and the account holds 200" in caplog.text


def test_more_selected_workloads_than_the_account_holds_publishes_nothing(
    publisher: Publisher, platform: FakePlatform, state: AuditState, audit_dir: Path
) -> None:
    """Publishing a truncated list would 404 every candidate the cap dropped.

    The cap ranks selected workloads first, so it only ever drops unselected
    ones -- unless the run itself took more than the page holds, and then there
    is no honest body to send. The run keeps every number locally.
    """
    selected = [entry(f"w{i}", "enum_label", "candidate") for i in range(MAX_WORKLOADS + 1)]
    lines: list[str] = []
    publisher = Publisher(platform, state, lines.append)

    publisher.start(
        audit_dir,
        {**SCAN, "workloads": selected},
        [str(w["id"]) for w in selected],
        floor=0.97,
        max_credits=500,
        sdk_version="9.9.9",
    )

    assert platform.audits == []
    assert state.audit_id is None
    assert lines == [
        "not published: this run took 201 workloads and an audit page holds 200;"
        " every number stays in the local report. Run fewer at a time (--workloads)"
        " to publish it."
    ]


def test_exactly_as_many_selected_workloads_as_the_account_holds_is_published(
    publisher: Publisher, platform: FakePlatform, audit_dir: Path
) -> None:
    """The cap is the last size that fits, not the first that does not."""
    selected = [entry(f"w{i}", "enum_label", "candidate") for i in range(MAX_WORKLOADS)]

    publisher.start(
        audit_dir,
        {**SCAN, "workloads": [*selected, entry("extra", "enum_label", "not_worth_it")]},
        [str(w["id"]) for w in selected],
        floor=0.97,
        max_credits=500,
        sdk_version="9.9.9",
    )

    published = platform.audits[0]["workloads"]
    assert len(published) == MAX_WORKLOADS
    assert all(w["selected"] for w in published)  # the unselected extra is what was dropped


def test_a_scan_within_the_cap_is_published_in_the_order_it_was_scanned(
    publisher: Publisher, platform: FakePlatform, audit_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    start(publisher, audit_dir)
    assert [w["workload_id"] for w in platform.audits[0]["workloads"]] == ["w1", "w2"]
    assert "the account holds" not in caplog.text


# ------------------------------------------------ where to watch what was published


def test_audit_url_points_at_the_site_that_goes_with_the_api() -> None:
    """The same derivation `dagnam login` uses; a private API host links to itself."""
    assert audit_url("https://api.dagnam.ai", "a1") == "https://dagnam.ai/audits/a1"
    assert audit_url("http://localhost:8000", "a1") == "http://localhost:5173/audits/a1"
    assert audit_url("https://corp.internal/", "a1") == "https://corp.internal/audits/a1"


def test_a_created_audit_says_where_to_watch_it(
    platform: FakePlatform, state: AuditState, audit_dir: Path
) -> None:
    lines: list[str] = []
    start(Publisher(platform, state, lines.append), audit_dir)
    assert lines == ["published: audit-1 — watch it at https://x/audits/audit-1"]


def test_nothing_is_said_when_the_audit_was_not_created(
    platform: FakePlatform, state: AuditState, audit_dir: Path
) -> None:
    lines: list[str] = []
    platform.publish_errors["create_audit"] = [APIError(503, "audit service down")]
    publisher = Publisher(platform, state, lines.append)

    start(publisher, audit_dir)
    assert lines == []

    state.audit_id = AUDIT_ID  # a resumed run: the audit exists, so nothing is created
    start(publisher, audit_dir)
    assert lines == []


def test_a_console_that_cannot_print_the_link_does_not_end_the_run(
    platform: FakePlatform, state: AuditState, audit_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A cp1252 console raises on the em dash; the audit is published either way."""

    def refuse(_line: str) -> None:
        raise UnicodeEncodeError("cp1252", "—", 0, 1, "undefined")

    start(Publisher(platform, state, refuse), audit_dir)

    assert state.audit_id == "audit-1"
    assert "the audit's link failed" in caplog.text
