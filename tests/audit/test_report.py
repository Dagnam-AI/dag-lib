"""The audit report: the JSON contract, the verdict view, the winner, the switch, the markdown."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import pytest
from tests.audit._records import make_workload

from dagnam.audit.candidates import HEAD_TUNE, HOSTED_FLOOR, SFT_SMALL, CandidateKind
from dagnam.audit.economics import serving_cost_usd_month
from dagnam.audit.prices import PriceRow, PriceTable
from dagnam.audit.report import (
    SCHEMA,
    build_audit_report,
    candidate_status,
    render_markdown,
    render_switch_snippet,
    write_audit_report,
)
from dagnam.audit.scan_report import Window, build_scan_report
from dagnam.audit.state import AuditState, StepState
from dagnam.audit.structure import StructureClass

HEAD, SFT, HOSTED = CandidateKind.HEAD_TUNE, CandidateKind.SFT_SMALL, CandidateKind.HOSTED_FLOOR
SECRET = "dk-secret-never-shown"
TABLE = PriceTable(
    version="2026-09",
    as_of=datetime.now(UTC).date(),
    rows={
        "big": PriceRow("big", 10.0, 50.0, None, "mini"),
        "mini": PriceRow("mini", 1.0, 5.0, None, None),
        "orphan": PriceRow("orphan", 1.0, 5.0, None, "ghost"),
    },
)
WINDOW = Window(datetime(2026, 8, 1, tzinfo=UTC), datetime(2026, 8, 31, tzinfo=UTC), 30.0)


def _scan(models: tuple[str, ...] = ("big",)) -> dict[str, Any]:
    workloads = [
        make_workload(calls_per_day=5_000, cost_month=4_000.0, models=models),
        make_workload(calls_per_day=5_000, cost_month=4_000.0, cls=StructureClass.JSON_OBJECT),
        make_workload(calls_per_day=200, cost_month=120.0),
        make_workload(calls_per_day=10_000, cost_month=5_000.0, cls=StructureClass.FREE_TEXT),
    ]
    ids = ["w1", "w2", "w3", "w4"]
    report = build_scan_report(
        workloads,
        source="langfuse",
        window=WINDOW,
        price_table=TABLE,
        pii_pass_list=["PII_EMAIL"],
        pii_counts={"PII_EMAIL": 3},
        datasets={
            "w1": {
                "rows": 800,
                "dedup_removed": 4,
                "redactions": 3,
                "truncated": 1,
                "split": {"train": 600, "eval_holdout": 200},
            }
        },
    ).to_json()
    for entry, workload_id in zip(report["workloads"], ids, strict=True):
        entry["id"] = entry["template_hash"] = workload_id
    return report


def _scored(ci: tuple[float, float], **extra: Any) -> StepState:
    fields: dict[str, Any] = {
        "dataset_id": "ds-1",
        "run_id": "run-1",
        "training_job_id": "job-1",
        "run_status": "completed",
        "model_version_id": "mv-1",
        "deployment_id": "dep-1",
        "key_ref": "w1/head_tune",
        "deploy_status": "running",
        "scored": True,
        "latency": {"p50": 40.0, "p95": 90.0, "calls": 200, "errors": 0},
        "training_cost_credits": 12.0,
        "agreement": {
            "metric": "exact",
            "value": 0.99,
            "ci95": [ci[0], ci[1]],
            "n": 200,
            "floor": 0.97,
            "passes_floor": ci[0] >= 0.97,
        },
    }
    return StepState(**{**fields, **extra})


def _state() -> AuditState:
    state = AuditState(project_id="proj-1", price_table_version="2026-09")
    state.workloads["w1"] = {HOSTED: StepState(), HEAD: _scored((0.98, 1.0))}
    state.workloads["w2"] = {
        HOSTED: StepState(),
        SFT: _scored((0.90, 0.96), deployment_id="dep-2", key_ref="w2/sft_small"),
    }
    return state


def test_report_follows_the_contract() -> None:
    scan = _scan()
    js = build_audit_report(_state(), scan, price_table=TABLE)

    assert js["schema"] == SCHEMA == "dagnam.audit.report/1"
    assert js["project_id"] == "proj-1"
    assert js["halted"] is None
    assert js["price_table_version"] == "2026-09"
    assert js["scan_generated_at"] == scan["generated_at"]
    assert js["generated_at"] != scan["generated_at"]
    w1, w2, w3, w4 = js["workloads"]
    assert [c["kind"] for c in w1["candidates"]] == ["hosted_floor", "head_tune"]
    hosted, head = w1["candidates"]
    assert set(head) == {
        "kind",
        "base",
        "run_id",
        "model_version_id",
        "deployment_id",
        "agreement",
        "latency_ms",
        "serving_cost_usd_month",
        "training_cost_credits",
        "status",
    }
    assert (head["run_id"], head["model_version_id"], head["deployment_id"]) == (
        "run-1",
        "mv-1",
        "dep-1",
    )
    assert head["agreement"]["ci95"] == [0.98, 1.0]
    assert head["latency_ms"] == {"p50": 40.0, "p95": 90.0, "source": "measured"}
    assert head["serving_cost_usd_month"] == {
        "value": serving_cost_usd_month(
            "cpu-classifier", calls_per_day=5_000, completion_tokens=10_000, calls=5_000
        ),
        "basis": "estimated",
    }
    assert head["training_cost_credits"] == 12.0
    assert head["status"] == "scored"
    # The hosted floor is the cheaper variant of the one model, priced over the month.
    assert hosted["base"] == "mini"
    assert hosted["serving_cost_usd_month"]["value"] == pytest.approx(
        (500_000 * 1.0 + 10_000 * 5.0) / 1e6 * 30 / (5_000 / 5_000)
    )
    assert hosted["latency_ms"] == {"p50": None, "p95": None, "source": "untested"}
    assert hosted["status"] == "untested"
    assert w1["winner"] == {
        "kind": "head_tune",
        "cost_usd_month": head["serving_cost_usd_month"]["value"],
        "agreement_lo": 0.98,
        "deployment_id": "dep-1",
    }
    assert w1["switch"] == {
        "base_url": "https://api.dagnam.ai/v1",
        "model": "dep-1",
        "key_ref": "w1/head_tune",
    }
    assert (w2["winner"], w2["switch"]) == (None, None)
    assert w2["candidates"][1]["kind"] == "sft_small"
    for unaudited in (w3, w4):
        assert (unaudited["candidates"], unaudited["winner"], unaudited["switch"]) == (
            [],
            None,
            None,
        )
    assert SECRET not in json.dumps(js)


def test_hosted_floor_needs_one_priced_model_with_a_variant() -> None:
    def hosted(models: tuple[str, ...]) -> dict[str, Any]:
        js = build_audit_report(_state(), _scan(models), price_table=TABLE)
        return js["workloads"][0]["candidates"][0]

    assert (
        hosted(("big", "mini"))["base"],
        hosted(("big", "mini"))["serving_cost_usd_month"]["value"],
    ) == (None, None)
    assert hosted(("mini",))["base"] is None  # priced, but no cheaper variant
    assert hosted(("nope",))["base"] is None  # not in the table
    ghost = hosted(("orphan",))
    assert (ghost["base"], ghost["serving_cost_usd_month"]["value"]) == ("ghost", None)


def test_winner_uses_the_recorded_floor_and_a_custom_base_url() -> None:
    state = _state()
    head = state.workloads["w1"][HEAD]
    assert head.agreement is not None
    head.agreement["floor"] = 0.99  # a --floor override recorded at scoring time
    js = build_audit_report(state, _scan(), price_table=TABLE, base_url="https://x/v1")
    assert js["workloads"][0]["winner"] is None

    head.agreement["floor"] = 0.5
    head.key_ref = None
    js = build_audit_report(state, _scan(), price_table=TABLE, base_url="https://x/v1")
    assert js["workloads"][0]["switch"] == {
        "base_url": "https://x/v1",
        "model": "dep-1",
        "key_ref": None,
    }
    md = render_markdown(js)
    assert "key_ref -:" in md


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (StepState(), "pending"),
        (StepState(dataset_id="ds"), "uploaded"),
        (StepState(dataset_id="ds", run_status="queued"), "queued"),
        (StepState(run_status="completed", deploy_status="deploying"), "deploying"),
        (StepState(run_status="completed", deploy_status="running"), "running"),
        (StepState(deploy_status="running", scored=True), "scored"),
        (StepState(scored=True, error="unreliable: 3 of 4 replay calls failed"), "unreliable"),
        (StepState(error="rejected_preflight: too big"), "rejected_preflight"),
    ],
)
def test_candidate_status_is_the_error_code_else_the_furthest_step(
    step: StepState, expected: str
) -> None:
    assert candidate_status(HEAD_TUNE, step) == expected
    assert candidate_status(SFT_SMALL, step) == expected
    assert candidate_status(HOSTED_FLOOR, step) == "untested"


def test_report_markdown_equals_json_view(tmp_path: Path) -> None:
    state = _state()
    state.halted = {"reason": "budget", "spent_credits": 12.0}
    report = build_audit_report(state, _scan(), price_table=TABLE)
    write_audit_report(report, tmp_path)
    js = json.loads((tmp_path / "audit-report.json").read_text(encoding="utf-8"))
    md = (tmp_path / "audit-report.md").read_text(encoding="utf-8")

    assert js == json.loads(json.dumps(report))
    assert md == render_markdown(js)
    order = ["## Not worth replacing", "## Candidates", "## Switch", "## Artifacts", "## Method"]
    positions = [md.index(section) for section in order]
    assert positions == sorted(positions)
    keep = md[positions[0] : positions[1]]
    assert "| w3 |" in keep
    assert "| w4 |" in keep
    assert "| w1 |" not in keep
    assert "### w1 - REPLACE" in md
    assert "### w2 - NOT YET" in md
    assert "No candidate cleared the floor" in md
    assert "| head_tune (winner) |" in md
    assert "0.990 [0.980, 1.000] n=200" in md
    assert "90 (measured)" in md
    assert "- dataset: 800 rows, 3 redactions, 4 duplicates removed, 1 truncated" in md
    assert 'base_url="https://api.dagnam.ai/v1"' in md
    assert 'model="dep-1"' in md
    assert "run run-1, model version mv-1, deployment dep-1" in md
    assert "`dagnam models download mv-1 <artifact>`" in md
    assert "- halted: {'reason': 'budget', 'spent_credits': 12.0}" in md
    assert "PII scanned for: PII_EMAIL; found: PII_EMAIL: 3" in md
    assert "2026-08-01T00:00:00+00:00 to 2026-08-31T00:00:00+00:00 (30 days)" in md
    assert SECRET not in md
    assert js["price_table_version"] in md
    assert js["generated_at"] in md


def test_markdown_handles_no_winner_no_pii_and_missing_artifacts() -> None:
    state = AuditState(project_id=None)
    state.workloads["w1"] = {HOSTED: StepState(), HEAD: StepState(run_id="run-1")}
    scan = _scan()
    scan["pii"]["counts"] = {}
    md = render_markdown(build_audit_report(state, scan, price_table=TABLE))

    assert "- project: -" in md
    assert "halted" not in md
    assert "## Switch\n\n## Artifacts" in md
    assert "run run-1, model version -, deployment -" in md
    assert "models download" not in md
    assert "found: none found" in md
    assert "| head_tune | - | - | - |" in md


def test_switch_snippet_names_base_url_model_and_key_ref() -> None:
    snippet = render_switch_snippet("dep-1", "w1/head_tune")
    assert 'base_url="https://api.dagnam.ai/v1"' in snippet
    assert 'model="dep-1"' in snippet
    assert "key_ref w1/head_tune" in snippet
    assert "from openai import OpenAI" in snippet
    assert "<deployment key" not in snippet
    assert 'base_url="https://x/v1"' in render_switch_snippet("d", "k", base_url="https://x/v1")
