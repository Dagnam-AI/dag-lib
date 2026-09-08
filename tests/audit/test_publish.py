"""The publisher: what a run mirrors into the account, and what it does when that fails."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from tests.audit._platform import FakePlatform

from dagnam._core.exceptions import APIError
from dagnam.audit.candidates import HEAD_TUNE, SFT_SMALL, CandidateKind
from dagnam.audit.economics import serving_cost_usd_month
from dagnam.audit.publish import (
    MAX_PENDING,
    STATUS_BY_STEP,
    UNKNOWN_VERSION,
    Publisher,
    installed_version,
    workload_body,
)
from dagnam.audit.state import AuditState, StepState
from dagnam.audit.steps import StepContext
from dagnam.audit.structure import StructureClass

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.typing_helpers import PytestMonkeyPatch

AUDIT_ID = "audit-1"


def _entry(workload_id: str, cls: str, status: str) -> dict[str, Any]:
    return {
        "id": workload_id,
        "structure_class": cls,
        "template_excerpt": "Classify the ticket",
        "calls": 3_000,
        "calls_per_day": 100.0,
        "tokens": {"prompt": 300_000, "completion": 6_000},
        "cost_usd_month": 900.0,
        "latency_ms": {"p50": 400.0, "p95": 900.0},
        "verdict": {"status": status, "ratio": 12.0, "reason": "12.0x over the floor"},
    }


SCAN: dict[str, Any] = {
    "generated_at": "2026-09-06T10:00:00+00:00",
    "price_table_version": "2026-09",
    "workloads": [
        _entry("w1", "enum_label", "candidate"),
        _entry("w2", "json_object", "marginal"),
    ],
}


@pytest.fixture
def state() -> AuditState:
    return AuditState(project_id="proj-1")


@pytest.fixture
def publisher(platform: FakePlatform, state: AuditState) -> Publisher:
    return Publisher(platform, state)


def _start(publisher: Publisher, audit_dir: Path, **overrides: Any) -> None:
    settings: dict[str, Any] = {
        "floor": 0.97,
        "max_credits": 500,
        "sdk_version": "9.9.9",
        **overrides,
    }
    publisher.start(audit_dir, SCAN, ["w1"], **settings)


# ---------------------------------------------------------------------- start


def test_start_publishes_every_workload_and_records_the_audit_id(
    publisher: Publisher, platform: FakePlatform, state: AuditState, audit_dir: Path
) -> None:
    _start(publisher, audit_dir)

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

    _start(publisher, audit_dir)

    w1, w2 = platform.audits[0]["workloads"]
    assert w1["pii_counts"] == {"PII_EMAIL": 7}
    assert w2["pii_counts"] == {}


def test_start_on_a_resumed_run_creates_no_second_audit(
    publisher: Publisher, platform: FakePlatform, state: AuditState, audit_dir: Path
) -> None:
    state.audit_id = AUDIT_ID
    _start(publisher, audit_dir)
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


# ----------------------------------------------------------------- candidates


def test_a_candidate_is_opened_once_and_remembered(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
) -> None:
    _start(publisher, audit_dir)
    ctx, step = make_ctx(), StepState()

    publisher.candidate(ctx, step)
    publisher.candidate(ctx, step)

    assert step.published_candidate_id == "cand-1"
    assert platform.candidates == [
        ("audit-1", {"workload_id": "w1", "kind": "head_tune", "recipe_key": HEAD_TUNE.recipe_key})
    ]


def test_nothing_is_published_before_the_audit_exists(
    publisher: Publisher, platform: FakePlatform, make_ctx: Callable[..., StepContext]
) -> None:
    ctx, step = make_ctx(), StepState()
    publisher.candidate(ctx, step)
    publisher.step(ctx, "upload", step)
    publisher.halt("error")
    assert platform.call_log == []


def test_a_step_publishes_nothing_when_the_candidate_could_not_be_opened(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
) -> None:
    _start(publisher, audit_dir)
    platform.publish_errors["create_audit_candidate"] = [APIError(500, "down")]

    publisher.step(make_ctx(), "upload", StepState(dataset_id="ds-1"))

    assert platform.patches == []


# ---------------------------------------------------------------- step -> status


@pytest.mark.parametrize(
    ("step_name", "status"),
    [
        ("upload", "uploading"),
        ("resolve_version", "uploading"),
        ("split", "splitting"),
        ("wait_split", "splitting"),
        ("pii_scan", "pii_check"),
        ("wait_pii", "pii_check"),
        ("submit", "submitting"),
        ("wait_run", "training"),
        ("resolve_model_version", "training"),
        ("create_deployment", "deploying"),
        ("create_revision", "deploying"),
        ("wait_active", "deploying"),
    ],
)
def test_each_step_maps_to_the_status_it_leaves_behind(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
    step_name: str,
    status: str,
) -> None:
    _start(publisher, audit_dir)
    publisher.step(make_ctx(), step_name, StepState())
    assert platform.patches == [("cand-1", {"step": step_name, "status": status})]


def test_every_step_of_the_frontier_has_a_status() -> None:
    from dagnam.audit.orchestrate import STEPS

    assert {run_step.__name__ for run_step in STEPS} == set(STATUS_BY_STEP)


def test_a_step_carries_the_artifact_ids_it_produced(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
) -> None:
    _start(publisher, audit_dir)
    step = StepState(
        dataset_id="ds-1",
        version_id="ver-1",
        training_job_id="job-1",
        deployment_id="dep-1",
        base="BERT small",
        training_cost_credits=12.0,
    )
    publisher.step(make_ctx(), "submit", step)
    assert platform.patches == [
        (
            "cand-1",
            {
                "step": "submit",
                "status": "submitting",
                "dataset_version_id": "ver-1",
                "training_job_id": "job-1",
                "deployment_id": "dep-1",
                "base_display_name": "BERT small",
                "training_cost_credits": 12.0,
            },
        )
    ]


def test_a_pii_disagreement_and_any_other_error_stop_the_candidate(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
) -> None:
    _start(publisher, audit_dir)
    publisher.step(make_ctx(), "wait_pii", StepState(error="pii_disagreement: server found {...}"))
    publisher.step(make_ctx(), "wait_run", StepState(error="run_failed: out of memory"))

    assert [body["status"] for _, body in platform.patches] == ["pii_disagreement", "failed"]
    assert platform.patches[1][1]["error"] == "run_failed: out of memory"


def test_a_long_error_is_truncated_to_what_the_server_stores(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
) -> None:
    _start(publisher, audit_dir)
    publisher.step(make_ctx(), "wait_run", StepState(error="run_failed: " + "x" * 900))
    assert len(platform.patches[0][1]["error"]) == 500


def test_replay_and_score_walks_through_replaying_and_carries_the_numbers(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
) -> None:
    _start(publisher, audit_dir)
    step = StepState(
        deployment_id="dep-1",
        scored=True,
        agreement={
            "metric": "exact_match",
            "value": 0.98,
            "ci95": [0.94, 1.0],
            "n": 50,
            "floor": 0.97,
            "passes_floor": False,
        },
        latency={"p50": 30.0, "p95": 90.0, "calls": 50, "errors": 0},
        training_cost_credits=12.0,
        replay_cost_credits=4.0,
    )

    publisher.step(make_ctx(), "replay_and_score", step)

    replaying, scored = platform.patches
    assert replaying == ("cand-1", {"step": "replay", "status": "replaying"})
    body = scored[1]
    assert body["step"] == "replay_and_score"
    assert body["status"] == "scored"
    assert body["scored_by"] == "cli"
    assert body["latency"] == {
        "p50": 30.0,
        "p95": 90.0,
        "calls": 50,
        "errors": 0,
        "measured_from": "client",
    }
    assert body["agreement"]["floor"] == 0.97
    assert body["agreement"]["passes_floor"] is False
    assert body["replay_cost_credits"] == 4.0
    assert body["serving_cost_usd_month"] == serving_cost_usd_month(
        "cpu-classifier", calls_per_day=100.0, completion_tokens=6_000, calls=3_000
    )


def test_an_unreliable_replay_still_publishes_its_score_with_the_reason(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
) -> None:
    _start(publisher, audit_dir)
    step = StepState(scored=True, error="unreliable: 30 of 50 replay calls failed")

    publisher.step(make_ctx(), "replay_and_score", step)

    body = platform.patches[1][1]
    assert body["status"] == "scored"
    assert body["error"] == "unreliable: 30 of 50 replay calls failed"


def test_a_workload_the_scan_never_carried_has_no_serving_cost(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
) -> None:
    _start(publisher, audit_dir)
    ctx = make_ctx(workload_id="w9")
    publisher.step(ctx, "replay_and_score", StepState(scored=True))
    assert "serving_cost_usd_month" not in platform.patches[1][1]


def test_the_hosted_floor_is_priced_by_the_report_not_by_a_serving_rate(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
) -> None:
    _start(publisher, audit_dir)
    from dagnam.audit.candidates import HOSTED_FLOOR

    publisher.step(make_ctx(spec=HOSTED_FLOOR), "replay_and_score", StepState(scored=True))
    assert "serving_cost_usd_month" not in platform.patches[1][1]


def test_a_json_candidate_is_priced_as_the_student_that_serves_it(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
) -> None:
    _start(publisher, audit_dir)
    ctx = make_ctx(workload_id="w2", spec=SFT_SMALL, structure_class=StructureClass.JSON_OBJECT)
    publisher.step(ctx, "replay_and_score", StepState(scored=True))
    assert platform.patches[1][1]["serving_cost_usd_month"] == serving_cost_usd_month(
        "gpu-small-llm", calls_per_day=100.0, completion_tokens=6_000, calls=3_000
    )


# -------------------------------------------------------------- halt / resilience


def test_halt_names_the_reason_the_run_stopped(
    publisher: Publisher, platform: FakePlatform, audit_dir: Path
) -> None:
    _start(publisher, audit_dir)
    publisher.halt("budget")
    assert platform.halts == [("audit-1", "budget")]


def test_a_failed_step_is_resent_with_the_next_one_and_never_raises(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _start(publisher, audit_dir)
    ctx = make_ctx()
    platform.publish_errors["patch_audit_candidate"] = [APIError(0, "Connection failed")]

    publisher.step(ctx, "upload", StepState())
    assert platform.patches == []
    assert "the run continues" in caplog.text

    publisher.step(ctx, "split", StepState(version_id="ver-1"))
    assert [body["step"] for _, body in platform.patches] == ["upload", "split"]


def test_a_body_the_server_calls_a_client_bug_is_logged_once_and_dropped(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
) -> None:
    _start(publisher, audit_dir)
    ctx = make_ctx()
    platform.publish_errors["patch_audit_candidate"] = [APIError(422, "extra keys not permitted")]

    publisher.step(ctx, "upload", StepState())
    publisher.step(ctx, "split", StepState())

    assert [body["step"] for _, body in platform.patches] == ["split"]


def test_the_resend_queue_is_capped_rather_than_grown_forever(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _start(publisher, audit_dir)
    ctx, step = make_ctx(), StepState()
    platform.publish_errors["patch_audit_candidate"] = [
        APIError(0, "down") for _ in range(MAX_PENDING + 5)
    ]

    for _ in range(MAX_PENDING + 2):
        publisher.step(ctx, "upload", step)

    assert "dropping the unsent step" in caplog.text


def test_a_failure_that_is_not_an_api_error_is_still_only_a_warning(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    state: AuditState,
    make_ctx: Callable[..., StepContext],
    caplog: pytest.LogCaptureFixture,
) -> None:
    platform.publish_errors["create_audit"] = [TypeError("not JSON serializable")]
    _start(publisher, audit_dir)
    assert state.audit_id is None

    state.audit_id = AUDIT_ID
    platform.publish_errors["patch_audit_candidate"] = [TypeError("not JSON serializable")]
    publisher.step(make_ctx(), "upload", StepState())
    assert platform.patches == []
    assert caplog.text.count("the run continues") == 2


# ----------------------------------------------------------------- local only


def test_a_publisher_without_a_client_records_nothing(
    audit_dir: Path, state: AuditState, make_ctx: Callable[..., StepContext]
) -> None:
    publisher = Publisher(None, state)
    ctx, step = make_ctx(), StepState()

    _start(publisher, audit_dir)
    publisher.candidate(ctx, step)
    publisher.step(ctx, "upload", step)
    publisher.halt("cancelled")

    assert state.audit_id is None
    assert step.published_candidate_id is None


def test_follow_points_the_publisher_at_the_state_the_run_loaded(
    publisher: Publisher, platform: FakePlatform, audit_dir: Path
) -> None:
    live = AuditState(project_id="proj-9")
    publisher.follow(live)
    _start(publisher, audit_dir)
    assert live.audit_id == "audit-1"
    assert platform.audits[0]["project_id"] == "proj-9"


# --------------------------------------------------------------- sdk version


def test_installed_version_reads_the_distribution(monkeypatch: PytestMonkeyPatch) -> None:
    assert installed_version() == __import__("dagnam").__version__

    def missing(_name: str) -> str:
        from importlib.metadata import PackageNotFoundError

        raise PackageNotFoundError(_name)

    monkeypatch.setattr("dagnam.audit.publish.version", missing)
    assert installed_version() == UNKNOWN_VERSION


def test_the_state_round_trips_the_published_ids(tmp_path: Path) -> None:
    from dagnam.audit.state import load_state, save_state

    state = AuditState(project_id="proj-1", audit_id=AUDIT_ID)
    state.candidate("w1", CandidateKind.HEAD_TUNE).published_candidate_id = "cand-1"
    save_state(tmp_path, state)

    read = load_state(tmp_path)
    assert read.audit_id == AUDIT_ID
    assert read.candidate("w1", CandidateKind.HEAD_TUNE).published_candidate_id == "cand-1"
