"""What a run mirrors into the account once the audit exists: candidates and their steps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from tests.audit._platform import FakePlatform
from tests.audit._publish import start

from dagnam._core.exceptions import APIError
from dagnam.audit.candidates import HEAD_TUNE, SFT_SMALL
from dagnam.audit.economics import serving_cost_usd_month
from dagnam.audit.publish import STATUS_BY_STEP, Publisher
from dagnam.audit.state import StepState
from dagnam.audit.structure import StructureClass

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dagnam.audit.steps import StepContext


# ----------------------------------------------------------------- candidates


def test_a_candidate_is_opened_once_and_remembered(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
) -> None:
    start(publisher, audit_dir)
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
    start(publisher, audit_dir)
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
    start(publisher, audit_dir)
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
    start(publisher, audit_dir)
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
    start(publisher, audit_dir)
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
    start(publisher, audit_dir)
    publisher.step(make_ctx(), "wait_run", StepState(error="run_failed: " + "x" * 900))
    assert len(platform.patches[0][1]["error"]) == 500


def test_replay_and_score_walks_through_replaying_and_carries_the_numbers(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
) -> None:
    start(publisher, audit_dir)
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
    start(publisher, audit_dir)
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
    start(publisher, audit_dir)
    ctx = make_ctx(workload_id="w9")
    publisher.step(ctx, "replay_and_score", StepState(scored=True))
    assert "serving_cost_usd_month" not in platform.patches[1][1]


def test_the_hosted_floor_is_priced_by_the_report_not_by_a_serving_rate(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
) -> None:
    start(publisher, audit_dir)
    from dagnam.audit.candidates import HOSTED_FLOOR

    publisher.step(make_ctx(spec=HOSTED_FLOOR), "replay_and_score", StepState(scored=True))
    assert "serving_cost_usd_month" not in platform.patches[1][1]


def test_a_json_candidate_is_priced_as_the_student_that_serves_it(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
) -> None:
    start(publisher, audit_dir)
    ctx = make_ctx(workload_id="w2", spec=SFT_SMALL, structure_class=StructureClass.JSON_OBJECT)
    publisher.step(ctx, "replay_and_score", StepState(scored=True))
    assert platform.patches[1][1]["serving_cost_usd_month"] == serving_cost_usd_month(
        "gpu-small-llm", calls_per_day=100.0, completion_tokens=6_000, calls=3_000
    )


# --------------------------------------------- nothing in the publisher can raise


def test_a_step_the_publisher_does_not_know_is_a_warning_not_a_crash(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
    caplog: pytest.LogCaptureFixture,
) -> None:
    start(publisher, audit_dir)
    publisher.step(make_ctx(), "a_step_from_the_future", StepState())
    assert platform.patches == []
    assert "the a_step_from_the_future body failed" in caplog.text
