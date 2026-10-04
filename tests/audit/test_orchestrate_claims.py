"""A directory run ``--local-only`` and then published hands its resources to the audit."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from tests.audit._chat import Clock, serve_chat, teacher
from tests.audit._platform import FakePlatform
from tests.typing_helpers import RequestsMocker

from dagnam._core.exceptions import APIError
from dagnam.audit.candidates import CandidateKind
from dagnam.audit.orchestrate import describe_failure, run_audit
from dagnam.audit.publish import Publisher
from dagnam.audit.state import DELETED_STATE, AuditDeletedError, AuditState, load_state, save_state

HEAD = CandidateKind.HEAD_TUNE


@pytest.fixture
def run(
    audit_dir: Path, platform: FakePlatform, clock: Clock, requests_mock: RequestsMocker
) -> Callable[..., AuditState]:
    serve_chat(requests_mock, teacher)

    def call(**overrides: Any) -> AuditState:
        settings: dict[str, Any] = {
            "floor": None,
            "workloads": ["w1"],
            "max_credits": 500,
            "wait": True,
            "client": platform,
            "sleep": clock.sleep,
            "now": clock.now,
        }
        settings.update(overrides)
        return run_audit(audit_dir, **settings)

    return call


def test_a_run_over_a_deleted_audit_is_refused_before_anything_is_asked(
    run: Callable[..., AuditState], audit_dir: Path, platform: FakePlatform
) -> None:
    save_state(audit_dir, AuditState(audit_id="audit-1", halted=dict(DELETED_STATE)))

    with pytest.raises(AuditDeletedError, match="deleted audit"):
        run()

    assert platform.call_log == []


def test_a_claim_that_fails_as_a_whole_halts_the_publish_and_the_next_run_asks_again(
    run: Callable[..., AuditState], audit_dir: Path, platform: FakePlatform
) -> None:
    run(wait=False)  # --local-only: no publisher, so the first run's resources carry no audit
    earlier = load_state(audit_dir)
    assert earlier.audit_id is None
    platform.publish_errors["claim_audit_resources"] = [APIError(500, "boom")]
    platform.call_log.clear()

    halted = run(wait=False, publisher=Publisher(platform, load_state(audit_dir)))

    assert halted.halted is not None
    assert halted.halted["reason"] == "publish_failed"
    assert "claim" in str(halted.halted["detail"])
    assert (halted.audit_id, halted.tagged, halted.claim_pending) == ("audit-1", True, True)
    assert halted.unclaimed_ids == []  # nothing recorded as claimed, nothing as refused
    assert "upload_dataset" not in platform.call_log  # nothing new was created beside it
    assert ("audit-1", "error") in platform.halts  # the account does not show it running for good
    assert "still exists" in str(halted.halted["detail"])

    resumed = run(wait=False, publisher=Publisher(platform, load_state(audit_dir)))

    assert resumed.halted is None
    assert resumed.claim_pending is False
    assert len(platform.claims) == 1  # the failed request was never answered, so one answer exists


def test_what_the_platform_refuses_to_claim_is_recorded_as_unclaimed_and_said(
    run: Callable[..., AuditState],
    audit_dir: Path,
    platform: FakePlatform,
) -> None:
    run(wait=False)
    spoken: list[str] = []
    platform.claim_answer = {
        "results": [{"kind": "dataset", "id": "ds-1", "result": "refused", "code": "linked"}]
    }

    state = run(
        wait=False, publisher=Publisher(platform, load_state(audit_dir)), notice=spoken.append
    )

    assert "ds-1" in state.unclaimed_ids
    assert state.kept_ids == []
    assert any("handle them directly" in line for line in spoken)


def test_a_fresh_directory_claims_nothing_but_is_tagged(
    run: Callable[..., AuditState], platform: FakePlatform
) -> None:
    state = run(wait=False, publisher=Publisher(platform, AuditState()))
    assert platform.claims == []
    assert (state.tagged, state.claim_pending) == (True, False)


def test_a_step_that_met_a_document_of_the_wrong_shape_says_so_in_the_halt() -> None:
    assert describe_failure(KeyError("run_id")) == "the platform's answer had no 'run_id'"
    assert "unexpected shape" in describe_failure(
        TypeError("'NoneType' object is not subscriptable")
    )
    assert describe_failure(RuntimeError("boom")) == "RuntimeError: boom"
