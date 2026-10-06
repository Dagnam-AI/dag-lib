"""The audit create is asked for again with the body it was first asked with, under the directory's nonce."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from tests.audit._platform import FakePlatform
from tests.audit._publish import start

from dagnam._core.exceptions import APIError
from dagnam._types import JsonObject
from dagnam.audit.publish import Publisher
from dagnam.audit.state import AuditState, load_state

if TYPE_CHECKING:
    from pathlib import Path


def test_the_create_is_keyed_by_the_directorys_nonce(
    publisher: Publisher, platform: FakePlatform, state: AuditState, audit_dir: Path
) -> None:
    state.project_nonce = "nonce-a"
    start(publisher, audit_dir)
    assert platform.audit_nonces == ["nonce-a"]


def test_the_body_is_on_disk_before_it_is_sent_and_gone_once_the_audit_exists(
    publisher: Publisher, platform: FakePlatform, state: AuditState, audit_dir: Path
) -> None:
    platform.publish_errors["create_audit"] = [APIError(0, "Request timed out")]

    start(publisher, audit_dir)

    assert state.audit_id is None
    assert state.pending_audit is not None
    assert load_state(audit_dir).pending_audit == state.pending_audit  # saved, not only held

    start(publisher, audit_dir)

    assert state.audit_id == "audit-1"
    assert state.pending_audit is None


@pytest.mark.parametrize("status", [502, 504, 409])
def test_a_failure_that_may_have_come_after_the_platform_committed_keeps_the_body(
    publisher: Publisher, platform: FakePlatform, state: AuditState, audit_dir: Path, status: int
) -> None:
    """A gateway's 502 or the in-progress 409 is an answer, but not that the body was wrong."""
    platform.publish_errors["create_audit"] = [APIError(status, "gateway")]

    start(publisher, audit_dir)

    assert state.pending_audit is not None
    assert publisher.create_failed is not None


@pytest.mark.parametrize("status", [400, 404, 422])
def test_a_body_the_platform_refused_is_dropped_so_the_next_run_builds_its_own(
    publisher: Publisher, platform: FakePlatform, state: AuditState, audit_dir: Path, status: int
) -> None:
    platform.publish_errors["create_audit"] = [APIError(status, "refused")]
    start(publisher, audit_dir)
    assert state.pending_audit is None


def test_an_answer_with_no_id_is_a_failed_create_that_says_what_was_missing(
    publisher: Publisher,
    platform: FakePlatform,
    state: AuditState,
    audit_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = platform.create_audit

    def no_id(payload: JsonObject, *, resume_nonce: str | None = None) -> JsonObject:
        real(payload, resume_nonce=resume_nonce)
        return {"status": "ok"}

    monkeypatch.setattr(platform, "create_audit", no_id)
    start(publisher, audit_dir)
    assert state.audit_id is None
    assert publisher.create_failed == "the platform's answer to the audit create had no 'id'"


def test_a_create_the_platform_answered_with_a_refusal_drops_its_body_and_halts(
    publisher: Publisher, platform: FakePlatform, state: AuditState, audit_dir: Path
) -> None:
    """An answer is heard: the next run builds a fresh body. A timeout is not an answer (above)."""
    platform.publish_errors["create_audit"] = [APIError(422, "refused")]

    start(publisher, audit_dir)

    assert state.audit_id is None
    assert state.pending_audit is None
    assert publisher.create_failed == "API error 422: refused"


def test_a_rerun_with_other_flags_asks_again_with_the_body_it_first_asked_with(
    platform: FakePlatform, audit_dir: Path
) -> None:
    """The platform replays a create only for the same body, so another ceiling would open a second audit.

    The first ask was lost to Ctrl+C after the platform committed it. The rerun has another
    ``--max-credits`` and ``--floor``; it sends what the first one sent, and finds that audit.
    """
    first_state = AuditState(project_id="proj-1", project_nonce="nonce-a")
    platform.publish_errors["create_audit"] = [APIError(0, "Request timed out")]
    start(Publisher(platform, first_state), audit_dir)
    sent = dict(first_state.pending_audit or {})
    assert sent["max_credits"] == 500

    rerun_state = load_state(audit_dir)
    rerun_state.project_id = "proj-1"
    rerun = Publisher(platform, rerun_state)
    start(rerun, audit_dir, max_credits=900, floor=0.5)

    assert rerun_state.audit_id == "audit-1"
    assert {k: v for k, v in platform.audits[-1].items() if k != "resume"} == sent
    assert platform.audits[-1]["max_credits"] == 500


def test_a_pending_body_that_is_not_an_object_is_refused_when_the_state_is_read(
    audit_dir: Path,
) -> None:
    path = audit_dir / "state.json"
    path.write_text(json.dumps({"schema": "dagnam.audit.state/1", "pending_audit": []}), "utf-8")
    with pytest.raises(ValueError, match="pending_audit must be a JSON object"):
        load_state(audit_dir)
