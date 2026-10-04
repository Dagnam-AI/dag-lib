"""run_audit asks the platform which contract it runs first, and never mints a resource before it has."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
from typing import Any

import pytest
from tests.audit._chat import Clock, serve_chat, teacher
from tests.audit._platform import FakePlatform
from tests.typing_helpers import RequestsMocker

from dagnam._core.exceptions import APIError, AuthError
from dagnam.audit.candidates import CandidateKind
from dagnam.audit.orchestrate import plan_credits, run_audit, run_audit_held, select_workloads
from dagnam.audit.preflight import PlatformTooOldError
from dagnam.audit.state import AuditState, load_state, save_state
from dagnam.audit.structure import StructureClass
from dagnam.audit.workspace import UnsafeWorkloadsError

HEAD = CandidateKind.HEAD_TUNE


@pytest.fixture(autouse=True)
def installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dagnam.audit.preflight.version", lambda _distribution: "0.4.0")


@pytest.fixture
def run(
    audit_dir: Path, platform: FakePlatform, clock: Clock, requests_mock: RequestsMocker
) -> Callable[..., AuditState]:
    serve_chat(requests_mock, teacher)

    def call(**overrides: Any) -> AuditState:
        settings: dict[str, Any] = {
            "floor": None,
            "workloads": None,
            "max_credits": 500,
            "wait": True,
            "client": platform,
            "sleep": clock.sleep,
            "now": clock.now,
        }
        settings.update(overrides)
        return run_audit(audit_dir, **settings)

    return call


@pytest.mark.parametrize("entry", [run_audit, run_audit_held], ids=["public", "held"])
def test_a_platform_behind_stops_every_entry_point_before_anything_is_created(
    entry: Callable[..., AuditState], audit_dir: Path, platform: FakePlatform, clock: Clock
) -> None:
    """The public ``run_audit`` skipped the check the CLI made: it uploaded, then stopped at the PII check."""
    platform.contracts = "0.3.9"
    with pytest.raises(
        PlatformTooOldError, match="platform_too_old: the platform at https://x runs"
    ):
        entry(
            audit_dir,
            floor=None,
            workloads=None,
            max_credits=500,
            wait=True,
            client=platform,
            sleep=clock.sleep,
            now=clock.now,
        )
    assert platform.call_log == ["get_platform_build"]
    assert not (audit_dir / "state.json").exists()


def test_the_client_is_set_to_resume_its_creates_by_the_run_itself(
    run: Callable[..., AuditState], platform: FakePlatform
) -> None:
    assert platform.resume_creates is False
    run()
    assert platform.resume_creates is True


def test_a_newer_patch_is_said_to_the_notice_and_the_run_goes_on(
    run: Callable[..., AuditState], platform: FakePlatform
) -> None:
    platform.contracts = "0.4.1"
    said: list[str] = []
    state = run(notice=said.append)
    assert state.halted is None
    assert len(said) == 1
    assert "pip install -U dagnam-contracts" in said[0]


def test_nothing_is_said_when_the_contracts_agree(
    run: Callable[..., AuditState], platform: FakePlatform
) -> None:
    said: list[str] = []
    run(notice=said.append)
    assert said == []


def test_a_workload_the_platform_stops_at_the_pii_check_is_told_both_versions(
    run: Callable[..., AuditState], platform: FakePlatform
) -> None:
    """The platform's version reaches the step that stops the workload, so its message can name the gap."""
    platform.contracts = "0.4.1"
    platform.pii_counts = {"ds-1": {"PII_EMAIL": 3}}
    state = run(workloads=["w1"])
    error = state.workloads["w1"][HEAD].error
    assert error is not None
    assert error.startswith("pii_disagreement: ")
    assert "The platform runs dagnam-contracts 0.4.1, this install 0.4.0" in error


def test_a_run_over_a_directory_with_a_project_but_no_nonce_mints_one_first(
    run: Callable[..., AuditState], audit_dir: Path
) -> None:
    """An older state has a project and no nonce; the dataset and audit keys need one."""
    save_state(audit_dir, AuditState(project_id="proj-old"))
    state = run()
    assert state.project_id == "proj-old"
    assert state.project_nonce is not None
    assert load_state(audit_dir).project_nonce == state.project_nonce


def test_a_scan_report_naming_a_folder_outside_workloads_is_refused_as_having_no_rows(
    audit_dir: Path,
) -> None:
    report = json.loads((audit_dir / "scan-report.json").read_text(encoding="utf-8"))
    report["workloads"][0]["id"] = "../outside"
    (audit_dir / "scan-report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match=r"without derived data.*outside"):
        select_workloads(audit_dir, ["../outside"])


@pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege on Windows")
def test_a_workload_folder_that_is_a_link_is_refused_never_read_through(
    audit_dir: Path, tmp_path: Path
) -> None:
    moved = tmp_path / "moved"
    (audit_dir / "workloads" / "w1").rename(moved)
    (audit_dir / "workloads" / "w1").symlink_to(moved, target_is_directory=True)
    with pytest.raises(UnsafeWorkloadsError):
        select_workloads(audit_dir, None)
    with pytest.raises(UnsafeWorkloadsError):
        plan_credits(audit_dir, [("w1", StructureClass.ENUM_LABEL)])


def test_an_account_error_in_the_preflight_ends_the_run_before_anything_is_created(
    audit_dir: Path, platform: FakePlatform, clock: Clock
) -> None:
    platform.build_error = AuthError("expired key")
    with pytest.raises(AuthError):
        run_audit(
            audit_dir,
            floor=None,
            workloads=None,
            max_credits=500,
            wait=True,
            client=platform,
            sleep=clock.sleep,
            now=clock.now,
        )
    assert platform.call_log == ["get_platform_build"]


def test_a_platform_that_cannot_be_asked_does_not_stop_the_run(
    run: Callable[..., AuditState], platform: FakePlatform
) -> None:
    platform.build_error = APIError(503, "down")
    assert run().halted is None
