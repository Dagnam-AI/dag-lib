"""CLI ``audit run`` mirrors the run into the account, and says so."""

from __future__ import annotations

from pathlib import Path
import socket
import sys
from typing import TYPE_CHECKING

import pytest
from tests.audit._chat import OutOfCreditsError, serve_chat
from tests.audit._platform import FakePlatform
from tests.cli._audit_run import Build

from dagnam._core.exceptions import APIError
from dagnam.audit.candidates import CandidateKind
from dagnam.audit.state import load_state
from dagnam.cli.audit_run import PUBLISH_LINE

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, RequestsMocker, StrCapture


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)


def test_run_publishes_the_scan_the_candidates_and_every_step(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform, capsys: StrCapture
) -> None:
    assert run_cli(["audit", "run", str(prepared_dir), "--yes", "--floor", "0.5"]) == 0
    captured = capsys.readouterr()
    out = captured.out
    assert PUBLISH_LINE in out
    # Discoverability: the audit page is named once, on stderr, so `--json`
    # keeps stdout to the report alone.
    assert "published: audit-1 — watch it at https://x/audits/audit-1" in captured.err
    assert "watch it at" not in out

    assert platform.call_log.count("create_audit") == 1
    published = platform.audits[0]
    assert published["source"] == "cli"
    assert published["floor"] == 0.5
    assert published["local_dir_name"] == "audit"
    assert [(w["workload_id"], w["selected"]) for w in published["workloads"]] == [
        ("w1", True),
        ("w2", True),
        ("w3", False),
    ]
    assert [w["pii_counts"] for w in published["workloads"]] == [{"PII_EMAIL": 2}, {}, {}]

    state = load_state(prepared_dir)
    assert state.audit_id == "audit-1"
    assert [kind for _, body in platform.candidates for kind in [body["kind"]]] == [
        "head_tune",
        "sft_small",
    ]
    head = [body for cid, body in platform.patches if cid == "cand-1"]
    assert [body["step"] for body in head] == [
        "upload",
        "resolve_version",
        "split",
        "wait_split",
        "pii_scan",
        "wait_pii",
        "submit",
        "wait_run",
        "resolve_model_version",
        "create_deployment",
        "create_revision",
        "wait_active",
        "replay",
        "replay_and_score",
    ]
    scored = head[-1]
    assert scored["status"] == "scored"
    assert scored["scored_by"] == "cli"
    assert scored["latency"]["measured_from"] == "client"
    assert scored["agreement"]["floor"] == 0.5
    assert scored["serving_cost_usd_month"] > 0
    assert scored["replay_cost_credits"] == 4.0
    assert platform.halts == []

    # A resumed run republishes nothing but its resume: the audit exists and no step moved.
    calls = list(platform.call_log)
    assert run_cli(["audit", "run", str(prepared_dir), "--yes"]) == 0
    assert platform.call_log == [*calls, "resume_audit"]


def test_run_local_only_opens_no_audit_route_and_says_nothing_about_the_account(
    run_cli: CliRunner,
    prepared_dir: Path,
    platform: FakePlatform,
    build: Build,
    capsys: StrCapture,
    monkeypatch: PytestMonkeyPatch,
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network")

    platform.forbid_publishing = True  # any /api/v1/audits call is an AssertionError
    monkeypatch.setattr(socket, "socket", refuse)

    assert (
        run_cli(["audit", "run", str(prepared_dir), "--yes", "--local-only", "--floor", "0.5"]) == 0
    )

    captured = capsys.readouterr()
    out = captured.out
    assert PUBLISH_LINE not in out
    assert "published to your account" not in out
    assert "watch it at" not in captured.err
    assert platform.forbidden_attempts == []  # the tripwire itself, not just its effect
    assert build.reads == 1  # it still uploads and trains there, so the platform is still asked
    assert platform.audits == []
    assert not [call for call in platform.call_log if "audit" in call]
    assert load_state(prepared_dir).audit_id is None


def test_publishing_a_directory_that_ran_local_only_hands_its_ids_to_the_new_audit(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform
) -> None:
    """The earlier run's resources carry no audit: the audit is asked to claim them, once."""
    assert (
        run_cli(["audit", "run", str(prepared_dir), "--yes", "--local-only", "--floor", "0.5"]) == 0
    )
    recorded = load_state(prepared_dir)
    assert recorded.audit_id is None

    assert run_cli(["audit", "run", str(prepared_dir), "--yes", "--floor", "0.5"]) == 0

    ((audit_id, entries),) = platform.claims
    assert audit_id == "audit-1"
    claimed = {(str(e["kind"]), str(e["id"])) for e in entries}
    assert ("dataset", "ds-1") in claimed
    assert {kind for kind, _ in claimed} == {"dataset", "training_job", "deployment", "project"}
    assert not any(kind == "model_version" for kind, _ in claimed)  # a version follows its run
    assert platform.call_log.index("create_audit") < platform.call_log.index(
        "claim_audit_resources"
    )
    state = load_state(prepared_dir)
    assert (state.kept_ids, state.unclaimed_ids) == ([], [])  # the platform claimed every one
    assert (state.tagged, state.claim_pending) == (True, False)

    assert run_cli(["audit", "run", str(prepared_dir), "--yes", "--floor", "0.5"]) == 0
    assert len(platform.claims) == 1  # a resumed run does not claim again


def test_run_halted_tells_the_account_why(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform
) -> None:
    with pytest.raises(SystemExit):
        run_cli(["audit", "run", str(prepared_dir), "--yes", "--max-credits", "160"])
    assert platform.halts == [("audit-1", "budget")]
    # w1 submitted; the submit w2's budget check refused is not published as a
    # step that happened, so exactly one `submit` reached the account.
    assert [body["step"] for _, body in platform.patches].count("submit") == 1


def test_an_account_that_runs_dry_mid_replay_stops_like_the_budget_refusal(
    run_cli: CliRunner,
    prepared_dir: Path,
    platform: FakePlatform,
    requests_mock: RequestsMocker,
    capsys: StrCapture,
) -> None:
    """Exit 1 with the budget halt; the account hears `budget`, never a scored or failed candidate."""

    def empty(_messages: list[dict[str, str]]) -> str:
        raise OutOfCreditsError

    serve_chat(requests_mock, empty)
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(prepared_dir), "--yes", "--floor", "0.5"])

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert (
        "audit halted: budget: the account ran out of credits during the w1/head_tune replay" in err
    )
    assert "add credits, then run `dagnam audit run` again to resume where it stopped" in err
    assert "unreliable" not in err
    assert platform.halts == [("audit-1", "budget")]
    assert "scored" not in [body.get("status") for _, body in platform.patches]
    step = load_state(prepared_dir).workloads["w1"][CandidateKind.HEAD_TUNE]
    assert (step.scored, step.error) == (None, None)


def test_a_resumed_run_that_stops_again_publishes_its_own_halt(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform
) -> None:
    """The second run un-halted the audit when it started, so its halt is a new one."""
    for _ in range(2):
        with pytest.raises(SystemExit):
            run_cli(["audit", "run", str(prepared_dir), "--yes", "--max-credits", "160"])
    assert platform.halts == [("audit-1", "budget"), ("audit-1", "budget")]
    assert platform.call_log.count("resume_audit") == 1  # the first run created the audit


def test_on_an_older_platform_a_run_that_stops_again_does_not_repeat_the_halt(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform
) -> None:
    """No resume route and nothing published before the halt: the account still shows the first."""
    platform.resume_route = False
    for _ in range(2):
        with pytest.raises(SystemExit):
            run_cli(["audit", "run", str(prepared_dir), "--yes", "--max-credits", "160"])
    assert platform.halts == [("audit-1", "budget")]


def test_a_run_that_crashes_publishes_the_halt_before_it_raises(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform
) -> None:
    platform.submit_errors = [RuntimeError("socket reset")]
    assert run_cli(["audit", "run", str(prepared_dir), "--yes"]) == 1
    assert platform.halts == [("audit-1", "error")]


def test_a_publish_outage_halts_the_run_before_anything_is_uploaded(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform
) -> None:
    """Without its audit a run's uploads would be untagged, so nothing is created until it exists."""
    platform.publish_errors["create_audit"] = [APIError(503, "audit service down")]

    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(prepared_dir), "--yes", "--floor", "0.5"])

    assert exc.value.code == 1
    assert platform.candidates == []
    for created in ("upload_dataset", "create_foundation_run", "create_deployment"):
        assert created not in platform.call_log
    state = load_state(prepared_dir)
    assert state.audit_id is None
    assert state.halted is not None
    assert state.halted["reason"] == "publish_failed"


def test_the_consent_line_is_exactly_what_the_account_is_told_it_may_keep() -> None:
    assert PUBLISH_LINE == (
        "  published to your account: progress and the report (workload ids, verdicts, spend,"
        " masked excerpts, the audit directory's name; never rows or keys);"
        " 'audit delete' removes them; --local-only keeps them here"
    )


def test_the_run_after_an_outage_publishes_first_and_every_step_reaches_the_account(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform
) -> None:
    """The audit service was down for the first run; the second creates the audit, then everything."""
    platform.publish_errors["create_audit"] = [APIError(503, "audit service down")]
    with pytest.raises(SystemExit):
        run_cli(["audit", "run", str(prepared_dir), "--yes", "--floor", "0.5"])

    assert run_cli(["audit", "run", str(prepared_dir), "--yes", "--floor", "0.5"]) == 0

    assert platform.call_log.index("create_audit") < platform.call_log.index("upload_dataset")
    assert set(platform.dataset_tags.values()) == {"audit-1"}  # tagged at creation
    head = [body for cid, body in platform.patches if cid == "cand-1"]
    assert [body["step"] for body in head][:3] == ["upload", "resolve_version", "split"]
    assert head[-1]["status"] == "scored"
