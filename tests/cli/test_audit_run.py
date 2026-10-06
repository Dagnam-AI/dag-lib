"""CLI ``audit run``: the upload listing, the confirmation, the frontier, the report."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from tests.audit._platform import FakePlatform
from tests.cli._audit_run import SCAN, workload

from dagnam.audit.readers.messages import TOOL_CALL_NOTE
from dagnam.audit.state import load_state, lock_audit
from dagnam.audit.steps_train import credits_spent
from dagnam.cli.audit_run import _render_run

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)


def test_run_lists_exactly_what_will_be_uploaded_and_needs_yes(
    run_cli: CliRunner,
    prepared_dir: Path,
    platform: FakePlatform,
    capsys: StrCapture,
    monkeypatch: PytestMonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # not a terminal
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(prepared_dir)])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    listing, _, _ = captured.out.partition("credit ceiling")
    assert "About to upload (redacted, derived rows only; raw traces stay here):" in listing
    assert (
        "  w1 (enum_label): 20 rows (split {'train': 16, 'eval_holdout': 4});"
        " redactions: PII_EMAIL: 2 in 2 rows; scanned for PII_EMAIL, PII_PHONE,"
        " PII_PAYMENT_CARD, PII_NATIONAL_ID"
    ) in listing
    assert "  w2 (json_object): 20 rows" in listing
    assert "redactions: none found in 0 rows" in listing
    assert "w3" not in listing
    assert "  to: a new private project 'workload-audit-audit'" in listing
    # No --max-credits, so the ceiling is the plan's own estimate rounded up to
    # 100 -- per workload a 120-credit training ceiling plus 5 for a 4-row replay.
    assert (
        "credit ceiling: 300 (the plan's estimate, rounded up to 100; --max-credits sets your own)"
    ) in captured.out
    assert "refusing to upload without confirmation on a non-interactive terminal" in captured.err
    assert f"dagnam audit run {prepared_dir} --yes" in captured.err
    assert platform.call_log == []  # nothing left the machine


def test_run_declined_at_the_prompt_uploads_nothing(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform, tty: None, capsys: StrCapture
) -> None:
    with mock.patch("builtins.input", return_value="no"), pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(prepared_dir), "--workloads", "w1", "--max-credits", "130"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "w1 (enum_label)" in captured.out
    assert "w2" not in captured.out
    assert "credit ceiling: 130 (training and the metered holdout replay)" in captured.out
    assert "confirmation not received" in captured.err
    assert platform.call_log == []


def test_run_confirmed_runs_the_frontier_and_writes_the_report(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform, tty: None, capsys: StrCapture
) -> None:
    # Four holdout rows give a wide interval, so the floor is lowered to see a winner.
    with mock.patch("builtins.input", return_value="yes"):
        assert run_cli(["audit", "run", str(prepared_dir), "--floor", "0.5"]) == 0
    out = capsys.readouterr().out
    assert platform.call_log[0] == "create_project"
    assert platform.submits == 2
    # The run's client is the one whose creates a re-run finds again.
    assert getattr(platform, "resume_creates", None) is True
    report = json.loads((prepared_dir / "audit-report.json").read_text(encoding="utf-8"))
    assert report["schema"] == "dagnam.audit.report/1"
    assert report["project_id"] == "proj-1"
    w1, w2, w3 = report["workloads"]
    assert w1["winner"]["kind"] == "head_tune"
    assert w1["switch"] == {"base_url": "https://x/v1", "model": "dep-1", "key_ref": "w1/head_tune"}
    assert w2["winner"]["kind"] == "sft_small"
    assert w3["candidates"] == []
    assert "dk-secret" not in json.dumps(report)
    md = (prepared_dir / "audit-report.md").read_text(encoding="utf-8")
    assert "### w1 - REPLACE" in md
    assert "dk-secret" not in md
    assert "w1: REPLACE - head_tune at $" in out
    assert "switch model dep-1" in out
    assert f"Report: {prepared_dir / 'audit-report.md'}" in out
    assert "  to: existing project" not in out  # the project was created by this run

    # A second run resumes: the listing names the existing project, and the one
    # platform call is the resume of its audit -- no step moved.
    calls = list(platform.call_log)
    assert run_cli(["audit", "run", str(prepared_dir), "--yes", "--json"]) == 0
    printed = json.loads(capsys.readouterr().out.partition("\n{")[2].join(["{", ""]))
    assert printed["workloads"][0]["winner"]["kind"] == "head_tune"
    assert platform.call_log == [*calls, "resume_audit"]


def test_run_json_listing_then_report(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform, capsys: StrCapture
) -> None:
    platform.revision_final = "deploying"  # no candidate reaches an endpoint: NOT YET
    assert run_cli(["audit", "run", str(prepared_dir), "--yes", "--floor", "0.5", "--json"]) == 0
    out = capsys.readouterr().out
    assert "  to: a new private project" in out
    report = json.loads(out[out.index("\n{") + 1 :])
    assert report["workloads"][0]["winner"] is None
    assert report["workloads"][0]["candidates"][1]["status"] == "deploy_timeout"


def test_run_halted_exits_nonzero_with_the_reason(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform, capsys: StrCapture
) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(prepared_dir), "--yes", "--max-credits", "130"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "audit halted: budget" in captured.err
    assert f"dagnam audit status {prepared_dir}" in captured.err
    assert (prepared_dir / "audit-report.md").exists()
    # w1 fits (a 120-credit ceiling and a 5-credit replay); after its 104 credits, w2 does not.
    assert platform.submits == 1

    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(prepared_dir), "--yes", "--max-credits", "130", "--json"])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    error = json.loads(out[out.index("\n{") + 1 :])
    assert error["error"].startswith("audit halted: budget")
    assert error["hint"] == f"dagnam audit status {prepared_dir}"


def test_a_stopped_candidate_says_why_on_the_terminal_not_only_in_the_report(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform, capsys: StrCapture
) -> None:
    """A PII stop printed ``w1: NOT YET ... (head_tune: pii_disagreement)`` and exited 0.

    The code named the stop and nothing named its cause or its cure: those were
    only in ``state.json``. Every candidate's recorded reason now goes under its
    workload's line, whatever stopped it, and is in ``audit-report.json`` for a script.
    """
    platform.pii_pass_list = [*platform.pii_pass_list, "PII_SECRET"]
    platform.pii_counts = {"ds-1": {"PII_SECRET": 185}}  # w1's dataset; w2 is clean
    platform.revision_final = "failed"  # and w2 gets as far as a deployment that fails

    assert run_cli(["audit", "run", str(prepared_dir), "--yes", "--floor", "0.5"]) == 0

    out = capsys.readouterr().out.splitlines()
    at = out.index(
        "w1: NOT YET - No candidate was measured in this audit (head_tune: pii_disagreement)."
    )
    assert out[at + 1] == (
        "  head_tune: pii_disagreement: the platform runs a newer privacy contract than these"
        " rows were redacted with: it found PII_SECRET (185) in classes the scan did not look"
        " for. `pip install -U dagnam-contracts` (or `pip install -U dagnam`), then scan again."
        " `dagnam audit delete <audit-dir>` removes the uploaded rows and the whole audit."
    )
    assert out[at + 2].startswith("w2: NOT YET")
    assert out[at + 3] == "  sft_small: deploy_failed: no gpu"
    assert platform.submits == 1  # the stop is a stop: w1 never trained
    report = json.loads((prepared_dir / "audit-report.json").read_text(encoding="utf-8"))
    assert report["workloads"][0]["candidates"][1]["error"] == out[at + 1].removeprefix(
        "  head_tune: "
    )


def test_run_no_wait_returns_early_and_says_how_to_resume(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform, capsys: StrCapture
) -> None:
    assert run_cli(["audit", "run", str(prepared_dir), "--yes", "--no-wait"]) == 0
    captured = capsys.readouterr()
    assert platform.submits == 1
    assert "get_foundation_run" not in platform.call_log
    assert f"Next: dagnam audit run {prepared_dir} --yes" in captured.err
    # A run still training is not a candidate that missed the floor.
    assert "w1: NOT YET - No candidate was measured in this audit (head_tune: queued)." in (
        captured.out
    )


def test_run_refuses_unknown_workloads_missing_scan_and_nothing_to_run(
    run_cli: CliRunner,
    prepared_dir: Path,
    platform: FakePlatform,
    tmp_path: Path,
    capsys: StrCapture,
) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(prepared_dir), "--yes", "--workloads", "w9"])
    assert exc.value.code == 1
    assert "workloads not in scan-report.json: ['w9']" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(tmp_path / "nowhere"), "--yes", "--json"])
    assert exc.value.code == 1
    assert (
        "is not an audit directory: run `dagnam audit scan`"
        in (json.loads(capsys.readouterr().out)["error"])
    )

    (tmp_path / "unscanned").mkdir()
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(tmp_path / "unscanned"), "--yes"])
    assert exc.value.code == 1
    assert "scan-report.json not found: run `dagnam audit scan` first" in capsys.readouterr().err

    scan = {**SCAN, "workloads": [workload("w3", "free_text", "not_audited")]}
    (prepared_dir / "scan-report.json").write_text(json.dumps(scan), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(prepared_dir), "--yes"])
    assert exc.value.code == 1
    assert "nothing to run" in capsys.readouterr().err
    assert platform.call_log == []


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--floor", "1.5", "--floor expects an agreement lower bound above 0 and at most 1"),
        ("--floor", "0", "--floor expects an agreement lower bound above 0 and at most 1"),
        ("--floor", "high", "--floor expects an agreement lower bound above 0 and at most 1"),
        ("--max-credits", "-5", "--max-credits expects a whole number of credits"),
        ("--max-credits", "5.5", "--max-credits expects a whole number of credits"),
    ],
)
def test_a_bad_floor_or_ceiling_fails_before_anything_is_uploaded(
    run_cli: CliRunner,
    prepared_dir: Path,
    platform: FakePlatform,
    capsys: StrCapture,
    flag: str,
    value: str,
    message: str,
) -> None:
    """The server holds the same bounds; catching them here beats a dropped 422 per publish."""
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(prepared_dir), "--yes", flag, value])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert platform.call_log == []


def test_run_refuses_a_directory_another_command_holds(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform, capsys: StrCapture
) -> None:
    """A second `audit run` over one directory would submit every run a second time."""
    with lock_audit(prepared_dir), pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(prepared_dir), "--yes", "--json"])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "is in use by another `dagnam audit` command" in json.loads(out)["error"]
    assert platform.call_log == []


def test_the_directory_is_held_from_the_listing_to_the_end_of_the_run(
    run_cli: CliRunner,
    prepared_dir: Path,
    platform: FakePlatform,
    tty: None,
    capsys: StrCapture,
    tmp_path: Path,
) -> None:
    """Nothing else can touch the directory while the user reads the listing.

    A second `audit run` would have both pass the prompt and both submit, and
    an `audit scan` would replace the rows between what was listed and what is
    uploaded. Both are refused for as long as the first holds it, and the first
    still runs what it listed.
    """
    argv = ["audit", "run", str(prepared_dir), "--floor", "0.5"]
    export = tmp_path / "export.jsonl"
    row = {
        "trace_id": "t1",
        "ts": "2026-09-01T00:00:00Z",
        "messages": [{"role": "user", "content": "hi"}],
        "response": "a",
    }
    export.write_text(json.dumps(row) + "\n", encoding="utf-8")
    refused: list[int | str | None] = []
    uploaded_meanwhile: list[str] = []

    def meanwhile(_prompt: str) -> str:
        """This run is at its prompt; another run and a scan both try the directory."""
        for other in (
            [*argv, "--yes"],
            ["audit", "scan", str(export), "--source", "jsonl", "--out", str(prepared_dir)],
        ):
            with pytest.raises(SystemExit) as exc:
                run_cli(other)
            refused.append(exc.value.code)
        uploaded_meanwhile.extend(platform.call_log)
        return "yes"

    with mock.patch("builtins.input", side_effect=meanwhile):
        assert run_cli(argv) == 0

    assert refused == [1, 1]
    assert uploaded_meanwhile == []  # nothing reached the platform while the listing was open
    assert capsys.readouterr().err.count("is in use by another `dagnam audit` command") == 2
    assert platform.submits == 2  # the first run went on to submit what it had listed, once
    assert platform.call_log.count("create_project") == 1
    assert credits_spent(load_state(prepared_dir), prepared_dir) <= 300  # the ceiling it listed


def test_the_directory_is_free_again_once_a_run_ends_or_is_declined(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform, tty: None
) -> None:
    with mock.patch("builtins.input", return_value="no"), pytest.raises(SystemExit):
        run_cli(["audit", "run", str(prepared_dir)])
    with lock_audit(prepared_dir):  # released on the way out of a declined prompt
        pass
    assert run_cli(["audit", "run", str(prepared_dir), "--yes", "--floor", "0.5"]) == 0
    with lock_audit(prepared_dir):  # and after a run
        pass


def test_a_tool_call_winner_is_told_what_its_replacement_returns(tmp_path: Path) -> None:
    # The same sentence the scan warning and the report's switch carry.
    winner = {
        "kind": "sft_small",
        "cost_usd_month": 4.0,
        "agreement_lo": 0.97,
        "deployment_id": "d",
    }
    workload = {
        "id": "w1",
        "candidates": [winner],
        "winner": winner,
        "verdict": {"status": "candidate"},
        "response_mode": "tool_call",
    }
    text = _render_run(tmp_path)({"workloads": [workload, {**workload, "response_mode": "text"}]})
    assert text.count(f"\n  {TOOL_CALL_NOTE}\n") == 1
