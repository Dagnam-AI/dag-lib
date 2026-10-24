"""``dagnam audit key``: the winner's deployment key on stdout, everything else on stderr."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import pytest
from tests.cli._audit_dirs import HEAD, HOSTED, SFT
from tests.cli._audit_run import SCAN

from dagnam._types import JsonObject
from dagnam.audit.secrets import SecretStore
from dagnam.audit.state import AuditState, StepState, save_state

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture

KEY = "dk-live-0123456789abcdef"
KEY_REF = "w1/head_tune"
AGREEMENT: JsonObject = {
    "metric": "exact",
    "value": 0.99,
    "ci95": [0.98, 1.0],
    "n": 200,
    "floor": 0.97,
}


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)


def _winner_step(*, key_ref: str | None = KEY_REF) -> StepState:
    return StepState(
        dataset_id="ds-1",
        training_job_id="job-1",
        model_version_id="mv-1",
        deployment_id="dep-1",
        key_ref=key_ref,
        deploy_status="running",
        scored=True,
        agreement=dict(AGREEMENT),
        training_cost_credits=10.0,
    )


def _audit(tmp_path: Path, *, key: str | None = KEY, key_ref: str | None = KEY_REF) -> Path:
    """A scanned, run audit directory whose ``w1`` has a winner (``head_tune``)."""
    root = tmp_path / "audit"
    root.mkdir()
    (root / "scan-report.json").write_text(json.dumps(SCAN), encoding="utf-8")
    state = AuditState(project_id="proj-1")
    state.workloads["w1"] = {HOSTED: StepState(), HEAD: _winner_step(key_ref=key_ref)}
    state.workloads["w2"] = {SFT: StepState(dataset_id="ds-2")}  # started, no winner yet
    save_state(root, state)
    if key is not None:
        SecretStore(root).store(KEY_REF, key)
    return root


def _refused(run_cli: CliRunner, capsys: StrCapture, *argv: str) -> str:
    """Run ``audit key``, expect exit 2 with nothing on stdout; return stderr."""
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "key", *argv])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    return captured.err


def test_prints_only_the_key_on_stdout(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    root = _audit(tmp_path)
    assert run_cli(["audit", "key", str(root), "w1"]) == 0
    captured = capsys.readouterr()
    assert captured.out == KEY + "\n"
    assert KEY not in captured.err


def test_json_prints_the_workload_deployment_key_ref_and_key(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    root = _audit(tmp_path)
    assert run_cli(["audit", "key", str(root), "w1", "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "workload": "w1",
        "deployment_id": "dep-1",
        "key_ref": KEY_REF,
        "key": KEY,
    }
    assert KEY not in captured.err


def test_an_unknown_workload_exits_2_and_names_the_known_ones(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    err = _refused(run_cli, capsys, str(_audit(tmp_path)), "nope")
    assert "nope" in err
    assert "w1" in err
    assert KEY not in err


def test_a_workload_without_a_winner_exits_2(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    err = _refused(run_cli, capsys, str(_audit(tmp_path)), "w2")
    assert "no winner" in err
    assert KEY not in err


def test_a_workload_the_run_never_reached_exits_2(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    err = _refused(run_cli, capsys, str(_audit(tmp_path)), "w3")
    assert "no winner" in err


@pytest.mark.parametrize(
    "kwargs", [{"key": None}, {"key_ref": None}], ids=["key-missing", "no-key-ref"]
)
def test_a_winner_with_no_stored_key_exits_2_and_says_to_rotate_in_the_studio(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture, kwargs: dict[str, str | None]
) -> None:
    err = _refused(run_cli, capsys, str(_audit(tmp_path, **kwargs)), "w1")
    assert "rotate" in err.lower()
    assert "Studio" in err
    assert KEY not in err


def test_the_key_never_reaches_a_log_record(
    run_cli: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = _audit(tmp_path)
    with caplog.at_level(logging.DEBUG):
        assert run_cli(["audit", "key", str(root), "w1"]) == 0
    assert KEY not in caplog.text
    assert all(KEY not in record.getMessage() for record in caplog.records)


def test_a_directory_with_no_scan_report_exits_1(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "key", str(tmp_path), "w1"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "scan-report.json" in captured.err
