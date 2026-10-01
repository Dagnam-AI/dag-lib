"""state.json round-trips atomically and refuses a schema it does not understand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dagnam.audit.candidates import CandidateKind
from dagnam.audit.state import (
    LOCK_FILE,
    SCHEMA,
    STATE_FILE,
    AuditBusyError,
    AuditState,
    StepState,
    load_state,
    lock_audit,
    save_state,
)


def _state() -> AuditState:
    state = AuditState(project_id="p1", price_table_version="2026-09")
    step = state.candidate("w1", CandidateKind.HEAD_TUNE)
    step.dataset_id = "d1"
    step.split_done = True
    step.agreement = {"metric": "exact", "value": 0.5, "ci95": [0.1, 0.9], "n": 4}
    step.training_cost_credits = 12.5
    step.replay_cost_credits = 396.0
    state.candidate("w1", CandidateKind.HOSTED_FLOOR)
    state.halted = {"reason": "budget"}
    return state


def test_missing_file_is_a_fresh_audit(tmp_path: Path) -> None:
    state = load_state(tmp_path)
    assert state == AuditState()
    assert state.schema == SCHEMA
    assert state.workloads == {}
    assert state.halted is None


def test_round_trip_keeps_every_field_and_the_spec_shape(tmp_path: Path) -> None:
    save_state(tmp_path / "audit", _state())
    on_disk = json.loads((tmp_path / "audit" / STATE_FILE).read_text(encoding="utf-8"))

    assert on_disk["schema"] == "dagnam.audit.state/1"
    assert set(on_disk["workloads"]["w1"]["candidates"]) == {"head_tune", "hosted_floor"}
    assert on_disk["workloads"]["w1"]["candidates"]["head_tune"]["dataset_id"] == "d1"
    assert on_disk["halted"] == {"reason": "budget"}
    assert load_state(tmp_path / "audit") == _state()


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path: Path) -> None:
    save_state(tmp_path, AuditState())
    save_state(tmp_path, _state())
    assert [p.name for p in tmp_path.iterdir()] == [STATE_FILE]
    assert load_state(tmp_path).project_id == "p1"


def test_unknown_schema_is_a_hard_error_naming_the_understood_version(tmp_path: Path) -> None:
    (tmp_path / STATE_FILE).write_text(json.dumps({"schema": "dagnam.audit.state/2"}))
    with pytest.raises(ValueError, match=r"dagnam\.audit\.state/2.*dagnam\.audit\.state/1"):
        load_state(tmp_path)


@pytest.mark.parametrize(
    ("document", "match"),
    [
        ([], "the document must be a JSON object"),
        ({"schema": SCHEMA, "workloads": []}, "workloads must be a JSON object"),
        ({"schema": SCHEMA, "workloads": {"w": 1}}, r"workloads\[w\] must be"),
        ({"schema": SCHEMA, "workloads": {"w": {"candidates": {"head_tune": 1}}}}, "candidates"),
        (
            {"schema": SCHEMA, "workloads": {"w": {"candidates": {"head_tune": {"bogus": 1}}}}},
            r"unknown step keys \['bogus'\] under w/head_tune",
        ),
        ({"schema": SCHEMA, "halted": "budget"}, "halted must be a JSON object"),
    ],
)
def test_malformed_documents_are_rejected(tmp_path: Path, document: object, match: str) -> None:
    (tmp_path / STATE_FILE).write_text(json.dumps(document))
    with pytest.raises(ValueError, match=match):
        load_state(tmp_path)


def test_unknown_candidate_kind_is_rejected(tmp_path: Path) -> None:
    doc = {"schema": SCHEMA, "workloads": {"w": {"candidates": {"judge": {}}}}}
    (tmp_path / STATE_FILE).write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="judge"):
        load_state(tmp_path)


def test_candidate_accessor_creates_once() -> None:
    state = AuditState()
    step = state.candidate("w", CandidateKind.SFT_SMALL)
    step.run_id = "r"
    assert state.candidate("w", CandidateKind.SFT_SMALL) is step
    assert state.workloads == {"w": {CandidateKind.SFT_SMALL: StepState(run_id="r")}}


def test_a_state_written_before_the_replay_cost_key_still_loads(tmp_path: Path) -> None:
    doc = {
        "schema": SCHEMA,
        "workloads": {"w1": {"candidates": {"head_tune": {"training_cost_credits": 12.5}}}},
    }
    (tmp_path / STATE_FILE).write_text(json.dumps(doc))
    step = load_state(tmp_path).candidate("w1", CandidateKind.HEAD_TUNE)
    assert (step.training_cost_credits, step.replay_cost_credits) == (12.5, None)


def test_one_command_at_a_time_holds_an_audit_directory(tmp_path: Path) -> None:
    """B13: two `audit run`s over one directory both saw no `run_id`, and both paid."""
    root = tmp_path / "audit"
    with pytest.raises(FileNotFoundError, match="is not an audit directory"), lock_audit(root):
        pass  # M8: a mistyped directory is refused, never created
    assert not root.exists()
    root.mkdir()
    with lock_audit(root):
        second = lock_audit(root)
        with pytest.raises(AuditBusyError, match="is in use by another `dagnam audit` command"):
            second.__enter__()
    with pytest.raises(RuntimeError, match="crash"), lock_audit(root):
        raise RuntimeError("crash")
    with lock_audit(root):  # released on the way out, crash or not
        assert (root / LOCK_FILE).exists()


@pytest.mark.parametrize(
    ("retired", "match"),
    [
        ({}, "retired must be a JSON array"),
        ([1], r"retired\[0\] must be a JSON object"),
        ([{"bogus": 1}], r"unknown step keys.*retired\[0\]"),
    ],
)
def test_invalid_retired_candidates_are_rejected(
    tmp_path: Path, retired: object, match: str
) -> None:
    (tmp_path / STATE_FILE).write_text(json.dumps({"schema": SCHEMA, "retired": retired}))
    with pytest.raises(ValueError, match=match):
        load_state(tmp_path)


def test_retired_candidates_round_trip_without_becoming_active(tmp_path: Path) -> None:
    state = _state()
    state.retired = [StepState(dataset_id="old", training_job_id="old-job")]
    state.retired_cost_credits = 19.0
    save_state(tmp_path, state)
    restored = load_state(tmp_path)
    assert restored == state
    assert list(restored.all_steps())[-1] == state.retired[0]
    assert restored.candidate("old", CandidateKind.HEAD_TUNE) == StepState()
