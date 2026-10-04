"""state.json round-trips atomically and refuses a schema it does not understand."""

from __future__ import annotations

import json
import os
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
from dagnam.audit.workspace import UnsafeWorkloadsError


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


def test_round_trip_keeps_every_field_and_the_on_disk_shape(tmp_path: Path) -> None:
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
        ({"schema": SCHEMA, "kept_ids": "ds-1"}, "kept_ids must be a JSON array"),
        ({"schema": SCHEMA, "confirmed_gone": "ds-1"}, "confirmed_gone must be a JSON array"),
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


def test_the_project_nonce_is_written_and_read_back(tmp_path: Path) -> None:
    save_state(tmp_path, AuditState(project_nonce="n-1"))
    assert json.loads((tmp_path / STATE_FILE).read_text(encoding="utf-8"))["project_nonce"] == "n-1"
    assert load_state(tmp_path).project_nonce == "n-1"


def test_a_state_written_before_the_project_nonce_still_loads(tmp_path: Path) -> None:
    doc = {"schema": SCHEMA, "project_id": "p1", "workloads": {}}
    (tmp_path / STATE_FILE).write_text(json.dumps(doc))
    state = load_state(tmp_path)
    assert (state.project_id, state.project_nonce) == ("p1", None)


def test_one_command_at_a_time_holds_an_audit_directory(tmp_path: Path) -> None:
    """Two `audit run`s over one directory both saw no `run_id`, and both paid."""
    root = tmp_path / "audit"
    with pytest.raises(FileNotFoundError, match="is not an audit directory"), lock_audit(root):
        pass  # a mistyped directory is refused, never created
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
        ([{"workload_id": "w1", "kind": "bogus"}], "'bogus' is not a valid CandidateKind"),
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


def test_a_retired_candidate_keeps_its_workload_and_kind(tmp_path: Path) -> None:
    """Retiring moved the bare steps into ``retired``; which candidate each had been was lost.

    ``audit status`` then listed a run still billing with no workload beside
    it. The step itself now knows which candidate it is -- from the moment it
    is created or read -- so it still does after the rescan moves it.
    """
    save_state(tmp_path, _state())
    state = load_state(tmp_path)
    fresh = state.candidate("w2", CandidateKind.SFT_SMALL)  # created in this process, never read
    fresh.training_job_id = "job-2"

    # Exactly what a forced rescan does to the workloads it rewrites.
    for workload_id in ("w1", "w2"):
        state.retired.extend(state.workloads.pop(workload_id).values())
    save_state(tmp_path, state)

    on_disk = json.loads((tmp_path / STATE_FILE).read_text(encoding="utf-8"))
    assert [(step["workload_id"], step["kind"]) for step in on_disk["retired"]] == [
        ("w1", "head_tune"),
        ("w1", "hosted_floor"),
        ("w2", "sft_small"),
    ]
    restored = load_state(tmp_path).retired
    assert [(step.workload_id, step.kind) for step in restored] == [
        ("w1", CandidateKind.HEAD_TUNE),
        ("w1", CandidateKind.HOSTED_FLOOR),
        ("w2", CandidateKind.SFT_SMALL),
    ]
    assert (restored[0].dataset_id, restored[2].training_job_id) == ("d1", "job-2")


def test_an_active_step_is_written_as_before_its_place_says_which_candidate_it_is(
    tmp_path: Path,
) -> None:
    """Only a retired step carries the two keys, so an active one reads the same in any version."""
    save_state(tmp_path, _state())
    on_disk = json.loads((tmp_path / STATE_FILE).read_text(encoding="utf-8"))
    step = on_disk["workloads"]["w1"]["candidates"]["head_tune"]
    assert "workload_id" not in step
    assert "kind" not in step
    loaded = load_state(tmp_path).workloads["w1"][CandidateKind.HEAD_TUNE]
    assert (loaded.workload_id, loaded.kind) == ("w1", CandidateKind.HEAD_TUNE)
    assert loaded == _state().workloads["w1"][CandidateKind.HEAD_TUNE]  # it is not part of equality

    on_disk["workloads"]["w1"]["candidates"]["head_tune"]["workload_id"] = "w9"
    (tmp_path / STATE_FILE).write_text(json.dumps(on_disk))
    with pytest.raises(ValueError, match=r"unknown step keys \['workload_id'\] under w1/head_tune"):
        load_state(tmp_path)


def test_a_state_whose_retired_steps_were_written_without_their_candidate_still_loads(
    tmp_path: Path,
) -> None:
    """The shape before this change: a retired entry was the step's own keys and nothing else."""
    old_step = {
        "dataset_id": "ds-old",
        "version_id": "ds-old-v2",
        "split_task_id": "split:1",
        "split_done": True,
        "pii_task_id": "pii:1",
        "pii_agrees": True,
        "base": "BERT small",
        "run_id": "run-old",
        "training_job_id": "job-old",
        "run_status": "running",
        "model_version_id": None,
        "deployment_id": None,
        "key_ref": None,
        "deploy_status": None,
        "scored": None,
        "agreement": None,
        "latency": None,
        "training_cost_credits": 100.0,
        "replay_cost_credits": None,
        "error": None,
        "published_candidate_id": None,
        "published_step": None,
    }
    document = {
        "schema": SCHEMA,
        "project_id": "p1",
        "audit_id": None,
        "price_table_version": "2026-09",
        "workloads": {"w1": {"candidates": {"head_tune": {"dataset_id": "ds-new"}}}},
        "halted": None,
        "retired": [old_step],
        "retired_cost_credits": 120.0,
    }
    (tmp_path / STATE_FILE).write_text(json.dumps(document))

    state = load_state(tmp_path)

    [retired] = state.retired
    assert (retired.workload_id, retired.kind) == (None, None)
    assert (retired.training_job_id, retired.run_status) == ("job-old", "running")
    assert state.retired_cost_credits == 120.0
    save_state(tmp_path, state)  # and it is written back in the current shape
    assert load_state(tmp_path) == state


def test_the_body_of_an_unanswered_audit_create_round_trips_and_older_states_have_none(
    tmp_path: Path,
) -> None:
    state = AuditState(project_nonce="n", pending_audit={"max_credits": 500, "workloads": []})
    save_state(tmp_path, state)
    assert load_state(tmp_path).pending_audit == {"max_credits": 500, "workloads": []}
    document = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    del document["pending_audit"]
    (tmp_path / "state.json").write_text(json.dumps(document), encoding="utf-8")
    assert load_state(tmp_path).pending_audit is None


@pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege on Windows")
def test_a_lock_file_that_is_a_link_is_refused_and_what_it_points_at_is_left_alone(
    tmp_path: Path,
) -> None:
    """Older ``filelock`` releases open the lock with ``O_TRUNC`` and follow a planted link."""
    victim = tmp_path / "victim.txt"
    victim.write_text("precious", encoding="utf-8")
    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "state.json.lock").symlink_to(victim)

    with pytest.raises(UnsafeWorkloadsError), lock_audit(audit):
        pytest.fail("the lock must not be taken through a link")

    assert victim.read_text(encoding="utf-8") == "precious"


def test_kept_ids_round_trip_and_an_older_state_has_none(tmp_path: Path) -> None:
    save_state(tmp_path, AuditState(kept_ids=["ds-1", "proj-1"]))
    assert load_state(tmp_path).kept_ids == ["ds-1", "proj-1"]
    document = json.loads((tmp_path / STATE_FILE).read_text(encoding="utf-8"))
    del document["kept_ids"]
    (tmp_path / STATE_FILE).write_text(json.dumps(document), encoding="utf-8")
    assert load_state(tmp_path).kept_ids == []


def test_confirmed_gone_round_trips_and_an_older_state_has_none(tmp_path: Path) -> None:
    save_state(tmp_path, AuditState(confirmed_gone=["ds-1"]))
    assert load_state(tmp_path).confirmed_gone == ["ds-1"]
    document = json.loads((tmp_path / STATE_FILE).read_text(encoding="utf-8"))
    del document["confirmed_gone"]
    (tmp_path / STATE_FILE).write_text(json.dumps(document), encoding="utf-8")
    assert load_state(tmp_path).confirmed_gone == []
