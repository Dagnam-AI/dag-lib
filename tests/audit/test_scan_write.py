"""Writing a scan into an audit directory: what a rescan keeps, replaces and removes."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest
from tests.audit._records import make_record, make_workload
from tests.audit._scans import FRESH, WINDOW, labels, scan

from dagnam.audit.candidates import TRAINING_CREDITS_MAX, CandidateKind
from dagnam.audit.derive import build_dataset
from dagnam.audit.orchestrate import select_workloads
from dagnam.audit.scan_report import (
    SupersededRunError,
    superseded_workloads,
    write_scan,
)
from dagnam.audit.state import AuditState, StepState, load_state, save_state
from dagnam.audit.steps_train import credits_spent
from dagnam.audit.workspace import write_workload


def test_superseded_workloads_names_every_run_workload_this_scan_would_change(
    tmp_path: Path,
) -> None:
    records = [
        make_record(system=f"Label ticket {i}.", response="ab"[i % 2], trace_id=str(i))
        for i in range(5)
    ]
    built = build_dataset(records, structure_class="enum_label", max_seq_length=2_048)
    assert superseded_workloads(tmp_path, {"w1": built}) == []  # no run, nothing to protect

    write_workload(tmp_path, "w1", built.rows, built.split, built.stats)
    write_workload(tmp_path, "w2", built.rows, built.split, built.stats)
    steps = {CandidateKind.HEAD_TUNE: StepState(dataset_id="ds")}
    save_state(tmp_path, AuditState(workloads={"w1": steps, "w2": steps, "gone": steps}))
    fewer = build_dataset(records[:4], structure_class="enum_label", max_seq_length=2_048)

    # w1 is unchanged; w2's rows would change; "gone" has no files and no rows any more.
    assert superseded_workloads(tmp_path, {"w1": built, "w2": fewer}) == ["w2", "gone"]
    assert superseded_workloads(tmp_path, {"w1": built, "w2": built, "gone": built}) == ["gone"]


@pytest.mark.parametrize("published", [False, True])
def test_forced_scan_never_resumes_superseded_candidates(tmp_path: Path, published: bool) -> None:
    records = [make_record(system=f"Label {i}.", response="ab"[i % 2]) for i in range(5)]
    old = build_dataset(records, structure_class="enum_label", max_seq_length=2_048)
    new = build_dataset(records[:4], structure_class="enum_label", max_seq_length=2_048)
    write_workload(tmp_path, "w1", old.rows, old.split, old.stats)
    step = StepState(
        dataset_id="ds",
        version_id="v",
        run_id="run",
        run_status="completed",
        training_job_id="job",
        model_version_id="model",
        deployment_id="dep",
        key_ref="key",
        scored=True,
        training_cost_credits=12.0,
        replay_cost_credits=7.0,
    )
    save_state(
        tmp_path,
        AuditState(
            audit_id="audit" if published else None,
            workloads={"w1": {CandidateKind.HEAD_TUNE: step}},
        ),
    )
    replay = tmp_path / "workloads/w1/replay-head_tune.jsonl"
    replay.write_text("old reply")
    for suffix in ("json", "md"):
        (tmp_path / f"audit-report.{suffix}").write_text("old score")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    if published:
        with pytest.raises(SupersededRunError, match="new --out"):
            write_scan(
                tmp_path,
                [],
                {"w1": new},
                source="jsonl",
                window=WINDOW,
                price_table=FRESH,
                force=True,
            )
        assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    else:
        write_scan(
            tmp_path, [], {"w1": new}, source="jsonl", window=WINDOW, price_table=FRESH, force=True
        )
        state = load_state(tmp_path)
        assert state.candidate("w1", CandidateKind.HEAD_TUNE) == StepState()
        assert not replay.exists()
        assert not (tmp_path / "audit-report.json").exists()
        assert not (tmp_path / "audit-report.md").exists()
        assert state.retired == [step]
        assert credits_spent(state) == 19.0
        write_scan(
            tmp_path, [], {"w1": new}, source="jsonl", window=WINDOW, price_table=FRESH, force=True
        )
        assert credits_spent(load_state(tmp_path)) == 19.0  # unchanged scans never add it twice
        second = state.candidate("w1", CandidateKind.HEAD_TUNE)
        second.run_id, second.run_status = "new-run", "running"
        save_state(tmp_path, state)
        replay.write_text(
            '{"deployment_id": "dep", "balance_before": 100}\n{ "row": 0, "answer": "yes", "ms": 1.0}\n'
        )
        write_scan(
            tmp_path, [], {"w1": old}, source="jsonl", window=WINDOW, price_table=FRESH, force=True
        )
        assert credits_spent(load_state(tmp_path)) == 19.0 + TRAINING_CREDITS_MAX + 2.0
        assert len(load_state(tmp_path).retired) == 2


@pytest.mark.parametrize("change", ["rows", "added", "added-retired", "removed", "unchanged"])
@pytest.mark.parametrize("force", [False, True])
def test_published_scan_is_protected_before_any_candidate_starts(
    tmp_path: Path, change: str, force: bool
) -> None:
    records = [make_record(system=f"Label {i}.", response="ab"[i % 2]) for i in range(5)]
    dataset = build_dataset(records, structure_class="enum_label", max_seq_length=2_048)
    write_scan(
        tmp_path,
        [make_workload(), replace(make_workload(), id="without-dataset")],
        {"w1": dataset},
        source="jsonl",
        window=WINDOW,
        price_table=FRESH,
    )
    # Publisher.start persists this association before the first candidate is instantiated.
    save_state(tmp_path, AuditState(audit_id="published", workloads={}))
    # A retired local dataset folder outside the published scan is not an active workload.
    write_workload(tmp_path, "retired", dataset.rows, dataset.split, dataset.stats)
    proposed = {"w1": dataset}
    if change == "rows":
        proposed["w1"] = build_dataset(
            records[:4], structure_class="enum_label", max_seq_length=2_048
        )
    elif change == "added":
        proposed["w2"] = dataset
    elif change == "added-retired":
        proposed["retired"] = dataset
    elif change == "removed":
        proposed.clear()
    workloads = [replace(make_workload(), id=key) for key in proposed]
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    if change == "unchanged":
        report = write_scan(
            tmp_path,
            workloads,
            proposed,
            source="jsonl",
            window=WINDOW,
            price_table=FRESH,
            force=force,
        )
        assert (tmp_path / "state.json").read_bytes() == before[tmp_path / "state.json"]
        for path, contents in before.items():
            if "w1" in path.parts:
                assert path.read_bytes() == contents
        # A folder this audit's scan never wrote is not the scan's to remove: it is left, and named.
        assert (tmp_path / "workloads" / "retired" / "dataset.jsonl").exists()
        assert any("workloads/retired" in warning for warning in report.warnings)
    else:
        with pytest.raises(SupersededRunError, match="new --out"):
            write_scan(
                tmp_path,
                workloads,
                proposed,
                source="jsonl",
                window=WINDOW,
                price_table=FRESH,
                force=force,
            )
        assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_a_rescan_removes_the_workloads_it_no_longer_derives(tmp_path: Path) -> None:
    # A rescan left ``workloads/<id>/`` of a workload the new export no longer
    # derived, and ``audit run --workloads <id>`` then trained on the previous
    # export's rows under the new scan report.
    scan(tmp_path, {"w1": labels(5), "w2": labels(5)})
    assert (tmp_path / "workloads/w2/dataset.jsonl").exists()

    scan(tmp_path, {"w1": labels(5)})

    assert sorted(p.name for p in (tmp_path / "workloads").iterdir()) == ["w1"]
    derived = json.loads((tmp_path / "scan-report.json").read_text())["workloads"]
    assert [w["id"] for w in derived if w["dataset"] is not None] == ["w1"]


def test_a_forced_rescan_removes_a_run_workload_it_no_longer_derives_and_keeps_its_handles(
    tmp_path: Path,
) -> None:
    scan(tmp_path, {"w1": labels(5), "w2": labels(5)})
    step = StepState(dataset_id="ds", run_id="run", run_status="completed", deployment_id="dep")
    save_state(tmp_path, AuditState(workloads={"w2": {CandidateKind.HEAD_TUNE: step}}))
    with pytest.raises(SupersededRunError, match="w2"):
        scan(tmp_path, {"w1": labels(5)})
    assert (tmp_path / "workloads/w2/dataset.jsonl").exists()  # refused: nothing is touched

    scan(tmp_path, {"w1": labels(5)}, force=True)

    assert not (tmp_path / "workloads/w2").exists()
    state = load_state(tmp_path)
    assert (state.workloads, state.retired) == ({}, [step])


def test_a_forced_rescan_drops_the_create_body_of_the_rows_it_replaced(tmp_path: Path) -> None:
    """A body built from the old scan would publish workloads the new rows no longer back."""
    scan(tmp_path, {"w1": labels(5)})
    step = StepState(dataset_id="ds", run_id="run", run_status="completed")
    save_state(
        tmp_path,
        AuditState(
            pending_audit={"workloads": []}, workloads={"w1": {CandidateKind.HEAD_TUNE: step}}
        ),
    )

    scan(tmp_path, {"w1": labels(4)}, force=True)

    assert load_state(tmp_path).pending_audit is None


def test_a_scan_that_dies_halfway_leaves_no_scan_report_for_a_run_to_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The forced reset was several separate writes. Stopped between them, the directory
    # held the old scan report over a workload whose rows were new and whose split was
    # still the old one, and ``audit run`` read that as a finished scan.
    old, new = labels(5), labels(4)
    scan(tmp_path, {"w1": old, "w2": old})
    step = StepState(dataset_id="ds", run_id="run", run_status="completed", deployment_id="dep")
    step.training_cost_credits = 12.0
    save_state(tmp_path, AuditState(workloads={"w1": {CandidateKind.HEAD_TUNE: step}}))
    replay = tmp_path / "workloads/w1/replay-head_tune.jsonl"
    replay.write_text('{"deployment_id": "dep", "balance_before": 9}\n')
    written: list[str] = []

    def dies_on_the_second(out_dir: Path, workload_id: str, *args: Any) -> Path:
        if written:
            raise OSError("disk full")
        written.append(workload_id)
        return write_workload(out_dir, workload_id, *args)

    monkeypatch.setattr("dagnam.audit.scan_report.write_workload", dies_on_the_second)
    with pytest.raises(OSError, match="disk full"):
        scan(tmp_path, {"w1": new, "w2": new}, force=True)

    # One workload holds the new rows, the other the old: nothing may read this as a scan.
    assert not (tmp_path / "scan-report.json").exists()
    with pytest.raises(FileNotFoundError, match="dagnam audit scan"):
        select_workloads(tmp_path, None)
    # The state was saved first, so the retired run is still there to cancel and to pay for.
    crashed = load_state(tmp_path)
    assert (crashed.workloads, crashed.retired) == ({}, [step])
    assert credits_spent(crashed, tmp_path) == 12.0

    monkeypatch.undo()
    scan(tmp_path, {"w1": new, "w2": new})  # the same scan again, to the end

    report = json.loads((tmp_path / "scan-report.json").read_text())
    assert [w["dataset"]["rows"] for w in report["workloads"]] == [len(new.rows)] * 2
    for workload_id in ("w1", "w2"):
        folder = tmp_path / "workloads" / workload_id
        assert len((folder / "dataset.jsonl").read_text().splitlines()) == len(new.rows)
        split = json.loads((folder / "split.json").read_text())
        assert split["member_row_indices"] == new.split
    assert not replay.exists()
    finished = load_state(tmp_path)
    assert finished.retired == [step]
    assert credits_spent(finished, tmp_path) == 12.0  # counted once


def test_answers_no_candidate_owns_are_removed_by_the_next_scan(tmp_path: Path) -> None:
    # What a forced rescan leaves when it stops right after saving the state: the retired
    # candidate's answers beside rows a fresh candidate will be replayed on.
    dataset = labels(5)
    scan(tmp_path, {"w1": dataset, "w2": dataset})
    kept = StepState(dataset_id="ds", deployment_id="dep")
    save_state(
        tmp_path,
        AuditState(
            workloads={"w2": {CandidateKind.HEAD_TUNE: kept}},
            retired=[StepState(deployment_id="old-dep")],
        ),
    )
    for workload_id in ("w1", "w2"):
        (tmp_path / "workloads" / workload_id / "replay-head_tune.jsonl").write_text("{}\n")

    scan(tmp_path, {"w1": dataset, "w2": dataset})

    assert not (tmp_path / "workloads/w1/replay-head_tune.jsonl").exists()
    assert (tmp_path / "workloads/w2/replay-head_tune.jsonl").exists()  # a live candidate's


def test_the_scan_report_is_the_last_file_a_scan_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan(tmp_path, {"w1": labels(5)})

    def unrenderable(js: object) -> str:
        raise OSError("disk full")

    monkeypatch.setattr("dagnam.audit.scan_report.render_markdown", unrenderable)
    with pytest.raises(OSError, match="disk full"):
        scan(tmp_path, {"w1": labels(5)})
    assert not (tmp_path / "scan-report.json").exists()
