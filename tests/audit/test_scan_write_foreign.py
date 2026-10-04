"""A rescan next to what it never wrote: foreign folders, links, an edited report, odd ids."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.audit._scans import labels, scan

from dagnam.audit.workspace import UnsafeWorkloadsError, write_workload


def _warnings(out_dir: Path) -> list[str]:
    return json.loads((out_dir / "scan-report.json").read_text())["warnings"]


def test_a_rescan_leaves_a_directory_it_never_wrote_and_names_it(tmp_path: Path) -> None:
    # ``--out .`` in a repository with its own ``workloads/`` folder lost every folder in it.
    scan(tmp_path, {"w1": labels(5), "w2": labels(5)})
    mine = tmp_path / "workloads/prod-api/manifests/deploy.yaml"
    mine.parent.mkdir(parents=True)
    mine.write_text("kind: Deployment\n")
    readme = tmp_path / "workloads/batch-jobs/README.md"
    readme.parent.mkdir()
    readme.write_text("not a scan's\n")

    scan(tmp_path, {"w1": labels(5)})

    assert mine.read_text() == "kind: Deployment\n"
    assert readme.read_text() == "not a scan's\n"
    assert not (tmp_path / "workloads/w2").exists()  # the one this audit's scan wrote
    named = [w for w in _warnings(tmp_path) if "left alone" in w]
    assert sorted(w.split()[0] for w in named) == ["workloads/batch-jobs", "workloads/prod-api"]


def test_a_workload_folder_holding_a_file_the_scan_did_not_write_is_kept_whole(
    tmp_path: Path,
) -> None:
    scan(tmp_path, {"w1": labels(5), "w2": labels(5)})
    (tmp_path / "workloads/w2/notes.txt").write_text("a person's notes\n")
    (tmp_path / "workloads/w2/replay-head_tune.jsonl").write_text("{}\n")

    scan(tmp_path, {"w1": labels(5)})

    assert sorted(p.name for p in (tmp_path / "workloads/w2").iterdir()) == ["notes.txt"]
    assert any("workloads/w2 holds files" in w for w in _warnings(tmp_path))


def test_a_scan_report_that_cannot_be_read_proves_nothing_was_written_by_a_scan(
    tmp_path: Path,
) -> None:
    scan(tmp_path, {"w1": labels(5), "w2": labels(5)})
    (tmp_path / "scan-report.json").write_text("{not json")

    scan(tmp_path, {"w1": labels(5)})

    assert (tmp_path / "workloads/w2/dataset.jsonl").exists()
    assert any("workloads/w2" in w for w in _warnings(tmp_path))
    assert any(
        "scan-report.json of an earlier scan could not be read" in w for w in _warnings(tmp_path)
    )


def test_a_first_scan_has_no_earlier_report_to_complain_about(tmp_path: Path) -> None:
    scan(tmp_path, {"w1": labels(5)})
    assert not any("could not be read" in w for w in _warnings(tmp_path))


def _edit_report(out_dir: Path, *extra: object) -> None:
    path = out_dir / "scan-report.json"
    report = json.loads(path.read_text())
    report["workloads"].extend(extra)
    path.write_text(json.dumps(report))


def test_an_edited_report_with_one_bad_entry_still_lets_the_good_ones_count(tmp_path: Path) -> None:
    scan(tmp_path, {"w1": labels(5), "w2": labels(5)})
    _edit_report(tmp_path, {"id": ["not", "a", "string"], "dataset": {}}, "loose", {"dataset": {}})

    scan(tmp_path, {"w1": labels(5)})

    assert not (tmp_path / "workloads/w2").exists()  # listed by a good entry: removed


def test_an_edited_report_that_lists_a_folder_that_is_not_a_plain_name_is_skipped_not_fatal(
    tmp_path: Path,
) -> None:
    scan(tmp_path, {"w1": labels(5)})
    odd = tmp_path / "workloads" / "my project"
    odd.mkdir()
    (odd / "notes.txt").write_text("mine")
    _edit_report(tmp_path, {"id": "my project", "dataset": {"rows": 1}})

    scan(tmp_path, {"w1": labels(5)})  # was: ValueError, on every scan until the report was fixed

    assert (odd / "notes.txt").read_text() == "mine"
    assert any("workloads/my project" in w for w in _warnings(tmp_path))


def test_a_listed_folder_the_scan_cannot_prove_it_wrote_loses_nothing(tmp_path: Path) -> None:
    # An edited report names a sibling folder holding files that happen to carry the scan's
    # names. The folder's own meta.json is the proof it is a scan's: without it, hands off.
    scan(tmp_path, {"w1": labels(5)})
    sibling = tmp_path / "workloads" / "prod-api"
    sibling.mkdir()
    for name in ("dataset.jsonl", "split.json", "meta.json", "deploy.yaml"):
        (sibling / name).write_text(f"mine: {name}")
    _edit_report(tmp_path, {"id": "prod-api", "dataset": {"rows": 1}})

    scan(tmp_path, {"w1": labels(5)})

    assert sorted(p.name for p in sibling.iterdir()) == [
        "dataset.jsonl",
        "deploy.yaml",
        "meta.json",
        "split.json",
    ]
    assert (sibling / "dataset.jsonl").read_text() == "mine: dataset.jsonl"
    assert any("workloads/prod-api holds files" in w for w in _warnings(tmp_path))


def test_a_symlinked_workloads_directory_is_refused_and_its_target_untouched(
    tmp_path: Path,
) -> None:
    elsewhere = tmp_path / "elsewhere"
    for name in ("photos-2024", "thesis"):
        (elsewhere / name).mkdir(parents=True)
        (elsewhere / name / "keep.txt").write_text(name)
    out = tmp_path / "proj"
    out.mkdir()
    (out / "workloads").symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(UnsafeWorkloadsError, match="symbolic link"):
        scan(out, {"w1": labels(5)})

    assert sorted(p.name for p in elsewhere.iterdir()) == ["photos-2024", "thesis"]
    assert (elsewhere / "thesis/keep.txt").read_text() == "thesis"
    assert not (out / "scan-report.json").exists()


@pytest.mark.parametrize("derived", [True, False])
def test_a_symlinked_workload_directory_is_refused_before_anything_is_written(
    tmp_path: Path, derived: bool
) -> None:
    scan(tmp_path, {"w1": labels(5), "w2": labels(5)})
    before = (tmp_path / "scan-report.json").read_bytes()
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "keep.txt").write_text("mine")
    folder = tmp_path / ("workloads/w2" if derived else "workloads/w1")
    for child in folder.iterdir():
        child.unlink()
    folder.rmdir()
    folder.symlink_to(target, target_is_directory=True)

    # w2 is derived again, or (not ``derived``) w1 is listed and no longer derived.
    proposed = {"w1": labels(5), "w2": labels(5)} if derived else {"w2": labels(5)}
    with pytest.raises(UnsafeWorkloadsError, match="symbolic link"):
        scan(tmp_path, proposed)

    assert sorted(p.name for p in target.iterdir()) == ["keep.txt"]
    assert (tmp_path / "scan-report.json").read_bytes() == before  # nothing was changed


def test_a_symlink_nothing_here_derived_or_listed_is_left_alone_and_named(tmp_path: Path) -> None:
    scan(tmp_path, {"w1": labels(5)})
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "keep.txt").write_text("mine")
    (tmp_path / "workloads/link").symlink_to(target, target_is_directory=True)

    scan(tmp_path, {"w1": labels(5)})

    assert (target / "keep.txt").read_text() == "mine"
    assert (tmp_path / "workloads/link").is_symlink()
    assert any("workloads/link" in w for w in _warnings(tmp_path))


@pytest.mark.parametrize("workload_id", ["../escape", "a/b", "..", ".", "", "/abs", "a\\b"])
def test_a_workload_id_that_is_not_one_plain_name_cannot_leave_the_directory(
    tmp_path: Path, workload_id: str
) -> None:
    out = tmp_path / "proj"
    out.mkdir()
    with pytest.raises(ValueError, match="not a plain name"):
        scan(out, {workload_id: labels(5)})
    with pytest.raises(ValueError, match="not a plain name"):
        write_workload(out, workload_id, [], {}, {})

    assert [p.name for p in tmp_path.iterdir()] == ["proj"]
    assert not list(out.iterdir())
