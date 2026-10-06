"""CLI ``audit scan``: what it does with an audit directory it did not entirely make."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import pytest
from tests.audit._fifo import make_fifo, refuse_a_blocking_open_of_a_pipe
from tests.cli._scan_exports import CALLS, export, folder_of, refused, scan

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)
    refuse_a_blocking_open_of_a_pipe(monkeypatch)


@pytest.fixture
def scanned(run_cli: CliRunner, tmp_path: Path) -> tuple[Path, Path]:
    """``(the export, an audit directory a first scan of it wrote)``."""
    source, out = export(tmp_path), tmp_path / "audit"
    assert scan(run_cli, source, out) == 0
    return source, out


def test_an_audit_directory_named_through_a_link_works_and_stays_the_real_one(
    run_cli: CliRunner, tmp_path: Path
) -> None:
    # Temp and home directories are often links: naming one is the user's own choice.
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "audit-link"
    link.symlink_to(real, target_is_directory=True)
    source = export(tmp_path)

    assert scan(run_cli, source, link) == 0
    assert scan(run_cli, source, link) == 0  # and again, over what the first wrote

    assert (real / "scan-report.json").exists()
    assert folder_of(real).parent == real / "workloads"
    assert link.is_symlink()


def test_a_link_inside_the_audit_directory_is_still_refused_naming_the_link(
    run_cli: CliRunner, tmp_path: Path, scanned: tuple[Path, Path], capsys: StrCapture
) -> None:
    source, out = scanned
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    link = out / "workloads"
    (out / "workloads").rename(tmp_path / "moved")
    link.symlink_to(elsewhere, target_is_directory=True)
    through = tmp_path / "reach"
    through.symlink_to(out, target_is_directory=True)  # the way in is a link; the one inside is not
    capsys.readouterr()

    refused(run_cli, source, through)

    assert f"{link} is a symbolic link" in capsys.readouterr().err  # the real path of the link
    assert list(elsewhere.iterdir()) == []


@pytest.mark.parametrize("where", ["report", "meta", "lock"])
def test_a_pipe_where_the_scan_reads_is_refused_not_waited_on(
    run_cli: CliRunner,
    tmp_path: Path,
    scanned: tuple[Path, Path],
    capsys: StrCapture,
    where: str,
) -> None:
    # A read of a FIFO blocks until something writes to it: the scan hung, holding the lock.
    source, out = scanned
    other = export(tmp_path, system="Another job.", name="b.jsonl")
    pipe = {
        "report": out / "scan-report.json",
        "meta": folder_of(out) / "meta.json",
        "lock": out / "state.json.lock",
    }[where]
    pipe.unlink(missing_ok=True)
    make_fifo(pipe)
    capsys.readouterr()

    refused(run_cli, other if where == "meta" else source, out)

    assert f"{pipe} is not a regular file" in capsys.readouterr().err


@pytest.mark.parametrize("what", ["dataset", "report-json", "report-md"])
def test_a_directory_where_a_file_belongs_is_refused_cleanly_before_anything_changes(
    run_cli: CliRunner,
    scanned: tuple[Path, Path],
    capsys: StrCapture,
    what: str,
) -> None:
    source, out = scanned
    folder = folder_of(out)
    target = {
        "dataset": folder / "dataset.jsonl",
        "report-json": out / "scan-report.json",
        "report-md": out / "scan-report.md",
    }[what]
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file() and p != target}
    target.unlink()
    target.mkdir()
    capsys.readouterr()

    refused(run_cli, source, out)

    err = capsys.readouterr().err
    assert f"{target} is not a regular file" in err
    assert "unexpected error" not in err
    assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == before  # nothing touched


def test_a_directory_named_like_an_answer_file_does_not_stop_a_rescan_from_a_dropped_workload(
    run_cli: CliRunner, tmp_path: Path, scanned: tuple[Path, Path], capsys: StrCapture
) -> None:
    _, out = scanned
    dropped = folder_of(out)
    (dropped / "replay-x.jsonl").mkdir()
    (dropped / "replay-x.jsonl" / "kept.txt").write_text("mine")
    other = export(tmp_path, system="Another job.", name="b.jsonl")
    capsys.readouterr()

    assert scan(run_cli, other, out) == 0  # was: PermissionError, on this and every later scan

    assert (dropped / "replay-x.jsonl" / "kept.txt").read_text() == "mine"
    assert sorted(p.name for p in dropped.iterdir()) == ["replay-x.jsonl"]
    report = json.loads((out / "scan-report.json").read_text(encoding="utf-8"))
    assert any(f"{dropped.parent.name}/{dropped.name} holds files" in w for w in report["warnings"])
    assert report["totals"]["calls"] == CALLS
    assert os.path.isdir(dropped / "replay-x.jsonl")
