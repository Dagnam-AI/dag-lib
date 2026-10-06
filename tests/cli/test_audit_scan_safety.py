"""CLI ``audit scan``: planted links, hostile replies and the warnings a scan owes."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import pytest
from tests.cli._scan_exports import (
    CALLS,
    export as _export,
    folder_of as _folder,
    refused as _refused,
    scan as _scan,
)

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)


@pytest.fixture
def scanned(run_cli: CliRunner, tmp_path: Path) -> tuple[Path, Path, Path]:
    """``(export, audit directory, a file outside it that a scan must never touch)``."""
    export, out = _export(tmp_path), tmp_path / "audit"
    assert _scan(run_cli, export, out) == 0
    victim = tmp_path / "outside" / "precious.txt"
    victim.parent.mkdir()
    victim.write_text("precious")
    return export, out, victim


@pytest.mark.parametrize("target", ["folder", "report"])
def test_a_planted_temp_link_is_never_written_through(
    run_cli: CliRunner, scanned: tuple[Path, Path, Path], target: str
) -> None:
    # The temp file had a fixed, guessable name, and was written through whatever it was.
    export, out, victim = scanned
    folder = _folder(out)
    planted = {
        "folder": folder / "dataset.jsonl.tmp",
        "report": out / "scan-report.json.tmp",
    }[target]
    planted.symlink_to(victim)

    assert _scan(run_cli, export, out, "--force") == 0

    assert victim.read_text() == "precious"
    assert not (folder / "dataset.jsonl").is_symlink()
    assert not (out / "scan-report.json").is_symlink()
    assert (folder / "dataset.jsonl").stat().st_size > 0


@pytest.mark.parametrize("where", ["dataset", "report-md"])
def test_a_planted_link_at_a_final_path_is_refused_before_anything_changes(
    run_cli: CliRunner, scanned: tuple[Path, Path, Path], capsys: StrCapture, where: str
) -> None:
    export, out, victim = scanned
    link = _folder(out) / "dataset.jsonl" if where == "dataset" else out / "scan-report.md"
    before = (out / "scan-report.json").read_bytes()
    link.unlink()
    link.symlink_to(victim)
    capsys.readouterr()

    _refused(run_cli, export, out)

    assert "symbolic link" in capsys.readouterr().err
    assert victim.read_text() == "precious"
    assert (out / "scan-report.json").read_bytes() == before  # refused up front


def test_a_planted_link_as_a_workload_folder_is_refused(
    run_cli: CliRunner, scanned: tuple[Path, Path, Path], capsys: StrCapture
) -> None:
    export, out, victim = scanned
    folder = _folder(out)
    outside = victim.parent
    for child in folder.iterdir():
        child.unlink()
    folder.rmdir()
    folder.symlink_to(outside, target_is_directory=True)
    capsys.readouterr()

    _refused(run_cli, export, out)

    assert f"{folder} is a symbolic link" in capsys.readouterr().err  # the link, by its path
    assert [p.name for p in outside.iterdir()] == ["precious.txt"]


def test_one_reply_of_open_brackets_is_text_and_not_the_end_of_the_scan(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    # A JSON decode of 100,000 open brackets raises RecursionError, which nothing caught:
    # one row ended the scan with "unexpected error".
    export = _export(tmp_path, "[" * 100_000, every=600)
    out = tmp_path / "audit"
    capsys.readouterr()

    assert _scan(run_cli, export, out) == 0

    report = json.loads((out / "scan-report.json").read_text(encoding="utf-8"))
    assert report["totals"]["calls"] == CALLS
    assert "unexpected error" not in capsys.readouterr().err


def test_a_row_nested_too_deeply_to_decode_is_one_malformed_row(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    export = _export(tmp_path)
    deep = '{"trace_id": "x", "ts": "2026-08-01T00:00:00Z", "response": ' + "[" * 100_000
    with export.open("a", encoding="utf-8") as sink:
        sink.write(deep + "\n")
    capsys.readouterr()

    assert _scan(run_cli, export, tmp_path / "audit") == 0

    assert "unexpected error" not in capsys.readouterr().err


@pytest.mark.parametrize("calls", [CALLS, 1_600])
def test_reasoning_only_calls_stay_counted_and_priced_and_the_scan_says_so(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture, calls: int
) -> None:
    # A call that ended inside its reasoning is billed all the same. Dropping it cut the
    # calls to 900 and the spend to $450 and turned the verdict into too_few_samples.
    reasoning = {"type": "reasoning", "summary": [{"type": "summary_text", "text": "hmm"}]}
    export = _export(tmp_path, reasoning, every=4, calls=calls)
    out = tmp_path / "audit"
    capsys.readouterr()

    assert _scan(run_cli, export, out) == 0

    report = json.loads((out / "scan-report.json").read_text(encoding="utf-8"))
    (label,) = report["workloads"]
    answerless = calls // 4
    assert label["calls"] == calls
    assert report["totals"]["cost_usd_month"] == pytest.approx(calls * 0.5)
    (warning,) = (w for w in report["warnings"] if "reasoning" in w)
    assert f"{answerless:,} of {calls:,} calls (25% of its spend)" in warning
    assert warning in capsys.readouterr().out  # the CLI prints it
    if calls == CALLS:
        # 900 trainable rows leave a holdout of 180 of the 200 a verdict needs: a fact about the
        # rows, said as such, and not the 900 calls the old reading reported.
        assert "180 holdout rows" in label["verdict"]["reason"]
    else:
        assert label["verdict"]["status"] == "candidate"  # the verdict does not flip
        assert label["dataset"]["rows"] == calls - answerless
