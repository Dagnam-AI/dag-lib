"""``python -m dagnam_record status|purge``: output, exit codes, safety and no trace content."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import subprocess
import sys
from typing import Any, List

from _api_support import envelope, write_data
import pytest

import dagnam_record
from dagnam_record import _cli, _files
from dagnam_record._writer import COUNT_KEYS

LEAK = "TOPLEAK-PROMPT-TEXT"
RECORD = Path(dagnam_record.__file__).parent.parent


def make_directory(root: Path) -> Path:
    directory = root / "traces"
    write_data(
        directory,
        "2026-10-12.jsonl",
        [envelope(LEAK, 1_791_797_400.0), envelope(LEAK, 1_791_797_500.0)],
    )
    write_data(directory, "2026-10-13.jsonl", [envelope(LEAK, 1_791_883_800.0)])
    (directory / ".status.json").write_text(
        '{"started": 1791797400, "expired": false, "counts": {"written": 3, "dropped_full": 1,'
        ' "dropped_large": 2, "dropped_error": 3, "dropped_expired": 4, "dropped_partial": 5}}',
        encoding="utf-8",
    )
    return directory


def run(arguments: List[str], capsys: pytest.CaptureFixture[str]) -> Any:
    code = _cli.main(arguments)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_status_prints_the_directory_counts_times_and_every_drop_counter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = make_directory(tmp_path)
    code, out, err = run(["status", "--dir", str(directory)], capsys)
    assert (code, err) == (0, "")
    assert out.splitlines() == [
        f"directory: {directory}",
        "files: 2",
        "lines: 3",
        "first: 2026-10-12T09:30:00.000Z",
        "last: 2026-10-13T09:30:00.000Z",
        "started: 2026-10-12T09:30:00Z",
        "expired: no",
        "written: 3",
        "dropped_full: 1",
        "dropped_large: 2",
        "dropped_error: 3",
        "dropped_expired: 4",
        "dropped_partial: 5",
    ]
    assert [line.split(":")[0] for line in out.splitlines()[7:]] == list(COUNT_KEYS)


def test_status_without_a_status_file_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = write_data(
        tmp_path / "t", "2026-10-12.jsonl", [envelope("a", 1_791_797_400.0)]
    ).parent
    code, out, _ = run(["status", "--dir", str(directory)], capsys)
    assert code == 0
    assert "counts: n/a (no readable .status.json)" in out
    assert "started: n/a" in out
    assert "expired: n/a" in out
    (directory / ".status.json").write_text('{"expired": true, "counts": {}}', encoding="utf-8")
    assert "expired: yes" in run(["status", "--dir", str(directory)], capsys)[1]


def test_nothing_the_recorder_wrote_is_ever_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = make_directory(tmp_path)
    write_data(directory, "2026-10-14.jsonl", [f'{{"ts": "{LEAK}"}}'])
    for command in (["status"], ["purge"]):
        _, out, err = run([*command, "--dir", str(directory)], capsys)
        assert LEAK not in out + err


def test_status_exit_codes_for_a_missing_or_unrelated_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run(["status", "--dir", str(tmp_path / "absent")], capsys)
    assert (code, out) == (1, "")
    assert "cannot use" in err
    assert "No such file" in err
    write_data(tmp_path / "other", "notes.jsonl", ["x"])
    code, out, err = run(["status", "--dir", str(tmp_path / "other")], capsys)
    assert (code, out) == (1, "")
    assert "holds no recorder file" in err


def test_purge_deletes_the_recorder_files_and_says_which(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = make_directory(tmp_path)
    write_data(directory, "notes.jsonl", ["mine"])
    code, out, err = run(["purge", "--dir", str(directory)], capsys)
    assert (code, err) == (0, "")
    assert out.splitlines() == [
        "deleted 2026-10-12.jsonl",
        "deleted 2026-10-13.jsonl",
        "deleted .status.json",
        f"deleted 3 file(s) from {directory}",
    ]
    assert sorted(path.name for path in directory.iterdir()) == ["notes.jsonl"]


def test_purge_refuses_an_unrelated_directory_and_deletes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_data(tmp_path, "notes.jsonl", ["mine"])
    code, out, err = run(["purge", "--dir", str(tmp_path)], capsys)
    assert (code, out) == (1, "")
    assert "refused" in err
    assert "holds no recorder file" in err
    assert (tmp_path / "notes.jsonl").exists()


def test_purge_of_a_missing_directory_is_safe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run(["purge", "--dir", str(tmp_path / "absent")], capsys)
    assert (code, err) == (0, "")
    assert out == f"nothing to delete: {tmp_path / 'absent'} does not exist\n"
    assert not (tmp_path / "absent").exists()


def test_purge_refuses_a_symlinked_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    real = make_directory(tmp_path)
    (tmp_path / "link").symlink_to(real)
    code, _, err = run(["purge", "--dir", str(tmp_path / "link")], capsys)
    assert code == 1
    assert "refused" in err
    assert "symlink" in err
    assert len(list(real.iterdir())) == 3


def test_purge_exits_1_when_a_file_cannot_be_deleted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = make_directory(tmp_path)
    real_unlink = os.unlink

    def flaky(path: str) -> None:
        if path.endswith("10-12.jsonl"):
            raise PermissionError(path)
        real_unlink(path)

    monkeypatch.setattr(os, "unlink", flaky)
    code, out, err = run(["purge", "--dir", str(directory)], capsys)
    assert code == 1
    assert "could not delete 2026-10-12.jsonl" in err
    assert "deleted 2 file(s)" in out


def test_dir_may_come_before_or_after_the_command_and_defaults_to_the_home(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    directory = make_directory(tmp_path / ".dagnam")
    assert directory == tmp_path / ".dagnam" / "traces"
    assert run(["status"], capsys)[1].startswith(f"directory: {directory}\n")
    assert run(["--dir", "~/.dagnam/traces", "status"], capsys)[1].startswith(
        f"directory: {directory}\n"
    )
    assert run(["status", "--dir", "~/.dagnam/traces"], capsys)[1].startswith(
        f"directory: {directory}\n"
    )
    other = make_directory(tmp_path / "elsewhere")
    assert run(["--dir", str(other), "status"], capsys)[1].startswith(f"directory: {other}\n")
    assert run(["--dir", str(tmp_path / "nope"), "status", "--dir", str(other)], capsys)[0] == 0
    assert _files.DEFAULT_DIR == "~/.dagnam/traces"


@pytest.mark.parametrize("arguments", [[], ["frobnicate"], ["status", "--bogus"], ["--dir"]])
def test_usage_errors_exit_2(arguments: List[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        _cli.main(arguments)
    assert stopped.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_help_and_version_exit_0(capsys: pytest.CaptureFixture[str]) -> None:
    for flag, text in (
        ("--help", "status"),
        ("--version", f"dagnam-record {dagnam_record.__version__}"),
    ):
        with pytest.raises(SystemExit) as stopped:
            _cli.main([flag])
        assert stopped.value.code == 0
        assert text in capsys.readouterr().out


def test_the_module_runs_as_a_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = make_directory(tmp_path)
    monkeypatch.setattr(sys, "argv", ["dagnam_record", "status", "--dir", str(directory)])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("dagnam_record", run_name="__main__")
    assert stopped.value.code == 0
    assert "files: 2" in capsys.readouterr().out


def run_process(arguments: List[str], home: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed interpreter and arguments, no untrusted input
        [sys.executable, "-m", "dagnam_record", *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env={"PYTHONPATH": str(RECORD), "PATH": "/usr/bin:/bin", "HOME": str(home)},
    )


def test_python_dash_m_status_then_purge_with_a_temporary_home(tmp_path: Path) -> None:
    make_directory(tmp_path / ".dagnam")
    shown = run_process(["status"], tmp_path)
    assert (shown.returncode, shown.stderr) == (0, "")
    assert "lines: 3" in shown.stdout
    assert LEAK not in shown.stdout
    purged = run_process(["purge"], tmp_path)
    assert purged.returncode == 0
    assert "deleted 3 file(s)" in purged.stdout
    again = run_process(["purge"], tmp_path)
    assert again.returncode == 1
    assert "holds no recorder file" in again.stderr
    assert run_process([], tmp_path).returncode == 2
