"""``audit scan``: the usual command-line mistakes end in a plain message, never in a bare "unexpected error"."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)


@pytest.fixture
def export(tmp_path: Path) -> Path:
    path = tmp_path / "traces.jsonl"
    path.write_text('{"messages": "hi", "response": "yes"}\n', encoding="utf-8")
    return path


def _scan(run_cli: CliRunner, tmp_path: Path, *argv: str) -> int:
    """The exit status of ``audit scan``, whether it returns or exits."""
    try:
        return run_cli(["audit", "scan", *argv, "--out", str(tmp_path / "audit")])
    except SystemExit as exc:
        return int(exc.code or 0)


def _error_line(capsys: StrCapture) -> str:
    lines = capsys.readouterr().err.splitlines()
    return next(line for line in lines if line.startswith("Error: "))


def test_a_map_on_a_source_that_has_no_columns_to_map_says_so(
    run_cli: CliRunner, export: Path, tmp_path: Path, capsys: StrCapture
) -> None:
    status = _scan(run_cli, tmp_path, str(export), "--source", "langfuse", "--map", "a=b")

    assert status == 1
    assert _error_line(capsys) == "Error: --map applies to the jsonl and csv sources only"
    assert not (tmp_path / "audit").exists()  # nothing was read or written


@pytest.mark.parametrize("source", ["openai", "langsmith"])
def test_the_other_fixed_format_sources_refuse_a_map_the_same_way(
    run_cli: CliRunner, export: Path, tmp_path: Path, capsys: StrCapture, source: str
) -> None:
    assert _scan(run_cli, tmp_path, str(export), "--source", source, "--map", "a=b") == 1
    assert _error_line(capsys) == "Error: --map applies to the jsonl and csv sources only"


def test_a_map_target_the_trace_has_no_field_for_lists_the_fields(
    run_cli: CliRunner, export: Path, tmp_path: Path, capsys: StrCapture
) -> None:
    status = _scan(run_cli, tmp_path, str(export), "--source", "jsonl", "--map", "nonsense=b")

    assert status == 1
    line = _error_line(capsys)
    assert line.startswith("Error: unknown column_map target 'nonsense'; expected one of ")
    assert "messages" in line


def test_an_export_of_a_type_no_reader_knows_names_the_type_and_the_types_it_reads(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    odd = tmp_path / "traces.txt"
    odd.write_text("x", encoding="utf-8")

    status = _scan(run_cli, tmp_path, str(odd), "--source", "jsonl")

    assert status == 1
    assert _error_line(capsys).startswith(
        "Error: unsupported export file type '.txt': expected .jsonl, .json, .parquet or"
    )


def test_a_missing_export_names_the_file(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    gone = tmp_path / "gone.jsonl"

    status = _scan(run_cli, tmp_path, str(gone), "--source", "jsonl")

    assert status == 1
    line = _error_line(capsys)
    assert "No such file or directory" in line
    assert str(gone) in line


def test_a_map_pair_without_an_equals_sign_is_a_usage_error(
    run_cli: CliRunner, export: Path, tmp_path: Path, capsys: StrCapture
) -> None:
    status = _scan(run_cli, tmp_path, str(export), "--source", "jsonl", "--map", "foo")

    assert status == 2  # argparse's usage error, like a bad --window
    assert _error_line(capsys) == "Error: argument --map: --map expects field=column, not 'foo'"
    assert not (tmp_path / "audit").exists()


def test_a_failure_that_is_not_the_users_mistake_still_hides_its_internals(
    run_cli: CliRunner,
    export: Path,
    tmp_path: Path,
    capsys: StrCapture,
    monkeypatch: PytestMonkeyPatch,
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("secret internals: /home/someone/token")

    monkeypatch.setattr("dagnam.audit.discover_workloads", broken)

    status = _scan(run_cli, tmp_path, str(export), "--source", "jsonl")

    assert status == 1
    err = capsys.readouterr().err
    assert "unexpected error (RuntimeError)" in err
    assert "secret internals" not in err
