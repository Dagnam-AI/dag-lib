"""``audit scan``: the usual command-line mistakes end in a plain message, never in a bare "unexpected error"."""

from __future__ import annotations

import gzip
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)


ROW = '{"trace_id": "t1", "ts": "2026-08-01T00:00:00Z", "messages": "hi", "response": "yes"}\n'


@pytest.fixture
def export(tmp_path: Path) -> Path:
    path = tmp_path / "traces.jsonl"
    path.write_text(ROW, encoding="utf-8")
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


def test_a_plain_value_error_from_the_read_stays_redacted_and_keeps_its_debug_traceback(
    run_cli: CliRunner,
    export: Path,
    tmp_path: Path,
    capsys: StrCapture,
    monkeypatch: PytestMonkeyPatch,
) -> None:
    """Only the reader's own user-mistake error is printed: any other ``ValueError`` is a bug."""

    def broken(*args: object, **kwargs: object) -> None:
        raise ValueError("secret internals: 'customer@example.com'")

    monkeypatch.setattr("dagnam.audit.discover_workloads", broken)
    argv = ["audit", "scan", str(export), "--source", "jsonl", "--out", str(tmp_path / "audit")]

    assert _scan(run_cli, tmp_path, str(export), "--source", "jsonl") == 1
    err = capsys.readouterr().err
    assert "unexpected error (ValueError)" in err
    assert "secret internals" not in err

    with pytest.raises(ValueError, match="secret internals"):  # --debug: the traceback
        run_cli(["--debug", *argv])


def test_a_lone_surrogate_in_a_trace_is_an_internal_error_not_a_message_to_print(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    """``UnicodeEncodeError`` is a ``ValueError``; it must not be mistaken for a user's mistake."""
    odd = tmp_path / "surrogate.jsonl"
    odd.write_text(
        '{"trace_id": "t1", "ts": "2026-08-01T00:00:00Z", "messages": "hi", "response": "\\ud800"}\n'
        * 3,
        encoding="utf-8",
    )

    status = _scan(run_cli, tmp_path, str(odd), "--source", "jsonl")

    assert status == 1
    err = capsys.readouterr().err
    assert "unexpected error (UnicodeEncodeError)" in err
    assert "surrogates not allowed" not in err
    with pytest.raises(UnicodeEncodeError):
        run_cli(["--debug", "audit", "scan", str(odd), "--source", "jsonl", "--out", "x"])


def test_the_reader_mistakes_are_value_errors_still_for_library_callers() -> None:
    from dagnam._core.exceptions import DagnamError
    from dagnam.audit.readers import ExportInputError

    assert issubclass(ExportInputError, ValueError)
    assert issubclass(ExportInputError, DagnamError)


def test_a_directory_is_not_an_export(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    folder = tmp_path / "exports"
    folder.mkdir()

    assert _scan(run_cli, tmp_path, str(folder), "--source", "jsonl") == 1
    assert _error_line(capsys) == f"Error: {folder} is a directory: name the export file"


def test_a_file_this_user_may_not_read_says_so(
    run_cli: CliRunner,
    export: Path,
    tmp_path: Path,
    capsys: StrCapture,
    monkeypatch: PytestMonkeyPatch,
) -> None:
    monkeypatch.setattr("os.access", lambda path, mode: False)

    assert _scan(run_cli, tmp_path, str(export), "--source", "jsonl") == 1
    assert _error_line(capsys) == f"Error: {export} cannot be read: permission denied"


def test_a_truncated_gzip_export_says_it_is_damaged(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    whole = gzip.compress(ROW.encode() * 500)
    cut = tmp_path / "traces.jsonl.gz"
    cut.write_bytes(whole[: len(whole) // 2])

    assert _scan(run_cli, tmp_path, str(cut), "--source", "jsonl") == 1
    assert _error_line(capsys) == f"Error: {cut} is not a complete gzip file: re-export it"


def test_a_gzip_export_that_is_not_gzip_never_echoes_its_bytes(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    fake = tmp_path / "traces.jsonl.gz"
    fake.write_bytes(b"SECRETCELL this is not gzip")

    assert _scan(run_cli, tmp_path, str(fake), "--source", "jsonl") == 1
    err = capsys.readouterr().err
    assert f"Error: {fake} is not a complete gzip file: re-export it" in err
    assert "SECRET" not in err


def test_an_end_of_file_error_from_a_file_that_is_not_gzip_stays_an_internal_error(
    run_cli: CliRunner,
    export: Path,
    tmp_path: Path,
    capsys: StrCapture,
    monkeypatch: PytestMonkeyPatch,
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise EOFError("secret internals")

    monkeypatch.setattr("dagnam.audit.discover_workloads", broken)

    assert _scan(run_cli, tmp_path, str(export), "--source", "jsonl") == 1
    err = capsys.readouterr().err
    assert "unexpected error (EOFError)" in err
    assert "secret internals" not in err


def test_a_gzip_export_damaged_in_the_middle_says_it_is_damaged(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    whole = bytearray(gzip.compress(ROW.encode() * 5_000, mtime=0))
    middle = len(whole) // 2
    whole[middle : middle + 8] = b"\xff" * 8
    broken = tmp_path / "traces.jsonl.gz"
    broken.write_bytes(bytes(whole))

    assert _scan(run_cli, tmp_path, str(broken), "--source", "jsonl") == 1
    assert _error_line(capsys) == f"Error: {broken} is not a complete gzip file: re-export it"
