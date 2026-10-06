"""The scan report's reads of an audit directory: only regular files, never blocked on a pipe."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.audit._fifo import make_fifo, refuse_a_blocking_open_of_a_pipe

from dagnam.audit.scan_report import SCAN_JSON, _written_by_a_scan, superseded_workloads
from dagnam.audit.state import AuditState, save_state
from dagnam.audit.workspace import NotRegularFileError, UnsafeWorkloadsError


def test_a_previous_report_that_is_a_pipe_is_refused_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refuse_a_blocking_open_of_a_pipe(monkeypatch)
    make_fifo(tmp_path / SCAN_JSON)

    with pytest.raises(NotRegularFileError, match=r"scan-report\.json is not a regular file"):
        _written_by_a_scan(tmp_path)


def test_a_previous_report_that_is_a_directory_or_a_link_is_refused(tmp_path: Path) -> None:
    (tmp_path / SCAN_JSON).mkdir()
    with pytest.raises(NotRegularFileError):
        _written_by_a_scan(tmp_path)
    (tmp_path / SCAN_JSON).rmdir()
    real = tmp_path / "real.json"
    real.write_text("{}")
    (tmp_path / SCAN_JSON).symlink_to(real)
    with pytest.raises(UnsafeWorkloadsError):
        _written_by_a_scan(tmp_path)


def test_the_files_a_published_audit_compares_are_read_as_regular_files(tmp_path: Path) -> None:
    save_state(tmp_path, AuditState(audit_id="published", workloads={}))
    make_fifo(tmp_path / SCAN_JSON)

    with pytest.raises(NotRegularFileError):
        superseded_workloads(tmp_path, {})
