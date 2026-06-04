"""CLI checkpoint subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture


# ---------------------------------------------------------------- checkpoint


def test_checkpoint_list_empty(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch("dagnam._core.client.DagnamClient.list_checkpoints", return_value=[]):
        run_cli(["checkpoint", "list", "job-1"])
    assert "No checkpoints" in capsys.readouterr().out


def test_checkpoint_list_with_rows(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch(
        "dagnam._core.client.DagnamClient.list_checkpoints",
        return_value=[
            {
                "id": "ck-1",
                "epoch": 3,
                "step": 100,
                "is_best": True,
                "is_final": False,
                "file_size": 2048,
            }
        ],
    ):
        run_cli(["checkpoint", "list", "job-1"])
    out = capsys.readouterr().out
    assert "ck-1" in out
    assert "True" in out


def test_checkpoint_list_apierror_exits(run_cli: CliRunner, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    from dagnam._core.exceptions import APIError

    with mock.patch(
        "dagnam._core.client.DagnamClient.list_checkpoints",
        side_effect=APIError(500, "boom"),
    ):
        with pytest.raises(SystemExit):
            run_cli(["checkpoint", "list", "job-1"])


def test_checkpoint_download(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    with mock.patch("dagnam.download_checkpoint", return_value="/some/path"):
        run_cli(["checkpoint", "download", "job-1"])
    assert "/some/path" in capsys.readouterr().out


def test_checkpoint_download_omits_checkpoint_id_for_latest(
    run_cli: CliRunner, monkeypatch: PytestMonkeyPatch
) -> None:
    with mock.patch("dagnam.download_checkpoint", return_value="/some/path") as download:
        run_cli(["checkpoint", "download", "job-1"])
    download.assert_called_once_with("job-1", None)


def test_checkpoint_download_requests_best_checkpoint(run_cli: CliRunner) -> None:
    with mock.patch("dagnam.download_checkpoint", return_value="/some/path") as download:
        run_cli(["checkpoint", "download", "job-1", "best"])
    download.assert_called_once_with("job-1", None, prefer_best=True)


def test_checkpoint_download_passes_output_dir(run_cli: CliRunner, tmp_path: Path) -> None:
    with mock.patch("dagnam.download_checkpoint", return_value=tmp_path / "cp.pt") as download:
        run_cli(["checkpoint", "download", "job-1", "cp-1", "--output-dir", str(tmp_path)])

    download.assert_called_once_with("job-1", "cp-1", cache_dir=tmp_path)


def test_checkpoint_download_apierror_exits(run_cli: CliRunner) -> None:
    from dagnam._core.exceptions import APIError

    with mock.patch("dagnam.download_checkpoint", side_effect=APIError(500, "boom")):
        with pytest.raises(SystemExit):
            run_cli(["checkpoint", "download", "job-1"])
