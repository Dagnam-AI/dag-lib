"""CLI checkpoint subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

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


def test_checkpoint_list_apierror_exits(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    from dagnam._core.exceptions import APIError

    with mock.patch(
        "dagnam._core.client.DagnamClient.list_checkpoints",
        side_effect=APIError(500, "boom"),
    ):
        assert run_cli(["checkpoint", "list", "job-1"]) == 1
    err = capsys.readouterr().err
    assert "Error: the Dagnam API had an internal error (HTTP 500)" in err
    assert "boom" in err


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


def test_checkpoint_download_apierror_exits(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import APIError

    with mock.patch("dagnam.download_checkpoint", side_effect=APIError(500, "boom")):
        assert run_cli(["checkpoint", "download", "job-1"]) == 1
    err = capsys.readouterr().err
    assert "Error: the Dagnam API had an internal error (HTTP 500)" in err
    assert "boom" in err


def test_checkpoint_download_best_with_output_dir(run_cli: CliRunner, tmp_path: Path) -> None:
    with mock.patch("dagnam.download_checkpoint", return_value=tmp_path / "best.pt") as download:
        run_cli(["checkpoint", "download", "job-1", "best", "--output-dir", str(tmp_path)])
    download.assert_called_once_with("job-1", None, cache_dir=tmp_path, prefer_best=True)


def test_checkpoint_download_specific_id(run_cli: CliRunner) -> None:
    with mock.patch("dagnam.download_checkpoint", return_value="/p") as download:
        run_cli(["checkpoint", "download", "job-1", "ck-9"])
    download.assert_called_once_with("job-1", "ck-9")


def test_checkpoint_list_ignores_non_dict_items(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch(
        "dagnam._core.client.DagnamClient.list_checkpoints",
        return_value=["not-a-dict", {"id": "ck-1", "epoch": 1, "step": 2, "file_size": 1024}],
    ):
        run_cli(["checkpoint", "list", "job-1"])
    assert "ck-1" in capsys.readouterr().out


def test_checkpoint_list_handles_bool_and_missing_file_size(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    """A bool or non-numeric ``file_size`` falls back to the 0-byte default."""
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch(
        "dagnam._core.client.DagnamClient.list_checkpoints",
        return_value=[
            {"id": "ck-bool", "epoch": 1, "step": 1, "file_size": True},
            {"id": "ck-str", "epoch": 1, "step": 2, "file_size": "not-a-number"},
            {"id": "ck-missing", "epoch": 1, "step": 3},
        ],
    ):
        run_cli(["checkpoint", "list", "job-1"])
    out = capsys.readouterr().out
    assert "ck-bool" in out
    assert "0.0 B" in out
