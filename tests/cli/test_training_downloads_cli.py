"""CLI coverage for the training download subcommands.

Covers ``dagnam training download-code`` and ``dagnam training dag`` - the
handler wiring and the saved-path confirmation message.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, StrCapture


def test_training_download_code_default_out(run_cli: CliRunner, capsys: StrCapture) -> None:
    saved = Path("j1-code.zip")
    download = mock.Mock(return_value=saved)
    with mock.patch("dagnam.download_code", download):
        run_cli(["training", "download-code", "j1"])
    download.assert_called_once_with("j1", out=None)
    assert f"Saved training code to {saved}" in capsys.readouterr().out


def test_training_download_code_with_out(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    saved = tmp_path / "j1-code.zip"
    download = mock.Mock(return_value=saved)
    with mock.patch("dagnam.download_code", download):
        run_cli(["training", "download-code", "j1", "--out", str(tmp_path)])
    download.assert_called_once_with("j1", out=str(tmp_path))
    assert f"Saved training code to {saved}" in capsys.readouterr().out


def test_training_dag_default_out(run_cli: CliRunner, capsys: StrCapture) -> None:
    saved = Path("j1-dag.json")
    download = mock.Mock(return_value=saved)
    with mock.patch("dagnam.download_dag", download):
        run_cli(["training", "dag", "j1"])
    download.assert_called_once_with("j1", out=None)
    assert f"Saved DAG to {saved}" in capsys.readouterr().out


def test_training_dag_with_out(run_cli: CliRunner, capsys: StrCapture, tmp_path: Path) -> None:
    saved = tmp_path / "j1-dag.json"
    download = mock.Mock(return_value=saved)
    with mock.patch("dagnam.download_dag", download):
        run_cli(["training", "dag", "j1", "--out", str(tmp_path)])
    download.assert_called_once_with("j1", out=str(tmp_path))
    assert f"Saved DAG to {saved}" in capsys.readouterr().out
