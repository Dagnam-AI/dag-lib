"""CLI dataset subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture


# ---------------------------------------------------------------- dataset


def test_dataset_list_empty(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch("dagnam._core.client.DagnamClient.list_datasets", return_value=[]):
        run_cli(["dataset", "list"])
    assert "No datasets found" in capsys.readouterr().out


def test_dataset_list_with_rows(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch(
        "dagnam._core.client.DagnamClient.list_datasets",
        return_value=[
            {
                "id": "ds-1",
                "name": "Iris",
                "format": "csv",
                "num_samples": 150,
                "dataset_type": "tabular",
            }
        ],
    ):
        run_cli(["dataset", "list"])
    out = capsys.readouterr().out
    assert "ds-1" in out
    assert "Iris" in out


def test_dataset_list_json_forwards_filters_and_overrides(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    list_datasets = mock.Mock(return_value=[{"id": "ds-1"}])
    with mock.patch("dagnam._core.client.DagnamClient") as client:
        client.return_value.list_datasets = list_datasets
        run_cli(
            [
                "dataset",
                "list",
                "--type",
                "tabular",
                "--search",
                "iris",
                "--api-url",
                "https://example.test",
                "--api-key",
                "key",
                "--json",
            ]
        )
    client.assert_called_once_with("https://example.test", "key")
    list_datasets.assert_called_once_with(type="tabular", search="iris")
    assert json.loads(capsys.readouterr().out) == [{"id": "ds-1"}]


def test_dataset_list_autherror_exits(
    run_cli: CliRunner, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    from pathlib import Path

    monkeypatch.delenv("DAGNAM_API_KEY", raising=False)
    # Redirect config to an empty location so auth resolution fails.
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", Path(tmp_path) / "missing.json")
    with pytest.raises(SystemExit):
        run_cli(["dataset", "list"])


def test_dataset_list_apierror_exits(run_cli: CliRunner, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    from dagnam._core.exceptions import APIError

    with mock.patch(
        "dagnam._core.client.DagnamClient.list_datasets",
        side_effect=APIError(500, "boom"),
    ):
        with pytest.raises(SystemExit):
            run_cli(["dataset", "list"])


def test_dataset_info(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch(
        "dagnam._core.client.DagnamClient.get_dataset_meta",
        return_value={
            "id": "ds-1",
            "schema": {"col1": "int"},
            "class_names": ["a", "b"],
        },
    ):
        run_cli(["dataset", "info", "ds-1"])
    out = capsys.readouterr().out
    assert "id: ds-1" in out
    assert "col1: int" in out
    assert "class_names: a, b" in out


def test_dataset_info_redacts_signed_download_url_by_default(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch(
        "dagnam._core.client.DagnamClient.get_dataset_meta",
        return_value={
            "id": "ds-1",
            "download_url": "https://signed.example/file?token=secret",
        },
    ):
        run_cli(["dataset", "info", "ds-1"])

    out = capsys.readouterr().out
    assert "download_url: <redacted>" in out
    assert "token=secret" not in out


def test_dataset_info_apierror_exits(run_cli: CliRunner, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    from dagnam._core.exceptions import APIError

    with mock.patch(
        "dagnam._core.client.DagnamClient.get_dataset_meta",
        side_effect=APIError(500, "boom"),
    ):
        with pytest.raises(SystemExit):
            run_cli(["dataset", "info", "ds-1"])


def test_dataset_download(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch("dagnam.load_dataset", return_value=None):
        run_cli(["dataset", "download", "ds-1"])
    assert "downloaded" in capsys.readouterr().out


def test_dataset_download_passes_output_dir_to_loader(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    output_dir = tmp_path / "downloads"
    with mock.patch("dagnam.load_dataset", return_value=None) as load_dataset:
        run_cli(["dataset", "download", "ds-1", "--output-dir", str(output_dir)])

    load_dataset.assert_called_once_with("ds-1", cache_dir=str(output_dir), show_progress=False)
    assert str(output_dir / "ds-1") in capsys.readouterr().out


def test_dataset_download_no_progress_passes_loader_flag(
    run_cli: CliRunner, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    output_dir = tmp_path / "downloads"
    with mock.patch("dagnam.load_dataset", return_value=None) as load_dataset:
        run_cli(["dataset", "download", "ds-1", "--output-dir", str(output_dir), "--no-progress"])

    load_dataset.assert_called_once_with("ds-1", cache_dir=str(output_dir), show_progress=False)


def test_dataset_download_apierror_exits(
    run_cli: CliRunner, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    from dagnam._core.exceptions import APIError

    with mock.patch("dagnam.load_dataset", side_effect=APIError(500, "boom")):
        with pytest.raises(SystemExit):
            run_cli(["dataset", "download", "ds-1"])


def test_dataset_info_writes_output_file(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    output = tmp_path / "ds.json"
    with mock.patch(
        "dagnam._core.client.DagnamClient.get_dataset_meta",
        return_value={"id": "ds-1", "name": "Iris"},
    ):
        run_cli(["dataset", "info", "ds-1", "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8"))["id"] == "ds-1"


def test_dataset_info_json_mode(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch(
        "dagnam._core.client.DagnamClient.get_dataset_meta",
        return_value={"id": "ds-1", "name": "Iris"},
    ):
        run_cli(["dataset", "info", "ds-1", "--json"])
    assert json.loads(capsys.readouterr().out) == {"id": "ds-1", "name": "Iris"}
