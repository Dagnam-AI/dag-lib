"""CLI training-jobs subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, StrCapture


# ---------------------------------------------------------------- training jobs


def test_training_create_passes_hyperparameters(run_cli: CliRunner, capsys: StrCapture) -> None:
    create = mock.Mock(return_value={"id": "j1", "status": "pending"})
    with mock.patch("dagnam.create_training_job", create):
        run_cli(
            [
                "training",
                "create",
                "p1",
                "--epochs",
                "2",
                "--batch-size",
                "32",
                "--learning-rate",
                "0.001",
                "--optimizer",
                "adam",
                "--loss-function",
                "cross_entropy",
                "--dataset-id",
                "ds1",
                "--max-duration-seconds",
                "600",
                "--confirm-resource-warning",
            ]
        )
    kwargs = create.call_args.kwargs
    assert create.call_args.args == ("p1",)
    assert kwargs["epochs"] == 2
    assert kwargs["batch_size"] == 32
    assert kwargs["learning_rate"] == 0.001
    assert kwargs["optimizer"] == "adam"
    assert kwargs["training_dataset_id"] == "ds1"
    assert kwargs["max_duration_seconds"] == 600
    assert kwargs["confirm_resource_warning"] is True
    assert '"id": "j1"' in capsys.readouterr().out


def test_training_create_parses_config_overrides_file(run_cli: CliRunner, tmp_path: Path) -> None:
    overrides = tmp_path / "cfg.json"
    overrides.write_text('{"logging_config": {"log_frequency": 5}}', encoding="utf-8")
    create = mock.Mock(return_value={"id": "j1"})
    with mock.patch("dagnam.create_training_job", create):
        run_cli(
            [
                "training",
                "create",
                "p1",
                "--epochs",
                "1",
                "--batch-size",
                "8",
                "--learning-rate",
                "0.01",
                "--optimizer",
                "sgd",
                "--loss-function",
                "mse",
                "--dataset-id",
                "ds1",
                "--config",
                f"@{overrides}",
            ]
        )
    assert create.call_args.kwargs["config_overrides"] == {"logging_config": {"log_frequency": 5}}


def test_training_create_bad_config_exits_cleanly(run_cli: CliRunner, capsys: StrCapture) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli(
            [
                "training",
                "create",
                "project-1",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--learning-rate",
                "0.1",
                "--optimizer",
                "adam",
                "--loss-function",
                "mse",
                "--dataset-id",
                "dataset-1",
                "--config",
                "{bad-json",
            ]
        )

    assert exc.value.code == 1
    assert "Could not read --config JSON" in capsys.readouterr().err


def test_training_attach_help_documents_replay_only_exit(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run_cli(["training", "attach", "--help"])
    assert excinfo.value.code == 0
    assert "replay existing events and exit" in capsys.readouterr().out


def test_training_list_concise_and_verbose(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = {
        "items": [
            {
                "id": "j1",
                "status": "running",
                "framework": "pytorch",
                "current_epoch": 1,
                "total_epochs": 2,
                "progress_percentage": 50,
                "created_at": "2026-05-26T10:00:00",
            }
        ],
        "total": 1,
    }
    with mock.patch("dagnam.list_training_jobs", mock.Mock(return_value=payload)):
        run_cli(["training", "list"])
        run_cli(["training", "list", "--verbose"])
    out = capsys.readouterr().out
    assert "j1" in out
    assert "running" in out
    assert '"items"' in out  # from the verbose run


def test_training_list_empty(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.list_training_jobs", mock.Mock(return_value={"items": [], "total": 0})):
        run_cli(["training", "list"])
    assert "No training jobs found" in capsys.readouterr().out


def test_training_list_json_writes_output(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    payload = {"items": [{"id": "j1"}], "total": 1}
    output = tmp_path / "training.json"
    with mock.patch("dagnam.list_training_jobs", mock.Mock(return_value=payload)):
        run_cli(["training", "list", "--json", "--output", str(output)])
    assert json.loads(capsys.readouterr().out) == payload
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_training_get(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = {
        "id": "j1",
        "status": "running",
        "framework": "pytorch",
        "current_epoch": 1,
        "total_epochs": 3,
        "progress_percentage": 33,
    }
    with mock.patch("dagnam.get_training_job", mock.Mock(return_value=payload)):
        run_cli(["training", "get", "j1"])
    out = capsys.readouterr().out
    assert "Training job j1" in out
    assert "Epoch: 1/3" in out
    assert '"id": "j1"' not in out


def test_training_get_json_prints_full_payload(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = {"id": "j1", "status": "running"}
    with mock.patch("dagnam.get_training_job", mock.Mock(return_value=payload)):
        run_cli(["training", "get", "j1", "--json"])
    assert json.loads(capsys.readouterr().out) == payload


def test_training_logs_dispatches_filters(run_cli: CliRunner, capsys: StrCapture) -> None:
    logs = mock.Mock(return_value={"items": []})
    with mock.patch("dagnam.training_logs", logs):
        run_cli(["training", "logs", "j1", "--log-level", "error", "--limit", "5"])
    logs.assert_called_once_with("j1", log_level="error", source=None, page=1, limit=5)
    assert json.loads(capsys.readouterr().out) == {"items": []}


def test_training_metrics_dispatches_filters(run_cli: CliRunner, capsys: StrCapture) -> None:
    metrics = mock.Mock(return_value={"items": []})
    with mock.patch("dagnam.training_metrics", metrics):
        run_cli(["training", "metrics", "j1", "--metric-type", "train_loss", "--epoch-summary"])
    metrics.assert_called_once_with(
        "j1",
        metric_type="train_loss",
        epoch_start=None,
        epoch_end=None,
        epoch_summary=True,
        page=1,
        limit=100,
    )
    assert json.loads(capsys.readouterr().out) == {"items": []}


def test_training_metrics_summary_dispatches(run_cli: CliRunner, capsys: StrCapture) -> None:
    summary = mock.Mock(return_value={"best_epoch": 2})
    with mock.patch("dagnam.training_metrics_summary", summary):
        run_cli(["training", "metrics-summary", "j1"])
    summary.assert_called_once_with("j1")
    assert json.loads(capsys.readouterr().out) == {"best_epoch": 2}


def test_training_cancel_prints_message(run_cli: CliRunner, capsys: StrCapture) -> None:
    cancel = mock.Mock(return_value={"message": "Training job cancelled successfully"})
    with mock.patch("dagnam.cancel_training_job", cancel):
        run_cli(["training", "cancel", "j1"])
    cancel.assert_called_once_with("j1")
    assert "cancelled successfully" in capsys.readouterr().out


def test_training_delete_bulk(run_cli: CliRunner, capsys: StrCapture) -> None:
    delete = mock.Mock(return_value={"deleted": 2})
    with mock.patch("dagnam.delete_training_jobs", delete):
        run_cli(["training", "delete", "j1", "j2"])
    delete.assert_called_once_with(["j1", "j2"])
    assert '"deleted": 2' in capsys.readouterr().out


@pytest.mark.parametrize(
    ("cmd_args", "attr"),
    [
        (["training", "get", "j1"], "get_training_job"),
        (["training", "cancel", "j1"], "cancel_training_job"),
        (["training", "delete", "j1"], "delete_training_jobs"),
        (["training", "list"], "list_training_jobs"),
    ],
)
def test_training_apierrors_exit(run_cli: CliRunner, cmd_args: list[str], attr: str) -> None:
    from dagnam._core.exceptions import APIError

    with mock.patch(f"dagnam.{attr}", mock.Mock(side_effect=APIError(500, "boom"))):
        with pytest.raises(SystemExit):
            run_cli(cmd_args)


# ---------------------------------------------------------------- training attach handler


def test_training_attach_invokes_runner_and_exits_with_code(run_cli: CliRunner) -> None:
    runner = mock.Mock(return_value=0)
    with mock.patch("dagnam.training_attach.run_training_attach", runner):
        with pytest.raises(SystemExit) as exc:
            run_cli(["training", "attach", "job-1", "--metrics-path", "m.jsonl", "--replay"])
    assert exc.value.code == 0
    kwargs = runner.call_args.kwargs
    assert kwargs["job_id"] == "job-1"
    assert kwargs["metrics_path"] == "m.jsonl"
    assert kwargs["replay"] is True


def test_training_attach_not_logged_in_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import AuthError

    with mock.patch(
        "dagnam.training_attach.run_training_attach", mock.Mock(side_effect=AuthError("nope"))
    ):
        with pytest.raises(SystemExit) as exc:
            run_cli(["training", "attach", "job-1"])
    assert exc.value.code == 1
    assert "Not logged in" in capsys.readouterr().err


def test_training_attach_file_not_found_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch(
        "dagnam.training_attach.run_training_attach",
        mock.Mock(side_effect=FileNotFoundError("no metrics file")),
    ):
        with pytest.raises(SystemExit) as exc:
            run_cli(["training", "attach", "job-1"])
    assert exc.value.code == 1
    assert "no metrics file" in capsys.readouterr().err


def test_training_attach_dagnam_error_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import APIError

    with mock.patch(
        "dagnam.training_attach.run_training_attach",
        mock.Mock(side_effect=APIError(500, "boom")),
    ):
        with pytest.raises(SystemExit) as exc:
            run_cli(["training", "attach", "job-1"])
    assert exc.value.code == 1
    assert "boom" in capsys.readouterr().err


# ---------------------------------------------------------------- training create config overrides


def test_training_create_rejects_non_object_config(run_cli: CliRunner, capsys: StrCapture) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli(
            [
                "training",
                "create",
                "p1",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--learning-rate",
                "0.1",
                "--optimizer",
                "adam",
                "--loss-function",
                "mse",
                "--dataset-id",
                "ds1",
                "--config",
                "[1, 2, 3]",
            ]
        )
    assert exc.value.code == 1
    assert "must be a JSON object" in capsys.readouterr().err


# ---------------------------------------------------------------- training logs/metrics/summary


def test_training_logs_writes_output_file(run_cli: CliRunner, tmp_path: Path) -> None:
    output = tmp_path / "logs.json"
    with mock.patch("dagnam.training_logs", mock.Mock(return_value={"items": []})):
        run_cli(["training", "logs", "j1", "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8")) == {"items": []}


def test_training_metrics_writes_output_file(run_cli: CliRunner, tmp_path: Path) -> None:
    output = tmp_path / "metrics.json"
    with mock.patch("dagnam.training_metrics", mock.Mock(return_value={"items": [1]})):
        run_cli(["training", "metrics", "j1", "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8")) == {"items": [1]}


def test_training_metrics_summary_writes_output_file(run_cli: CliRunner, tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    with mock.patch("dagnam.training_metrics_summary", mock.Mock(return_value={"best_epoch": 2})):
        run_cli(["training", "metrics-summary", "j1", "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8")) == {"best_epoch": 2}


@pytest.mark.parametrize(
    ("cmd_args", "attr"),
    [
        (["training", "logs", "j1"], "training_logs"),
        (["training", "metrics", "j1"], "training_metrics"),
        (["training", "metrics-summary", "j1"], "training_metrics_summary"),
        (
            [
                "training",
                "create",
                "p1",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--learning-rate",
                "0.1",
                "--optimizer",
                "adam",
                "--loss-function",
                "mse",
                "--dataset-id",
                "ds1",
            ],
            "create_training_job",
        ),
    ],
)
def test_training_more_apierrors_exit(run_cli: CliRunner, cmd_args: list[str], attr: str) -> None:
    from dagnam._core.exceptions import APIError

    with mock.patch(f"dagnam.{attr}", mock.Mock(side_effect=APIError(500, "boom"))):
        with pytest.raises(SystemExit):
            run_cli(cmd_args)


def test_training_cancel_default_message_when_absent(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    with mock.patch("dagnam.cancel_training_job", mock.Mock(return_value={})):
        run_cli(["training", "cancel", "j1"])
    assert "cancelled." in capsys.readouterr().out


def test_training_get_writes_output_file(run_cli: CliRunner, tmp_path: Path) -> None:
    output = tmp_path / "job.json"
    payload = {"id": "j1", "status": "running"}
    with mock.patch("dagnam.get_training_job", mock.Mock(return_value=payload)):
        run_cli(["training", "get", "j1", "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8")) == payload
