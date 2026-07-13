"""CLI coverage for the training lifecycle subcommands.

Covers ``dagnam training restart|restore|estimate|allowed-strategies`` - the
handler wiring plus the human render for estimates and the Yes/No strategy
table.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest import mock

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, StrCapture


# ----------------------------------------------------------------- restart


def test_training_restart_prints_and_suggests_stream(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    restart = mock.Mock(return_value={"id": "j2", "status": "pending"})
    with mock.patch("dagnam.restart", restart):
        run_cli(["training", "restart", "j1"])
    restart.assert_called_once_with("j1")
    captured = capsys.readouterr()
    assert '"id": "j2"' in captured.out
    assert "Next: dagnam stream j2" in captured.err


def test_training_restart_missing_id_uses_placeholder(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    with mock.patch("dagnam.restart", mock.Mock(return_value={})):
        run_cli(["training", "restart", "j1"])
    assert "Next: dagnam stream <job-id>" in capsys.readouterr().err


# ----------------------------------------------------------------- restore


def test_training_restore_prints_and_suggests_stream(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    restore = mock.Mock(return_value={"id": "j2"})
    with mock.patch("dagnam.restore_checkpoint", restore):
        run_cli(["training", "restore", "j1", "c1"])
    restore.assert_called_once_with("j1", "c1")
    captured = capsys.readouterr()
    assert '"id": "j2"' in captured.out
    assert "Next: dagnam stream j2" in captured.err


def test_training_restore_missing_id_uses_placeholder(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    with mock.patch("dagnam.restore_checkpoint", mock.Mock(return_value={})):
        run_cli(["training", "restore", "j1", "c1"])
    assert "Next: dagnam stream <job-id>" in capsys.readouterr().err


# ---------------------------------------------------------------- estimate

_ESTIMATE = {
    "estimated_memory_mb": 512,
    "estimated_training_time_seconds": 60,
    "estimated_disk_space_mb": 128,
    "estimated_cost_usd": 1.5,
    "warnings": ["low memory"],
    "recommendations": ["use a smaller batch"],
}


def test_training_estimate_passes_hyperparameters(run_cli: CliRunner, capsys: StrCapture) -> None:
    estimate = mock.Mock(return_value=_ESTIMATE)
    with mock.patch("dagnam.estimate_resources", estimate):
        run_cli(
            [
                "training",
                "estimate",
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
            ]
        )
    kwargs = estimate.call_args.kwargs
    assert kwargs["epochs"] == 2
    assert kwargs["batch_size"] == 32
    assert kwargs["training_dataset_id"] == "ds1"
    out = capsys.readouterr().out
    assert "Estimated memory:   512 MB" in out
    assert "Estimated cost:     1.5" in out
    assert "low memory" in out
    assert "use a smaller batch" in out


def test_training_estimate_json_prints_raw(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.estimate_resources", mock.Mock(return_value=_ESTIMATE)):
        run_cli(
            [
                "training",
                "estimate",
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
                "--json",
            ]
        )
    assert json.loads(capsys.readouterr().out) == _ESTIMATE


def test_training_estimate_empty_render_has_no_optional_sections(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    result = {"estimated_memory_mb": 256, "warnings": [], "recommendations": []}
    with mock.patch("dagnam.estimate_resources", mock.Mock(return_value=result)):
        run_cli(
            [
                "training",
                "estimate",
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
            ]
        )
    out = capsys.readouterr().out
    assert "Estimated memory:   256 MB" in out
    assert "Warnings:" not in out
    assert "Recommendations:" not in out


# ------------------------------------------------------------ allowed-strategies


def test_training_allowed_strategies_renders_yes_no(run_cli: CliRunner, capsys: StrCapture) -> None:
    strategies = mock.Mock(return_value={"cpu": True, "multi_gpu_ddp": False})
    with mock.patch("dagnam.allowed_strategies", strategies):
        run_cli(["training", "allowed-strategies"])
    out = capsys.readouterr().out
    assert "Strategy" in out
    assert "Available" in out
    lines = out.splitlines()
    cpu_line = next(line for line in lines if line.startswith("cpu"))
    ddp_line = next(line for line in lines if line.startswith("multi_gpu_ddp"))
    assert "Yes" in cpu_line
    assert "No" in ddp_line
    # Without a `required_tiers` map the Required-Tier cell falls back to "-".
    assert cpu_line.strip().endswith("-")
    assert ddp_line.strip().endswith("-")


def test_training_allowed_strategies_renders_required_tier_for_locked(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    body = {
        "cpu": True,
        "multi_gpu_ddp": False,
        "multi_node": False,
        # Registry-driven metadata shipped alongside the flat availability map.
        "required_tiers": {"multi_gpu_ddp": "pro", "multi_node": "enterprise"},
    }
    with mock.patch("dagnam.allowed_strategies", mock.Mock(return_value=body)):
        run_cli(["training", "allowed-strategies"])
    out = capsys.readouterr().out
    lines = out.splitlines()
    # `required_tiers` is metadata, not a strategy — it must never render as a row.
    assert not any(line.startswith("required_tiers") for line in lines)
    assert "Required Tier" in out
    cpu_line = next(line for line in lines if line.startswith("cpu"))
    ddp_line = next(line for line in lines if line.startswith("multi_gpu_ddp"))
    node_line = next(line for line in lines if line.startswith("multi_node"))
    # Each locked strategy shows the minimum tier that unlocks it, title-cased.
    assert ddp_line.strip().endswith("Pro")
    assert node_line.strip().endswith("Enterprise")
    # An available strategy needs no tier.
    assert cpu_line.strip().endswith("-")


def test_training_allowed_strategies_empty(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.allowed_strategies", mock.Mock(return_value={})):
        run_cli(["training", "allowed-strategies"])
    assert "No strategies available." in capsys.readouterr().out


def test_training_allowed_strategies_json_prints_raw(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    body = {"cpu": True, "single_gpu": False}
    with mock.patch("dagnam.allowed_strategies", mock.Mock(return_value=body)):
        run_cli(["training", "allowed-strategies", "--json"])
    assert json.loads(capsys.readouterr().out) == body
