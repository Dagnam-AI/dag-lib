"""Delegation coverage for the training-lifecycle resource functions.

Covers ``restart``, ``restore_checkpoint``, ``estimate_resources``,
``allowed_strategies``, ``download_code``, and ``download_dag`` - each thin
wrapper's delegation to the resolved client - plus the shared
``_build_training_config`` helper that ``create_training_job`` and
``estimate_resources`` both use.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from dagnam._core.client import DagnamClient
from dagnam.resources.training import (
    _build_training_config,
    allowed_strategies,
    create_training_job,
    download_code,
    download_dag,
    estimate_resources,
    restart,
    restore_checkpoint,
)


def test_restart_delegates() -> None:
    c = MagicMock(spec=DagnamClient, restart_training_job=MagicMock(return_value={"id": "j2"}))
    assert restart("j1", client=c) == {"id": "j2"}
    c.restart_training_job.assert_called_once_with("j1")


def test_restore_checkpoint_delegates() -> None:
    c = MagicMock(spec=DagnamClient, restore_from_checkpoint=MagicMock(return_value={"id": "j2"}))
    assert restore_checkpoint("j1", "c1", client=c) == {"id": "j2"}
    c.restore_from_checkpoint.assert_called_once_with("j1", "c1")


def test_allowed_strategies_delegates() -> None:
    c = MagicMock(spec=DagnamClient, get_allowed_strategies=MagicMock(return_value={"cpu": True}))
    assert allowed_strategies(client=c) == {"cpu": True}
    c.get_allowed_strategies.assert_called_once_with()


def test_estimate_resources_builds_config_and_delegates() -> None:
    c = MagicMock(
        spec=DagnamClient,
        estimate_training_resources=MagicMock(return_value={"estimated_memory_mb": 512}),
    )
    out = estimate_resources(
        epochs=3,
        batch_size=16,
        learning_rate=0.01,
        optimizer="sgd",
        loss_function="mse",
        training_dataset_id="ds_1",
        validation_dataset_id="ds_val",
        config_overrides={"logging_config": {"log_frequency": 5}},
        client=c,
    )
    assert out == {"estimated_memory_mb": 512}
    config = c.estimate_training_resources.call_args.args[0]
    assert config["epochs"] == 3
    assert config["optimizer"] == "sgd"
    assert config["logging_config"] == {"log_frequency": 5}
    ds = config["dataset_config"]
    assert ds["training_dataset_id"] == "ds_1"
    assert ds["validation_dataset_id"] == "ds_val"
    assert "test_dataset_id" not in ds


def test_estimate_shares_build_config_with_create() -> None:
    """estimate_resources and create_training_job must produce the same inner config."""
    est_client = MagicMock(
        spec=DagnamClient, estimate_training_resources=MagicMock(return_value={})
    )
    create_client = MagicMock(
        spec=DagnamClient, create_training_job=MagicMock(return_value={"id": "j1"})
    )
    kwargs = {
        "epochs": 2,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "loss_function": "cross_entropy",
        "training_dataset_id": "ds_1",
    }
    estimate_resources(client=est_client, **kwargs)  # type: ignore[arg-type]
    create_training_job("proj_1", client=create_client, **kwargs)  # type: ignore[arg-type]

    est_config = est_client.estimate_training_resources.call_args.args[0]
    created_config = create_client.create_training_job.call_args.args[0]["config"]
    assert est_config == created_config
    # And both equal a direct call to the shared helper.
    assert est_config == _build_training_config(
        epochs=2,
        batch_size=32,
        learning_rate=1e-3,
        optimizer="adam",
        loss_function="cross_entropy",
        training_dataset_id="ds_1",
        validation_dataset_id=None,
        test_dataset_id=None,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        config_overrides=None,
    )


def test_download_code_defaults_to_cwd(tmp_path: Path) -> None:
    c = MagicMock(
        spec=DagnamClient,
        download_training_code=MagicMock(return_value=tmp_path / "j1-code.zip"),
    )
    out = download_code("j1", client=c)
    assert out == tmp_path / "j1-code.zip"
    c.download_training_code.assert_called_once_with("j1", ".")


def test_download_code_passes_out_dir(tmp_path: Path) -> None:
    c = MagicMock(
        spec=DagnamClient,
        download_training_code=MagicMock(return_value=tmp_path / "code.zip"),
    )
    download_code("j1", out=tmp_path, client=c)
    c.download_training_code.assert_called_once_with("j1", tmp_path)


def test_download_dag_defaults_to_cwd(tmp_path: Path) -> None:
    c = MagicMock(spec=DagnamClient, download_dag=MagicMock(return_value=tmp_path / "j1-dag.json"))
    out = download_dag("j1", client=c)
    assert out == tmp_path / "j1-dag.json"
    c.download_dag.assert_called_once_with("j1", ".")


def test_download_dag_passes_out_dir(tmp_path: Path) -> None:
    c = MagicMock(spec=DagnamClient, download_dag=MagicMock(return_value=tmp_path / "dag.json"))
    download_dag("j1", out=tmp_path, client=c)
    c.download_dag.assert_called_once_with("j1", tmp_path)
