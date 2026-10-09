"""``dagnam.cli.training_parser``: every ``stream`` / ``training`` command parses to its handler."""

from __future__ import annotations

import argparse

import pytest

from dagnam.cli import training
from dagnam.cli.training_parser import register_training


def _parse(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="dagnam")
    register_training(parser.add_subparsers(dest="command", required=True))
    return parser.parse_args(argv)


@pytest.mark.parametrize(
    ("argv", "handler"),
    [
        (("stream", "j1"), training.cmd_stream),
        (("training", "get", "j1"), training.cmd_training_get),
        (("training", "list"), training.cmd_training_list),
        (("training", "cancel", "j1"), training.cmd_training_cancel),
        (("training", "restart", "j1"), training.cmd_training_restart),
        (("training", "resume", "j1"), training.cmd_training_resume),
        (("training", "restore", "j1", "c1"), training.cmd_training_restore),
        (("training", "allowed-strategies"), training.cmd_training_allowed_strategies),
        (("training", "download-code", "j1"), training.cmd_training_download_code),
        (("training", "dag", "j1"), training.cmd_training_dag),
        (("training", "delete", "j1", "j2"), training.cmd_training_delete),
        (("training", "logs", "j1"), training.cmd_training_logs),
        (("training", "metrics", "j1"), training.cmd_training_metrics),
        (("training", "metrics-summary", "j1"), training.cmd_training_metrics_summary),
        (("training", "attach", "j1", "--", "python", "t.py"), training.cmd_training_attach),
    ],
)
def test_each_command_parses_to_its_handler(argv: tuple[str, ...], handler: object) -> None:
    assert _parse(*argv).func is handler


def test_create_and_estimate_take_the_hyperparameters() -> None:
    flags = [
        *("--epochs", "3", "--batch-size", "8", "--learning-rate", "0.01"),
        *("--optimizer", "adam", "--loss-function", "cross_entropy", "--dataset-id", "d1"),
    ]
    created = _parse("training", "create", "p1", *flags)
    assert created.func is training.cmd_training_create
    assert (created.project_id, created.epochs, created.framework) == ("p1", 3, "pytorch")
    estimate = _parse("training", "estimate", *flags)
    assert estimate.func is training.cmd_training_estimate
    assert estimate.train_split == pytest.approx(0.8)


def test_resume_takes_exactly_one_job_id() -> None:
    assert _parse("training", "resume", "j9").job_id == "j9"
    with pytest.raises(SystemExit):
        _parse("training", "resume")
    with pytest.raises(SystemExit):
        _parse("training", "resume", "a", "b")


def test_stream_options() -> None:
    args = _parse("stream", "j1", "--heartbeats", "--json")
    assert (args.heartbeats, args.json) == (True, True)
