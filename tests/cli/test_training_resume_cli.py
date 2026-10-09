"""CLI coverage for ``dagnam training resume`` and how a paused job is shown."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest import mock

import pytest

from dagnam._core.exceptions import (
    InsufficientCreditsError,
    TrainingJobNotFoundError,
    TrainingStateError,
)
from dagnam._core.sse import SSEEvent

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, StrCapture

PAUSE = 'Your training run "mnist" is paused at epoch 3 because there weren\'t enough credits.'


def test_training_resume_prints_the_job_and_suggests_streaming(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    resume = mock.Mock(return_value={"id": "j1", "status": "pending"})
    with mock.patch("dagnam.resume", resume):
        assert run_cli(["training", "resume", "j1"]) == 0
    resume.assert_called_once_with("j1")
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"id": "j1", "status": "pending"}
    assert "Next: dagnam stream j1" in captured.err


def test_training_resume_without_an_id_in_the_answer_uses_a_placeholder(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    with mock.patch("dagnam.resume", mock.Mock(return_value={})):
        run_cli(["training", "resume", "j1"])
    assert "Next: dagnam stream <job-id>" in capsys.readouterr().err


def test_training_resume_insufficient_credits_exits_one_with_the_platforms_sentence(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    exc = InsufficientCreditsError(
        "Resuming needs 120 credits to continue and your balance is 30.",
        required_credits=120,
        available_credits=30,
        next_steps=("request_credits",),
    )
    with mock.patch("dagnam.resume", side_effect=exc):
        assert run_cli(["training", "resume", "j1"]) == 1
    err = capsys.readouterr().err
    assert "Error: not enough credits" in err
    assert "Resuming needs 120 credits to continue and your balance is 30." in err
    assert "Ask for more credits at https://dagnam.ai/support" in err


def test_training_resume_unknown_job_exits_one(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.resume", side_effect=TrainingJobNotFoundError("j1")):
        assert run_cli(["training", "resume", "j1"]) == 1
    err = capsys.readouterr().err
    assert "Training job 'j1' not found" in err
    assert "dagnam training list" in err


@pytest.mark.parametrize(
    ("reason", "title", "hint"),
    [
        ("not_paused", "the training job is not paused", "dagnam training get <id>"),
        (
            "checkpoint_unavailable",
            "the paused job's saved checkpoint cannot be read",
            "dagnam checkpoint list <id>",
        ),
        (
            "not_accepting_checkpoints",
            "the training job is not accepting checkpoints",
            "dagnam training get <id>",
        ),
        (None, "the training job is not ready for that yet", "Wait a few minutes, then retry."),
    ],
)
def test_training_resume_conflicts_say_what_to_do(
    run_cli: CliRunner, capsys: StrCapture, reason: str | None, title: str, hint: str
) -> None:
    with mock.patch(
        "dagnam.resume", side_effect=TrainingStateError("The platform's sentence.", reason=reason)
    ):
        assert run_cli(["training", "resume", "j1"]) == 1
    err = capsys.readouterr().err
    assert f"Error: {title}" in err
    assert "The platform's sentence." in err
    assert hint in err


def test_training_resume_unknown_marker_is_treated_as_not_ready(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    with mock.patch(
        "dagnam.resume", side_effect=TrainingStateError("Hmm.", reason="something_new")
    ):
        assert run_cli(["training", "resume", "j1"]) == 1
    assert "Error: the training job is not ready for that yet" in capsys.readouterr().err


# ------------------------------------------------------------ the paused status


def test_training_get_shows_the_pause_reason_and_how_to_resume(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    payload = {"id": "j1", "status": "paused", "error_message": PAUSE, "current_epoch": 3}
    with mock.patch("dagnam.get_training_job", mock.Mock(return_value=payload)):
        run_cli(["training", "get", "j1"])
    captured = capsys.readouterr()
    assert "Status: paused" in captured.out
    assert f"Paused: {PAUSE}" in captured.out
    assert "Next: dagnam training resume j1" in captured.err


def test_training_get_strips_terminal_escapes_from_the_pause_reason(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    payload = {"id": "j1", "status": "paused", "error_message": "run \x1b[31mred\x1b[0m"}
    with mock.patch("dagnam.get_training_job", mock.Mock(return_value=payload)):
        run_cli(["training", "get", "j1"])
    assert "\x1b" not in capsys.readouterr().out


def test_training_get_paused_without_a_reason_still_points_to_resume(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    payload = {"id": "j1", "status": "paused", "error_message": None}
    with mock.patch("dagnam.get_training_job", mock.Mock(return_value=payload)):
        run_cli(["training", "get", "j1"])
    captured = capsys.readouterr()
    assert "Paused:" not in captured.out
    assert "Next: dagnam training resume j1" in captured.err


def test_training_get_of_a_running_job_suggests_nothing(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    with mock.patch(
        "dagnam.get_training_job", mock.Mock(return_value={"id": "j1", "status": "running"})
    ):
        run_cli(["training", "get", "j1"])
    captured = capsys.readouterr()
    assert "Paused:" not in captured.out
    assert "Next:" not in captured.err


def test_stream_that_ends_on_a_pause_suggests_resuming(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    events = [
        SSEEvent(event="paused", data={"message": PAUSE}),
        SSEEvent(event="stream_end", data={}),
    ]
    with mock.patch("dagnam.stream_training", return_value=iter(events)):
        assert run_cli(["stream", "job-1"]) == 0
    captured = capsys.readouterr()
    assert "[paused]" in captured.out
    assert "Next: dagnam training resume job-1" in captured.err


def test_stream_without_a_pause_suggests_nothing(run_cli: CliRunner, capsys: StrCapture) -> None:
    events = [SSEEvent(event="complete", data={})]
    with mock.patch("dagnam.stream_training", return_value=iter(events)):
        run_cli(["stream", "job-1"])
    assert "Next:" not in capsys.readouterr().err
