"""CLI stream subcommand."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, StrCapture


# ---------------------------------------------------------------- stream


def test_stream_emits_human_readable(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake_event = SimpleNamespace(event="progress", data={"step": 1}, id="e1", retry=None)
    with mock.patch("dagnam.stream_training", return_value=iter([fake_event])):
        run_cli(["stream", "job-1"])
    assert "[progress]" in capsys.readouterr().out


def test_stream_json_mode(run_cli: CliRunner, capsys: StrCapture) -> None:
    # asdict requires a dataclass; use the real SSEEvent.
    from dagnam._core.sse import SSEEvent

    ev = SSEEvent(event="progress", data={"step": 1}, id="e1", retry=None)
    with mock.patch("dagnam.stream_training", return_value=iter([ev])):
        run_cli(["stream", "job-1", "--json"])
    assert json.loads(capsys.readouterr().out.strip()) == {
        "event": "progress",
        "data": {"step": 1},
        "id": "e1",
        "retry": None,
    }


def test_stream_keyboard_interrupt_exits_130(run_cli: CliRunner) -> None:
    with mock.patch("dagnam.stream_training", side_effect=KeyboardInterrupt):
        with pytest.raises(SystemExit) as exc_info:
            run_cli(["stream", "job-1"])
    assert exc_info.value.code == 130


def test_stream_apierror_exits(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import APIError

    with mock.patch("dagnam.stream_training", side_effect=APIError(500, "boom")):
        assert run_cli(["stream", "job-1"]) == 1
    err = capsys.readouterr().err
    assert "the Dagnam API had an internal error (HTTP 500)" in err
    assert "boom" in err
