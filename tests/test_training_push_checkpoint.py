"""``dagnam.training.push_checkpoint``: the call generated training code makes after each save."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import requests_mock as rm_module

from dagnam._core import checkpoint_push

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, StrCapture

API = "https://api.test"
PUSH = f"{API}/api/v1/training/jobs/job-1/checkpoints/push"


@pytest.fixture
def reporter(tmp_path: Path, monkeypatch: PytestMonkeyPatch):
    """``dagnam.training`` on a platform run, writing its metrics to a temp file."""
    metrics = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("DAGNAM_METRICS_PATH", str(metrics))
    monkeypatch.delenv("DAGNAM_INTERNAL", raising=False)
    monkeypatch.setenv("DAGNAM_JOB_ID", "job-1")
    monkeypatch.setenv("DAGNAM_API_KEY", "run-token")
    monkeypatch.setenv("DAGNAM_API_URL", API)
    monkeypatch.setattr(checkpoint_push, "_stopped", False)
    import dagnam.training as training

    training._reset()
    yield training, metrics
    training._close_file()


def _events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_a_pushed_checkpoint_returns_its_id(reporter, tmp_path: Path) -> None:
    training, _ = reporter
    checkpoint = tmp_path / "checkpoint_epoch_1.pth"
    checkpoint.write_bytes(b"w")
    with rm_module.Mocker() as m:
        m.post(PUSH, status_code=201, json={"checkpoint_id": "ck-1"})
        assert training.push_checkpoint(checkpoint, epoch=1, step=10) == "ck-1"


def test_a_refused_push_lands_in_the_runs_log_and_the_run_goes_on(reporter, tmp_path: Path) -> None:
    training, metrics = reporter
    checkpoint = tmp_path / "checkpoint_epoch_1.pth"
    checkpoint.write_bytes(b"w")
    with rm_module.Mocker() as m:
        m.post(PUSH, status_code=429, json={"detail": "too soon"})
        assert training.push_checkpoint(str(checkpoint), epoch=1, step=10) is None

    [event] = _events(metrics)
    assert event["type"] == "log"
    assert event["level"] == "WARNING"
    assert "HTTP 429" in str(event["message"])


def test_nothing_the_push_does_can_raise_into_the_training_loop(
    reporter, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    training, _ = reporter

    def explode(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("the logger itself is broken")

    monkeypatch.setattr(checkpoint_push, "push_checkpoint", explode)

    assert training.push_checkpoint("ck.pth", epoch=1, step=1) is None
    assert training.push_checkpoint("ck.pth", epoch=2, step=2) is None
    assert capsys.readouterr().err.count("could not push a checkpoint") == 1
