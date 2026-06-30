"""Unit + security tests for dagnam.training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def reporter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Import dagnam.training with DAGNAM_METRICS_PATH pointed at a temp file."""
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("DAGNAM_METRICS_PATH", str(metrics_path))
    import dagnam.training as training

    training._reset()
    yield training, metrics_path
    training._close_file()


def _read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_report_metric_round_trip(reporter):
    training, path = reporter
    training.report_metric(epoch=1, step=50, metrics={"train_loss": 0.25, "accuracy": 0.9})
    events = _read_events(path)
    assert len(events) == 1
    assert events[0]["type"] == "metric"
    assert events[0]["epoch"] == 1
    assert events[0]["step"] == 50
    assert events[0]["metrics"] == {"train_loss": 0.25, "accuracy": 0.9}
    assert "timestamp" in events[0]


def test_timestamp_carries_explicit_utc_marker(reporter):
    training, path = reporter
    training.report_metric(epoch=1, step=1, metrics={"loss": 0.1})
    ts = _read_events(path)[0]["timestamp"]
    # The marker must be explicit UTC so consumers never read it as local time.
    assert ts.endswith("+00:00"), f"timestamp must be explicit UTC, got {ts!r}"

    from datetime import datetime as _dt

    parsed = _dt.fromisoformat(ts)
    assert parsed.tzinfo is not None
    offset = parsed.utcoffset()
    assert offset is not None
    assert offset.total_seconds() == 0


def test_report_progress_round_trip(reporter):
    training, path = reporter
    training.report_progress(epoch=2, total_epochs=10, step=3, total_steps=100)
    e = _read_events(path)[0]
    assert e["type"] == "progress"
    assert (e["epoch"], e["total_epochs"], e["step"], e["total_steps"]) == (2, 10, 3, 100)


def test_report_error_truncates(reporter):
    training, path = reporter
    training.report_error(category="user_code", technical_summary="x" * 9000, traceback="t" * 20000)
    e = _read_events(path)[0]
    assert e["type"] == "error"
    assert len(e["technical_summary"]) == 500
    assert len(e["traceback"]) == 8192


def test_falls_back_to_local_file_when_env_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DAGNAM_METRICS_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    import dagnam.training as training

    training._reset()
    training.report_metric(epoch=0, step=0, metrics={"loss": 1.0})
    training._close_file()
    assert (tmp_path / "dagnam_metrics.jsonl").is_file()


def test_uses_configured_training_metrics_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DAGNAM_METRICS_PATH", raising=False)
    config_dir = tmp_path / ".dagnam"
    config_dir.mkdir()
    configured_path = tmp_path / "configured" / "metrics.jsonl"
    (config_dir / "config.json").write_text(
        json.dumps({"training_metrics_path": str(configured_path)}),
        encoding="utf-8",
    )
    import dagnam._core.config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_FILE", config_dir / "config.json")
    import dagnam.training as training

    training._reset()
    training.report_metric(epoch=1, step=1, metrics={"loss": 0.1})
    training._close_file()
    assert configured_path.is_file()


def test_env_metrics_path_overrides_configured_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    configured_path = tmp_path / "configured" / "metrics.jsonl"
    env_path = tmp_path / "env" / "metrics.jsonl"
    config_dir = tmp_path / ".dagnam"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps({"training_metrics_path": str(configured_path)}),
        encoding="utf-8",
    )
    import dagnam._core.config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_FILE", config_dir / "config.json")
    monkeypatch.setenv("DAGNAM_METRICS_PATH", str(env_path))
    import dagnam.training as training

    training._reset()
    training.report_metric(epoch=1, step=1, metrics={"loss": 0.1})
    training._close_file()
    assert env_path.is_file()
    assert not configured_path.exists()


def test_fallback_warning_emitted_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.delenv("DAGNAM_METRICS_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    import dagnam._core.config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "missing.json")
    import dagnam.training as training

    training._reset()
    training.report_metric(epoch=1, step=1, metrics={"loss": 0.1})
    training.report_metric(epoch=1, step=2, metrics={"loss": 0.09})
    training._close_file()
    captured = capsys.readouterr()
    assert captured.err.count("training_metrics_path") == 1


def test_write_training_state_uses_utc_timestamp(
    reporter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    training, _ = reporter
    monkeypatch.setenv("DAGNAM_TRAINING_DIR", str(tmp_path))
    training.write_training_state(epoch=1, step=5, latest_checkpoint_path="/tmp/best.pth")
    state = json.loads((tmp_path / ".training_state.json").read_text(encoding="utf-8"))
    assert state["epoch"] == 1
    assert state["step"] == 5
    assert state["latest_checkpoint_path"] == "/tmp/best.pth"
    assert state["last_update_iso"].endswith("+00:00")


def test_never_raises_on_bad_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DAGNAM_METRICS_PATH", "/nonexistent_dir_xyz/cannot/write.jsonl")
    import dagnam.training as training

    training._reset()
    training.report_metric(epoch=0, step=0, metrics={"loss": 1.0})
    training._reset()


def test_module_is_stdlib_only_no_network():
    src = (Path(__file__).resolve().parents[1] / "dagnam" / "training.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "import requests",
        "from dagnam._core",
        "import urllib",
        "import socket",
        "import http",
    ):
        assert forbidden not in src, f"dagnam.training must not contain {forbidden!r}"


def test_metrics_path_resolution_is_lazy():
    """Importing dagnam.training must not read ~/.dagnam/config.json."""
    src = (Path(__file__).resolve().parents[1] / "dagnam" / "training.py").read_text(
        encoding="utf-8"
    )
    assert not src.rstrip().endswith(
        "_metrics_path, _using_fallback_path = _resolve_metrics_path()"
    )


def test_public_api_surface():
    import dagnam.training as training

    for name in (
        "report_metric",
        "report_progress",
        "report_system",
        "report_log",
        "report_error",
        "write_training_state",
    ):
        assert callable(getattr(training, name))


# ---------------------------------------------------------------- one-time diagnostics


def test_write_event_failure_warns_once(
    reporter, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    training, _ = reporter

    def _boom(*_a: object, **_k: object) -> str:
        raise OSError("disk full")

    monkeypatch.setattr(training.json, "dumps", _boom)
    training.report_metric(epoch=1, step=1, metrics={"loss": 0.1})
    training.report_metric(epoch=2, step=2, metrics={"loss": 0.2})  # must NOT re-warn

    err = capsys.readouterr().err
    assert err.count("failed to write a metrics event") == 1


def test_close_file_failure_warns_once(
    reporter, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    training, _ = reporter

    class _BadFile:
        closed = False

        def close(self) -> None:
            raise OSError("cannot close")

    monkeypatch.setattr(training, "_file", _BadFile())
    training._close_file()
    training._close_file()  # _file is now None -> no second warning

    err = capsys.readouterr().err
    assert err.count("failed to close the metrics file") == 1


def test_write_training_state_failure_warns_once(
    reporter, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    training, _ = reporter
    monkeypatch.setenv("DAGNAM_TRAINING_DIR", str(tmp_path))

    def _boom(*_a: object, **_k: object) -> str:
        raise OSError("disk full")

    monkeypatch.setattr(training.json, "dumps", _boom)
    training.write_training_state(epoch=1, step=1, latest_checkpoint_path=None)
    training.write_training_state(epoch=2, step=2, latest_checkpoint_path=None)

    err = capsys.readouterr().err
    assert err.count("failed to write training state") == 1


def test_warn_once_survives_stderr_failure(reporter, monkeypatch: pytest.MonkeyPatch):
    training, _ = reporter

    class _BadStderr:
        def write(self, _msg: str) -> int:
            raise OSError("no stderr")

    monkeypatch.setattr(training.sys, "stderr", _BadStderr())
    # The diagnostic itself must never raise, even if stderr is unwritable.
    training._warn_once("unique-key", "message\n")
    assert "unique-key" in training._warned_keys


def test_reset_clears_warned_keys(reporter):
    training, _ = reporter
    training._warn_once("some-key", "msg\n")
    assert "some-key" in training._warned_keys
    training._reset()
    assert training._warned_keys == set()
