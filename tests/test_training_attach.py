"""Tests for authenticated local training metrics attach/upload."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest import mock

import pytest
import requests_mock as rm_module

from dagnam._core.client import DagnamClient
from dagnam.cli import main as cli_main

API = "https://api.test"


@pytest.fixture
def run_cli(monkeypatch: pytest.MonkeyPatch):
    def _run(argv: list[str]) -> None:
        monkeypatch.setattr("sys.argv", ["dagnam", *argv])
        cli_main()

    return _run


def test_upload_training_events_posts_job_scoped_ingest_with_idempotency_key() -> None:
    client = DagnamClient(API, "sk_test")
    event = {
        "event_id": "job-1:0",
        "type": "metric",
        "timestamp": "2026-05-24T12:34:56.123456",
        "epoch": 1,
        "step": 2,
        "metrics": {"train_loss": 0.5},
    }

    with rm_module.Mocker() as rmock:
        rmock.post(
            f"{API}/api/v1/training/jobs/job-1/metrics/events",
            json={"accepted": 1, "duplicates": 0},
        )
        result = client.upload_training_events("job-1", [event])

    assert result == {"accepted": 1, "duplicates": 0}
    request = rmock.last_request
    assert request.headers["Authorization"] == "Bearer sk_test"
    assert "Idempotency-Key" not in request.headers
    assert request.json() == {
        "events": [event],
        "source": {"kind": "local_attach", "sdk_version": mock.ANY},
    }


def test_jsonl_tailer_holds_partial_lines_and_can_replay(tmp_path: Path) -> None:
    from dagnam.training_attach import MetricsJsonlTailer

    path = tmp_path / "metrics.jsonl"
    path.write_text('{"type":"metric","step":1}\n{"type":"metric"', encoding="utf-8")
    tailer = MetricsJsonlTailer(path, replay=True)

    assert list(tailer.read_available()) == [
        {"offset": 0, "event": {"type": "metric", "step": 1}},
    ]
    second_line_offset = path.read_bytes().index(b"\n") + 1

    with path.open("a", encoding="utf-8") as fh:
        fh.write(',"step":2}\n')

    assert list(tailer.read_available()) == [
        {"offset": second_line_offset, "event": {"type": "metric", "step": 2}},
    ]


def test_jsonl_tailer_accepts_utf8_bom_on_first_line(tmp_path: Path) -> None:
    from dagnam.training_attach import MetricsJsonlTailer

    path = tmp_path / "metrics.jsonl"
    path.write_bytes(b'\xef\xbb\xbf{"type":"metric","step":1}\n')
    tailer = MetricsJsonlTailer(path, replay=True)

    assert list(tailer.read_available()) == [
        {"offset": 0, "event": {"type": "metric", "step": 1}},
    ]


def test_jsonl_tailer_starts_at_end_without_replay(tmp_path: Path) -> None:
    from dagnam.training_attach import MetricsJsonlTailer

    path = tmp_path / "metrics.jsonl"
    path.write_text('{"type":"metric","step":1}\n', encoding="utf-8")
    tailer = MetricsJsonlTailer(path, replay=False)
    append_offset = path.stat().st_size

    assert list(tailer.read_available()) == []

    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"type":"metric","step":2}\n')

    assert list(tailer.read_available()) == [
        {"offset": append_offset, "event": {"type": "metric", "step": 2}},
    ]


def test_attach_sets_child_metrics_path_and_uploads_final_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dagnam.training_attach import run_training_attach

    metrics_path = tmp_path / "events.jsonl"
    captured_env: dict[str, str] = {}

    class FakeProcess:
        returncode = 7

        def __init__(self) -> None:
            self.poll_count = 0

        def poll(self) -> int | None:
            self.poll_count += 1
            if self.poll_count == 1:
                metrics_path.write_text(
                    json.dumps({"type": "metric", "epoch": 1, "step": 1, "metrics": {"loss": 0.1}})
                    + "\n",
                    encoding="utf-8",
                )
                return None
            return self.returncode

        def wait(self) -> int:
            return self.returncode

    def fake_popen(command: list[str], env: dict[str, str]) -> FakeProcess:
        captured_env.update(env)
        assert command == ["python", "train.py"]
        return FakeProcess()

    uploaded: list[list[dict[str, object]]] = []

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    client = SimpleNamespace(
        upload_training_events=lambda _job_id, events: (
            uploaded.append(events) or {"accepted": len(events)}
        )
    )
    code = run_training_attach(
        job_id="job-1",
        metrics_path=metrics_path,
        command=["python", "train.py"],
        client=client,
        poll_interval=0,
    )

    assert code == 7
    assert captured_env["DAGNAM_METRICS_PATH"] == str(metrics_path)
    assert uploaded[0][0]["event_id"] == "job-1:0"
    assert uploaded[0][0]["metrics"] == {"loss": 0.1}


def test_attach_replay_without_child_exits_after_existing_events(tmp_path: Path) -> None:
    from dagnam.training_attach import run_training_attach

    metrics_path = tmp_path / "events.jsonl"
    metrics_path.write_text(
        json.dumps({"type": "metric", "epoch": 1, "step": 1, "metrics": {"loss": 0.1}}) + "\n",
        encoding="utf-8",
    )
    uploaded: list[list[dict[str, object]]] = []
    client = SimpleNamespace(
        upload_training_events=lambda _job_id, events: (
            uploaded.append(events) or {"accepted": len(events)}
        )
    )

    code = run_training_attach(
        job_id="job-1",
        metrics_path=metrics_path,
        replay=True,
        client=client,
        poll_interval=0,
    )

    assert code == 0
    assert uploaded[0][0]["event_id"] == "job-1:0"


def test_attach_replay_without_child_fails_when_upload_never_recovers(tmp_path: Path) -> None:
    from dagnam.training_attach import run_training_attach

    metrics_path = tmp_path / "events.jsonl"
    metrics_path.write_text(
        json.dumps({"type": "metric", "step": 1}) + "\n",
        encoding="utf-8",
    )
    attempts = 0

    def upload(_job_id: str, _events: list[dict[str, object]]) -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("temporary outage")

    code = run_training_attach(
        job_id="job-1",
        metrics_path=metrics_path,
        replay=True,
        client=SimpleNamespace(upload_training_events=upload),
        retry_initial_interval=0,
        final_upload_attempts=3,
    )

    assert code == 1
    assert attempts == 3


def test_attach_retries_upload_errors_while_child_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dagnam.training_attach import run_training_attach

    metrics_path = tmp_path / "events.jsonl"

    class FakeProcess:
        def __init__(self) -> None:
            self.poll_count = 0

        def poll(self) -> int | None:
            self.poll_count += 1
            if self.poll_count == 1:
                metrics_path.write_text(
                    json.dumps({"type": "metric", "epoch": 1, "step": 1, "metrics": {"loss": 0.1}})
                    + "\n",
                    encoding="utf-8",
                )
                return None
            if self.poll_count == 2:
                return None
            return 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr("subprocess.Popen", lambda _command, env: FakeProcess())

    attempts = 0
    uploaded: list[list[dict[str, object]]] = []

    def upload(_job_id: str, events: list[dict[str, object]]) -> dict[str, int]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary upload failure")
        uploaded.append(events)
        return {"accepted": len(events), "duplicates": 0}

    client = SimpleNamespace(upload_training_events=upload)

    code = run_training_attach(
        job_id="job-1",
        metrics_path=metrics_path,
        command=["python", "train.py"],
        client=client,
        poll_interval=0,
        retry_initial_interval=0,
    )

    assert code == 0
    assert attempts == 2
    assert uploaded[0][0]["event_id"] == "job-1:0"


def test_attach_drains_metrics_on_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dagnam.training_attach import run_training_attach

    metrics_path = tmp_path / "events.jsonl"

    class FakeProcess:
        terminated = False

        def poll(self) -> int | None:
            metrics_path.write_text(
                json.dumps({"type": "metric", "epoch": 1, "step": 1, "metrics": {"loss": 0.1}})
                + "\n",
                encoding="utf-8",
            )
            raise KeyboardInterrupt

        def terminate(self) -> None:
            self.terminated = True

    fake_process = FakeProcess()
    monkeypatch.setattr("subprocess.Popen", lambda _command, env: fake_process)
    uploaded: list[list[dict[str, object]]] = []
    client = SimpleNamespace(
        upload_training_events=lambda _job_id, events: (
            uploaded.append(events) or {"accepted": len(events)}
        )
    )

    code = run_training_attach(
        job_id="job-1",
        metrics_path=metrics_path,
        command=["python", "train.py"],
        client=client,
        poll_interval=0,
        retry_initial_interval=0,
    )

    assert code == 130
    assert fake_process.terminated
    assert uploaded[0][0]["event_id"] == "job-1:0"


def test_attach_stops_fast_on_terminal_upload_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dagnam._core.exceptions import AuthError
    from dagnam.training_attach import run_training_attach

    metrics_path = tmp_path / "events.jsonl"

    class FakeProcess:
        terminated = False

        def __init__(self) -> None:
            self.poll_count = 0

        def poll(self) -> int | None:
            self.poll_count += 1
            if self.poll_count == 1:
                metrics_path.write_text(
                    json.dumps({"type": "metric", "epoch": 1, "step": 1, "metrics": {"loss": 0.1}})
                    + "\n",
                    encoding="utf-8",
                )
            # Always "running": without a terminal-error exit this would loop forever.
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self) -> int:
            return 0

    fake_process = FakeProcess()
    monkeypatch.setattr("subprocess.Popen", lambda _command, env: fake_process)

    attempts = 0

    def upload(_job_id: str, _events: list[dict[str, object]]) -> dict[str, int]:
        nonlocal attempts
        attempts += 1
        raise AuthError("invalid or expired API key")

    client = SimpleNamespace(upload_training_events=upload)

    with pytest.raises(AuthError):
        run_training_attach(
            job_id="job-1",
            metrics_path=metrics_path,
            command=["python", "train.py"],
            client=client,
            poll_interval=0,
            retry_initial_interval=0,
        )

    assert attempts == 1, "terminal auth error must not be retried"
    assert fake_process.terminated, "child must be terminated when attach exits on a terminal error"


def test_attach_kills_child_when_terminate_does_not_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dagnam._core.exceptions import AuthError
    from dagnam.training_attach import run_training_attach

    metrics_path = tmp_path / "events.jsonl"

    class FakeProcess:
        terminated = False
        killed = False

        def poll(self) -> None:
            metrics_path.write_text(json.dumps({"type": "metric", "step": 1}) + "\n")
            return None

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float | None = None) -> int:
            if timeout is not None and not self.killed:
                raise subprocess.TimeoutExpired(cmd="train", timeout=timeout)
            return 0

    process = FakeProcess()
    monkeypatch.setattr("subprocess.Popen", lambda _command, env: process)
    client = SimpleNamespace(
        upload_training_events=lambda *_args: (_ for _ in ()).throw(AuthError("bad key"))
    )

    with pytest.raises(AuthError):
        run_training_attach(
            job_id="job-1",
            metrics_path=metrics_path,
            command=["python", "train.py"],
            client=client,
            poll_interval=0,
        )

    assert process.terminated
    assert process.killed


def test_attach_kills_windows_child_process_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dagnam._core.exceptions import AuthError
    from dagnam.training_attach import run_training_attach

    metrics_path = tmp_path / "events.jsonl"

    class FakeProcess:
        pid = 1234

        def poll(self) -> None:
            metrics_path.write_text(json.dumps({"type": "metric", "step": 1}) + "\n")
            return None

        def terminate(self) -> None:
            raise AssertionError("Windows cleanup must terminate the owned process tree")

        def wait(self, timeout: float | None = None) -> int:
            return 0

    taskkill = mock.Mock()
    monkeypatch.setattr("dagnam.training_attach.os.name", "nt")
    monkeypatch.setattr("dagnam.training_attach.subprocess.run", taskkill)
    monkeypatch.setattr("subprocess.Popen", lambda _command, env: FakeProcess())
    client = SimpleNamespace(
        upload_training_events=lambda *_args: (_ for _ in ()).throw(AuthError("bad key"))
    )

    with pytest.raises(AuthError):
        run_training_attach(
            job_id="job-1",
            metrics_path=metrics_path,
            command=["python", "train.py"],
            client=client,
            poll_interval=0,
        )

    taskkill.assert_called_once_with(
        ["taskkill", "/F", "/T", "/PID", "1234"],
        capture_output=True,
        check=False,
    )


def test_attach_bounds_backlog_and_drops_oldest_during_outage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dagnam.training_attach import run_training_attach

    metrics_path = tmp_path / "events.jsonl"

    class FakeProcess:
        def __init__(self) -> None:
            self.poll_count = 0

        def poll(self) -> int | None:
            self.poll_count += 1
            if self.poll_count == 1:
                lines = "".join(
                    json.dumps({"type": "metric", "epoch": 1, "step": i, "metrics": {"loss": 0.1}})
                    + "\n"
                    for i in range(5)
                )
                metrics_path.write_text(lines, encoding="utf-8")
                return None
            return 0

        def terminate(self) -> None:
            pass

        def wait(self) -> int:
            return 0

    monkeypatch.setattr("subprocess.Popen", lambda _command, env: FakeProcess())

    attempts = 0
    uploaded: list[list[dict[str, object]]] = []

    def upload(_job_id: str, events: list[dict[str, object]]) -> dict[str, int]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary outage")  # transient: backlog accumulates, then retried
        uploaded.append(events)
        return {"accepted": len(events), "duplicates": 0}

    client = SimpleNamespace(upload_training_events=upload)

    code = run_training_attach(
        job_id="job-1",
        metrics_path=metrics_path,
        command=["python", "train.py"],
        client=client,
        poll_interval=0,
        retry_initial_interval=0,
        batch_size=100,
        max_pending=2,
    )

    assert code == 0
    # Five events were buffered while the first upload failed, but max_pending=2
    # kept only the two newest (steps 3 and 4); the oldest three were dropped.
    assert [e["step"] for e in uploaded[0]] == [3, 4]


def test_attach_cli_parses_command_after_separator(
    run_cli, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "sk_test")
    with mock.patch("dagnam.cli.training.cmd_training_attach") as attach:
        run_cli(
            [
                "training",
                "attach",
                "job-1",
                "--metrics-path",
                "events.jsonl",
                "--",
                "python",
                "train.py",
            ]
        )

    args = attach.call_args.args[0]
    assert args.job_id == "job-1"
    assert args.metrics_path == "events.jsonl"
    assert args.command == ["python", "train.py"]
