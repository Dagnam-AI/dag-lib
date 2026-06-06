from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.typing_helpers import PytestMonkeyPatch

from dagnam._core.exceptions import APIError, AuthError, TrainingJobNotFoundError
from dagnam._core.metrics_uploader import (
    HTTPSink,
    ListSink,
    UploadRetriesExhaustedError,
    drain_jsonl_to_sink,
    is_terminal_upload_error,
    run_upload_loop,
)


def test_drain_batches_and_assigns_event_ids(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"
    lines = [
        json.dumps({"type": "metric", "epoch": 0, "step": step, "metrics": {"loss": 1.0}})
        for step in range(3)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sink = ListSink()

    sent = drain_jsonl_to_sink(path=path, job_id="run_x", sink=sink, replay=True)

    assert sent == 3
    flat = [event for batch in sink.batches for event in batch]
    raw = path.read_bytes()
    offsets = [raw.index(line.encode("utf-8")) for line in lines]
    assert [event["event_id"] for event in flat] == [f"run_x:{offset}" for offset in offsets]


def test_list_sink_ignores_empty_batch():
    sink = ListSink()
    sink.send([])
    assert sink.batches == []


def test_http_sink_ignores_empty_batch():
    class RecordingClient:
        def __init__(self) -> None:
            self.calls = 0

        def upload_training_events(self, *args, **kwargs):
            self.calls += 1

    client = RecordingClient()
    sink = HTTPSink(client, "run_x")
    sink.send([])
    assert client.calls == 0


def test_http_sink_inner_send_ignores_empty_batch():
    class RecordingClient:
        def __init__(self) -> None:
            self.calls = 0

        def upload_training_events(self, *args, **kwargs):
            self.calls += 1

    client = RecordingClient()
    sink = HTTPSink(client, "run_x")
    sink._send([])  # exercises the empty false-leg of the inner sender
    assert client.calls == 0


def test_http_sink_send_uses_source_when_provided():
    events = [{"event_id": "run_x:0", "type": "heartbeat"}]

    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list = []

        def upload_training_events(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    client = RecordingClient()
    sink = HTTPSink(client, "run_x", source={"origin": "test"})
    sink.send(events)
    assert client.calls == [(("run_x", events), {"source": {"origin": "test"}})]


def test_http_sink_reraises_auth_error_without_refresh():
    events = [{"event_id": "run_x:0", "type": "heartbeat"}]

    class ExpiredClient:
        def upload_training_events(self, *args, **kwargs):
            raise AuthError("expired")

    sink = HTTPSink(ExpiredClient(), "run_x")
    with pytest.raises(AuthError):
        sink.send(events)


def test_terminal_for_auth_and_job_not_found():
    assert is_terminal_upload_error(AuthError("expired")) is True
    assert is_terminal_upload_error(TrainingJobNotFoundError("run_x")) is True
    assert is_terminal_upload_error(RuntimeError("nope")) is False


def test_http_sink_refreshes_expired_run_token_once():
    events = [{"event_id": "run_x:0", "type": "heartbeat"}]

    class ExpiredClient:
        def upload_training_events(self, *args, **kwargs):
            raise AuthError("expired")

    class RefreshedClient:
        def __init__(self):
            self.uploads = []

        def upload_training_events(self, *args, **kwargs):
            self.uploads.append((args, kwargs))

    refreshed = RefreshedClient()
    sink = HTTPSink(
        ExpiredClient(),
        "run_x",
        refresh_client=lambda: refreshed,
    )

    sink.send(events)

    assert refreshed.uploads == [(("run_x", events), {})]


def test_409_is_a_terminal_stop_not_a_retry():
    # A cancelled/finished run makes ingest return 409. That is the platform's
    # "stop streaming" signal, so it must be classified terminal (never retried).
    assert is_terminal_upload_error(APIError(409, "run cancelled")) is True
    # A transient 5xx / 429 must NOT be terminal — those keep retrying.
    assert is_terminal_upload_error(APIError(503, "unavailable")) is False
    assert is_terminal_upload_error(APIError(429, "slow down")) is False


def test_cancelled_run_stops_the_upload_loop_promptly(tmp_path: Path):
    # Regression for the cancel-stops-stream contract: a 409 from the sink must
    # propagate out of the loop (terminating the uploader thread) instead of
    # being swallowed into an unbounded retry.
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps({"type": "metric", "epoch": 0, "step": 0, "metrics": {"loss": 1.0}}) + "\n",
        encoding="utf-8",
    )

    class CancelledSink:
        def __init__(self) -> None:
            self.calls = 0

        def send(self, events):
            self.calls += 1
            raise APIError(409, "run cancelled")

    sink = CancelledSink()
    with pytest.raises(APIError) as excinfo:
        run_upload_loop(
            path=path,
            job_id="run_x",
            sink=sink,
            should_continue=lambda: True,
            replay_existing=True,
            poll_interval=0.0,
            retry_initial_interval=0.0,
        )
    assert excinfo.value.status_code == 409
    assert sink.calls == 1  # stopped on the first 409, no retry storm


def _write_metrics(path: Path, count: int) -> None:
    lines = [
        json.dumps({"type": "metric", "epoch": 0, "step": step, "metrics": {"loss": 1.0}})
        for step in range(count)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_transient_error_retries_with_backoff_then_succeeds(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
):
    path = tmp_path / "metrics.jsonl"
    _write_metrics(path, 1)

    sleeps: list[float] = []
    monkeypatch.setattr("dagnam._core.metrics_uploader.time.sleep", lambda s: sleeps.append(s))

    class FlakySink:
        def __init__(self) -> None:
            self.calls = 0

        def send(self, events):
            self.calls += 1
            if self.calls == 1:
                raise APIError(503, "unavailable")

    sink = FlakySink()
    sent = run_upload_loop(
        path=path,
        job_id="run_x",
        sink=sink,
        should_continue=lambda: False,
        replay=True,
        retry_initial_interval=0.5,
    )
    assert sent == 1
    assert sink.calls == 2
    # backoff slept once on the failed attempt
    assert sleeps == [0.5]


def test_cap_pending_drops_oldest_and_warns_once(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    path = tmp_path / "metrics.jsonl"
    _write_metrics(path, 5)
    monkeypatch.setattr("dagnam._core.metrics_uploader.time.sleep", lambda _s: None)

    sink = ListSink()
    sent = run_upload_loop(
        path=path,
        job_id="run_x",
        sink=sink,
        should_continue=lambda: False,
        replay=True,
        batch_size=100,  # avoid mid-drain flush so cap applies to backlog
        max_pending=2,
    )
    # 5 read, capped to 2 -> only 2 ultimately sent
    assert sent == 2
    err = capsys.readouterr().err
    assert "backlog exceeded 2 events" in err


def test_replay_unrecoverable_drain_raises_exhausted(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
):
    path = tmp_path / "metrics.jsonl"
    _write_metrics(path, 1)
    monkeypatch.setattr("dagnam._core.metrics_uploader.time.sleep", lambda _s: None)

    class AlwaysTransientSink:
        def send(self, events):
            raise APIError(503, "unavailable")

    with pytest.raises(UploadRetriesExhaustedError):
        run_upload_loop(
            path=path,
            job_id="run_x",
            sink=AlwaysTransientSink(),
            should_continue=lambda: False,
            replay=True,
            retry_initial_interval=0.0,
            final_upload_attempts=2,
        )


def test_drain_once_flushes_when_batch_size_reached(tmp_path: Path, monkeypatch: PytestMonkeyPatch):
    path = tmp_path / "metrics.jsonl"
    _write_metrics(path, 4)
    monkeypatch.setattr("dagnam._core.metrics_uploader.time.sleep", lambda _s: None)

    sink = ListSink()
    sent = run_upload_loop(
        path=path,
        job_id="run_x",
        sink=sink,
        should_continue=lambda: False,
        replay=True,
        batch_size=2,
    )
    assert sent == 4
    # mid-drain flushes produced multiple batches of size 2
    assert [len(batch) for batch in sink.batches] == [2, 2]


def test_polling_loop_runs_then_stops(tmp_path: Path, monkeypatch: PytestMonkeyPatch):
    path = tmp_path / "metrics.jsonl"
    _write_metrics(path, 1)
    sleeps: list[float] = []
    monkeypatch.setattr("dagnam._core.metrics_uploader.time.sleep", lambda s: sleeps.append(s))

    states = iter([True, False])
    sink = ListSink()
    sent = run_upload_loop(
        path=path,
        job_id="run_x",
        sink=sink,
        should_continue=lambda: next(states),
        replay_existing=True,
        poll_interval=0.25,
        retry_initial_interval=0.0,
    )
    assert sent == 1
    assert 0.25 in sleeps  # poll_interval slept inside the loop


def test_keyboard_interrupt_drains_then_reraises(tmp_path: Path, monkeypatch: PytestMonkeyPatch):
    path = tmp_path / "metrics.jsonl"
    _write_metrics(path, 1)
    monkeypatch.setattr("dagnam._core.metrics_uploader.time.sleep", lambda _s: None)

    sink = ListSink()

    def boom() -> bool:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_upload_loop(
            path=path,
            job_id="run_x",
            sink=sink,
            should_continue=boom,
            replay_existing=True,
            poll_interval=0.0,
            retry_initial_interval=0.0,
        )
    # final drain+flush ran before re-raising
    assert sink.batches
    assert sink.batches[0][0]["event_id"].startswith("run_x:")


def test_streaming_loop_final_flush_failure_raises_exhausted(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
):
    path = tmp_path / "metrics.jsonl"
    _write_metrics(path, 1)
    monkeypatch.setattr("dagnam._core.metrics_uploader.time.sleep", lambda _s: None)

    class AlwaysTransientSink:
        def send(self, events):
            raise APIError(503, "unavailable")

    with pytest.raises(UploadRetriesExhaustedError):
        run_upload_loop(
            path=path,
            job_id="run_x",
            sink=AlwaysTransientSink(),
            should_continue=lambda: False,
            replay_existing=True,
            poll_interval=0.0,
            retry_initial_interval=0.0,
            final_upload_attempts=1,
        )
