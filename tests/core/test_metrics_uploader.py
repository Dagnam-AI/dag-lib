from __future__ import annotations

import json
from pathlib import Path

import pytest

from dagnam._core.exceptions import APIError, AuthError
from dagnam._core.metrics_uploader import (
    HTTPSink,
    ListSink,
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
    assert [event["event_id"] for event in flat] == [
        f"run_x:{offset}" for offset in offsets
    ]


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
