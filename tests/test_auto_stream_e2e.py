"""init() registers a run, mints a token, and uploads report events."""

from __future__ import annotations

import importlib
import time

import requests_mock as requests_mock_module

API = "https://api.test"


def test_init_streams_reported_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("DAGNAM_METRICS_PATH", str(tmp_path / "metrics.jsonl"))
    monkeypatch.delenv("DAGNAM_INTERNAL", raising=False)
    monkeypatch.setenv("DAGNAM_API_URL", API)
    monkeypatch.setenv("DAGNAM_API_KEY", "sk_test")

    import dagnam.training as training

    importlib.reload(training)
    training._reset()
    uploaded: list[dict] = []

    with requests_mock_module.Mocker() as mock:
        mock.post(
            f"{API}/api/v1/training/jobs",
            json={"id": "run_1", "execution_mode": "local", "status": "pending"},
            status_code=201,
        )
        mock.post(
            f"{API}/api/v1/training/jobs/run_1/stream-token",
            json={"token": "rt", "expires_in": 1800},
        )

        def record(request, context):
            uploaded.extend(request.json()["events"])
            context.status_code = 202
            return {"accepted": len(request.json()["events"]), "duplicates": 0}

        mock.post(f"{API}/api/v1/training/jobs/run_1/metrics/events", json=record)

        training.init(project_id="proj_1", framework="pytorch", name="run-fixed")
        training.report_metric(epoch=0, step=1, metrics={"loss": 1.0})
        training.report_metric(epoch=0, step=2, metrics={"loss": 0.9})
        time.sleep(1.5)
        training._finalize_stream()

    assert [event["type"] for event in uploaded][-1] == "complete"
    assert sum(event["type"] == "metric" for event in uploaded) == 2
    training._reset()
