"""Wire-level coverage for the sync training client mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
import requests

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError, AuthError, TrainingJobNotFoundError

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, RequestsMocker

API = "https://api.test"


# ---------------------------------------------------------------- _training_request


def test_create_training_job_posts_payload(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/training/jobs", json={"id": "j1"})
    assert client.create_training_job({"project_id": "p1"}) == {"id": "j1"}
    assert rmock.last_request.json() == {"project_id": "p1"}


def test_training_request_connectionerror(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "request", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.create_training_job({"project_id": "p1"})


def test_training_request_timeout(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "request", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.create_training_job({"project_id": "p1"})


def test_training_request_collection_404_is_generic_apierror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    # No job_id supplied -> 404 maps to APIError, not TrainingJobNotFoundError.
    rmock.post(f"{API}/api/v1/training/jobs", status_code=404, text="missing")
    with pytest.raises(APIError):
        client.create_training_job({"project_id": "p1"})


def test_training_request_job_404_maps_to_not_found(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.get(f"{API}/api/v1/training/jobs/j1", status_code=404)
    with pytest.raises(TrainingJobNotFoundError):
        client.get_training_job("j1")


def test_training_request_returns_none_on_empty_body(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    # 200 with no content -> _expect_object sees None and raises TypeError.
    rmock.post(f"{API}/api/v1/training/jobs", content=b"")
    with pytest.raises(TypeError):
        client.create_training_job({"project_id": "p1"})


def test_training_request_non_json_body_falls_back_to_text(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    # Non-JSON body -> response_json_value raises ValueError -> returns text,
    # which _expect_object then rejects (string, not object).
    rmock.post(
        f"{API}/api/v1/training/jobs",
        content=b"plain text",
        headers={"Content-Type": "text/plain"},
    )
    with pytest.raises(TypeError):
        client.create_training_job({"project_id": "p1"})


# ---------------------------------------------------------------- register_local_run


def test_register_local_run_without_max_duration(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/training/jobs", json={"id": "run-1"})
    result = client.register_local_run(project_id="p1", framework="pytorch", config={"epochs": 1})
    assert result == {"id": "run-1"}
    body = rmock.last_request.json()
    assert body["execution_mode"] == "local"
    assert "max_duration_seconds" not in body


def test_register_local_run_with_max_duration(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/training/jobs", json={"id": "run-1"})
    client.register_local_run(
        project_id="p1", framework="pytorch", config={}, max_duration_seconds=300
    )
    assert rmock.last_request.json()["max_duration_seconds"] == 300


# ---------------------------------------------------------------- other endpoints


def test_mint_run_token(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/training/jobs/j1/stream-token", json={"token": "t"})
    assert client.mint_run_token("j1") == {"token": "t"}


def test_mint_training_stream_token_posts_with_bearer_header(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/training/jobs/j1/stream-access-token", json={"token": "stream-t"})
    assert client.mint_training_stream_token("j1") == "stream-t"
    assert rmock.last_request.headers["Authorization"] == "Bearer k"


def test_mint_training_stream_token_401_maps_auth_error(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/training/jobs/j1/stream-access-token", status_code=401)
    with pytest.raises(AuthError):
        client.mint_training_stream_token("j1")


def test_mint_training_stream_token_404_maps_job_not_found(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/training/jobs/missing/stream-access-token", status_code=404)
    with pytest.raises(TrainingJobNotFoundError):
        client.mint_training_stream_token("missing")


def test_list_training_jobs_passes_filters(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/training/jobs", json={"items": []})
    client.list_training_jobs(status="running", page=2)
    assert rmock.last_request.qs == {"status": ["running"], "page": ["2"]}


def test_cancel_training_job(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/training/jobs/j1/cancel", json={"message": "ok"})
    assert client.cancel_training_job("j1") == {"message": "ok"}


def test_bulk_delete_training_jobs(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/training/jobs/bulk-delete", json={"deleted": 2})
    assert client.bulk_delete_training_jobs(["j1", "j2"]) == {"deleted": 2}
    assert rmock.last_request.json() == {"job_ids": ["j1", "j2"]}


def test_get_training_logs(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/training/jobs/j1/logs", json={"items": []})
    client.get_training_logs("j1", limit=5)
    assert rmock.last_request.qs == {"limit": ["5"]}


def test_get_training_metrics(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/training/jobs/j1/metrics", json={"items": []})
    client.get_training_metrics("j1", metric_type="train_loss")
    assert rmock.last_request.qs == {"metric_type": ["train_loss"]}


def test_get_training_metrics_summary(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/training/jobs/j1/metrics/summary", json={"best_epoch": 2})
    assert client.get_training_metrics_summary("j1") == {"best_epoch": 2}


# ---------------------------------------------------------------- upload_training_events


def test_upload_training_events_empty_short_circuits(client: DagnamClient) -> None:
    # No HTTP mock needed: empty list returns immediately.
    assert client.upload_training_events("j1", []) == {"accepted": 0, "duplicates": 0}


def test_upload_training_events_posts_default_source(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(
        f"{API}/api/v1/training/jobs/j1/metrics/events",
        json={"accepted": 1, "duplicates": 0},
    )
    result = client.upload_training_events("j1", [{"type": "metric"}])
    assert result == {"accepted": 1, "duplicates": 0}
    source = rmock.last_request.json()["source"]
    assert isinstance(source, dict)
    assert source["kind"] == "local_attach"


def test_upload_training_events_custom_source(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(
        f"{API}/api/v1/training/jobs/j1/metrics/events",
        json={"accepted": 1, "duplicates": 0},
    )
    client.upload_training_events("j1", [{"type": "metric"}], source={"kind": "local_stream"})
    assert rmock.last_request.json()["source"] == {"kind": "local_stream"}


def test_upload_training_events_unknown_sdk_version(
    client: DagnamClient, rmock: RequestsMocker, monkeypatch: PytestMonkeyPatch
) -> None:
    from importlib import metadata

    def _missing(_name: str) -> str:
        raise metadata.PackageNotFoundError("dagnam")

    monkeypatch.setattr(metadata, "version", _missing)
    rmock.post(
        f"{API}/api/v1/training/jobs/j1/metrics/events",
        json={"accepted": 1, "duplicates": 0},
    )
    client.upload_training_events("j1", [{"type": "metric"}])
    source = rmock.last_request.json()["source"]
    assert isinstance(source, dict)
    assert source["sdk_version"] == "0+unknown"


def test_upload_training_events_connectionerror(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.upload_training_events("j1", [{"type": "metric"}])


def test_upload_training_events_timeout(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.upload_training_events("j1", [{"type": "metric"}])


def test_upload_training_events_404_raises_job_not_found(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/training/jobs/j1/metrics/events", status_code=404)
    with pytest.raises(TrainingJobNotFoundError):
        client.upload_training_events("j1", [{"type": "metric"}])


# ---------------------------------------------------------------- open_training_stream


def test_open_training_stream_success(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/training/jobs/j1/stream-access-token", json={"token": "stream-t"})
    rmock.get(
        f"{API}/api/v1/streaming/training-jobs/j1/stream",
        text="data: hi\n\n",
        headers={"Content-Type": "text/event-stream"},
    )
    resp = client.open_training_stream("j1")
    assert resp.status_code == 200
    history = cast("Any", rmock).request_history
    assert history[0].method == "POST"
    assert history[1].method == "GET"
    assert rmock.last_request.qs == {"token": ["stream-t"]}
    assert "api_key" not in rmock.last_request.qs


def test_open_training_stream_uses_sse_read_timeout(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch
) -> None:
    # The SSE open must use a (connect, read) tuple with a read timeout above the
    # heartbeat interval, not the bare 30s used for ordinary requests — otherwise
    # a quiet or slow-to-start stream trips a spurious ReadTimeout.
    from unittest.mock import MagicMock

    from dagnam._core.client import base as base_mod, training as training_mod

    captured: dict[str, object] = {}

    def fake_get(_url: str, **kwargs: object) -> MagicMock:
        captured.update(kwargs)
        resp = MagicMock()
        resp.status_code = 200
        resp.ok = True
        return resp

    monkeypatch.setattr(client, "mint_training_stream_token", lambda _job_id: "stream-t")
    monkeypatch.setattr(training_mod.requests, "get", fake_get)
    client.open_training_stream("j1")
    assert captured["timeout"] == (base_mod.STREAM_CONNECT_TIMEOUT, base_mod.SSE_READ_TIMEOUT)


def test_open_training_stream_with_last_event_id(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/training/jobs/j1/stream-access-token", json={"token": "stream-t"})
    rmock.get(f"{API}/api/v1/streaming/training-jobs/j1/stream", text="ok")
    client.open_training_stream("j1", last_event_id="evt-9")
    assert rmock.last_request.headers["Last-Event-ID"] == "evt-9"


def test_open_training_stream_connectionerror(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(client, "mint_training_stream_token", lambda _job_id: "stream-t")
    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.open_training_stream("j1")


def test_open_training_stream_timeout(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(client, "mint_training_stream_token", lambda _job_id: "stream-t")
    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.open_training_stream("j1")


def test_open_training_stream_401(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/training/jobs/j1/stream-access-token", json={"token": "stream-t"})
    rmock.get(f"{API}/api/v1/streaming/training-jobs/j1/stream", status_code=401)
    with pytest.raises(AuthError):
        client.open_training_stream("j1")


def test_open_training_stream_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/training/jobs/j1/stream-access-token", json={"token": "stream-t"})
    rmock.get(f"{API}/api/v1/streaming/training-jobs/j1/stream", status_code=404)
    with pytest.raises(TrainingJobNotFoundError):
        client.open_training_stream("j1")


def test_open_training_stream_500(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/training/jobs/j1/stream-access-token", json={"token": "stream-t"})
    rmock.get(f"{API}/api/v1/streaming/training-jobs/j1/stream", status_code=500, text="boom")
    with pytest.raises(APIError):
        client.open_training_stream("j1")


def test_open_training_stream_scrubs_token_on_connection_error(
    monkeypatch: PytestMonkeyPatch,
) -> None:
    # The SSE token rides in params=, so the leak is via the requests exception
    # text (urllib3 embeds the composed ?token=… URL), not the local url var.
    client = DagnamClient(API, "key")
    monkeypatch.setattr(client, "mint_training_stream_token", lambda _job: "SECRET")

    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError(
            "HTTPSConnectionPool(host='api.test', port=443): Max retries exceeded "
            "with url: /api/v1/streaming/training-jobs/j1/stream?token=SECRET"
        )

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError) as ei:
        client.open_training_stream("j1")
    assert "SECRET" not in str(ei.value)
    assert "token=***" in str(ei.value)


def test_open_training_stream_scrubs_token_on_timeout(monkeypatch: PytestMonkeyPatch) -> None:
    client = DagnamClient(API, "key")
    monkeypatch.setattr(client, "mint_training_stream_token", lambda _job: "SECRET")

    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("timed out with url: /stream?token=SECRET")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError) as ei:
        client.open_training_stream("j1")
    assert "SECRET" not in str(ei.value)
