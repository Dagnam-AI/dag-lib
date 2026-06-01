"""Unit tests for dagnam.training (SSE event streaming)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import StreamError
from dagnam.resources.training import (
    cancel_training_job,
    create_training_job,
    delete_training_jobs,
    get_training_job,
    list_training_jobs,
    parse_event,
    stream_training,
    training_logs,
    training_metrics,
    training_metrics_summary,
)

if TYPE_CHECKING:
    from tests.typing_helpers import JsonObject


def _sse(event: str, data: str, id: str | None = None, retry: str | None = None) -> None:
    return SimpleNamespace(event=event, data=data, id=id, retry=retry)


class TestParseEvent:
    def test_decodes_json_payload(self) -> None:
        ev = parse_event(_sse("metric", '{"loss": 0.5}', id="1"))
        assert ev.event == "metric"
        assert ev.data == {"loss": 0.5}
        assert ev.id == "1"

    def test_non_json_falls_back_to_string(self) -> None:
        ev = parse_event(_sse("log", "plain text"))
        assert ev.data == "plain text"

    def test_empty_data_becomes_empty_dict(self) -> None:
        ev = parse_event(_sse("heartbeat", ""))
        assert ev.data == {}


class _FakeSSE:
    def __init__(self, events: list[JsonObject]) -> None:
        self._events = events

    def events(self):
        for e in self._events:
            yield e


class TestStreamTraining:
    def test_yields_events_until_terminal(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.open_training_stream.return_value = MagicMock()
        events = [
            _sse("metric", '{"epoch": 1, "loss": 0.5}', id="1"),
            _sse("metric", '{"epoch": 2, "loss": 0.3}', id="2"),
            _sse("complete", '{"status": "done"}', id="3"),
            _sse("metric", '{"never": "seen"}', id="4"),
        ]
        with patch("sseclient.SSEClient", return_value=_FakeSSE(events)):
            out = list(stream_training("job_1", client=client))
        assert [e.event for e in out] == ["metric", "metric", "complete"]
        assert out[0].data == {"epoch": 1, "loss": 0.5}

    def test_skips_heartbeats_by_default(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.open_training_stream.return_value = MagicMock()
        events = [
            _sse("heartbeat", "{}", id="1"),
            _sse("metric", '{"loss": 0.1}', id="2"),
            _sse("stream_end", "{}", id="3"),
        ]
        with patch("sseclient.SSEClient", return_value=_FakeSSE(events)):
            out = list(stream_training("job_1", client=client))
        assert [e.event for e in out] == ["metric", "stream_end"]

    def test_include_heartbeats(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.open_training_stream.return_value = MagicMock()
        events = [
            _sse("heartbeat", "{}", id="1"),
            _sse("complete", "{}", id="2"),
        ]
        with patch("sseclient.SSEClient", return_value=_FakeSSE(events)):
            out = list(stream_training("job_1", client=client, include_heartbeats=True))
        assert [e.event for e in out] == ["heartbeat", "complete"]

    def test_reconnects_with_last_event_id(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.open_training_stream.return_value = MagicMock()

        # First batch drops without a terminal; second batch finishes.
        batches = [
            [_sse("metric", '{"loss": 0.9}', id="e1")],
            [
                _sse("metric", '{"loss": 0.5}', id="e2"),
                _sse("complete", "{}", id="e3"),
            ],
        ]
        fake_iter = iter(batches)

        def fake_sse_client(response: object):
            return _FakeSSE(next(fake_iter))

        with (
            patch("sseclient.SSEClient", side_effect=fake_sse_client),
            patch("dagnam.resources.training.time.sleep"),
        ):
            out = list(stream_training("job_1", client=client))

        # Reconnect call should include last_event_id=e1
        assert client.open_training_stream.call_count == 2
        first_call = client.open_training_stream.call_args_list[0]
        second_call = client.open_training_stream.call_args_list[1]
        assert first_call.kwargs.get("last_event_id") is None
        assert second_call.kwargs.get("last_event_id") == "e1"
        assert [e.event for e in out] == ["metric", "metric", "complete"]

    def test_reconnect_exhaustion_raises(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.open_training_stream.return_value = MagicMock()
        with (
            patch("sseclient.SSEClient", return_value=_FakeSSE([])),
            patch("dagnam.resources.training.time.sleep"),
            pytest.raises(StreamError),
        ):
            list(stream_training("job_1", client=client, max_reconnects=2))


class TestCreateTrainingJob:
    def test_builds_nested_config_payload(self) -> None:
        c = MagicMock(spec=DagnamClient, create_training_job=MagicMock(return_value={"id": "j1"}))
        out = create_training_job(
            "proj_1",
            epochs=2,
            batch_size=32,
            learning_rate=1e-3,
            optimizer="adam",
            loss_function="cross_entropy",
            training_dataset_id="ds_1",
            client=c,
        )
        payload = c.create_training_job.call_args.args[0]
        assert out == {"id": "j1"}
        assert payload["project_id"] == "proj_1"
        assert payload["framework"] == "pytorch"
        assert payload["confirm_resource_warning"] is False
        assert "max_duration_seconds" not in payload
        config = payload["config"]
        assert config["epochs"] == 2
        assert config["optimizer"] == "adam"
        ds = config["dataset_config"]
        assert ds["training_dataset_id"] == "ds_1"
        assert ds["train_split"] == 0.8
        assert "validation_dataset_id" not in ds

    def test_optional_datasets_duration_and_overrides(self) -> None:
        c = MagicMock(spec=DagnamClient, create_training_job=MagicMock(return_value={"id": "j1"}))
        create_training_job(
            "proj_1",
            framework="flax",
            epochs=1,
            batch_size=8,
            learning_rate=0.01,
            optimizer="sgd",
            loss_function="mse",
            training_dataset_id="ds_1",
            validation_dataset_id="ds_val",
            test_dataset_id="ds_test",
            train_split=0.7,
            val_split=0.2,
            test_split=0.1,
            config_overrides={"logging_config": {"log_frequency": 5}},
            max_duration_seconds=600,
            confirm_resource_warning=True,
            client=c,
        )
        payload = c.create_training_job.call_args.args[0]
        assert payload["framework"] == "flax"
        assert payload["max_duration_seconds"] == 600
        assert payload["confirm_resource_warning"] is True
        config = payload["config"]
        assert config["logging_config"] == {"log_frequency": 5}
        ds = config["dataset_config"]
        assert ds["validation_dataset_id"] == "ds_val"
        assert ds["test_dataset_id"] == "ds_test"
        assert (ds["train_split"], ds["val_split"], ds["test_split"]) == (0.7, 0.2, 0.1)


class TestJobReadAndLifecycle:
    def test_get_delegates(self) -> None:
        c = MagicMock(spec=DagnamClient, get_training_job=MagicMock(return_value={"id": "j1"}))
        assert get_training_job("j1", client=c) == {"id": "j1"}
        c.get_training_job.assert_called_once_with("j1")

    def test_list_passes_filters_and_joins_status(self) -> None:
        c = MagicMock(spec=DagnamClient, list_training_jobs=MagicMock(return_value={"items": []}))
        list_training_jobs(
            page=2, limit=5, status=["running", "completed"], project_id="proj_1", client=c
        )
        params = c.list_training_jobs.call_args.kwargs
        assert params["page"] == 2
        assert params["limit"] == 5
        assert params["status_filter"] == "running,completed"
        assert params["project_id"] == "proj_1"

    def test_list_accepts_single_status_string(self) -> None:
        c = MagicMock(spec=DagnamClient, list_training_jobs=MagicMock(return_value={"items": []}))
        list_training_jobs(status="running", client=c)
        assert c.list_training_jobs.call_args.kwargs["status_filter"] == "running"

    def test_list_omits_unset_filters(self) -> None:
        c = MagicMock(spec=DagnamClient, list_training_jobs=MagicMock(return_value={"items": []}))
        list_training_jobs(client=c)
        params = c.list_training_jobs.call_args.kwargs
        assert "status_filter" not in params
        assert "project_id" not in params

    def test_cancel_delegates(self) -> None:
        c = MagicMock(
            spec=DagnamClient,
            cancel_training_job=MagicMock(return_value={"message": "Training job cancelled"}),
        )
        assert cancel_training_job("j1", client=c) == {"message": "Training job cancelled"}
        c.cancel_training_job.assert_called_once_with("j1")

    def test_delete_stringifies_ids(self) -> None:
        c = MagicMock(
            spec=DagnamClient,
            bulk_delete_training_jobs=MagicMock(return_value={"deleted": 2}),
        )
        delete_training_jobs(["j1", "j2"], client=c)
        c.bulk_delete_training_jobs.assert_called_once_with(["j1", "j2"])


class TestTrainingHistory:
    def test_logs_delegate_filters(self) -> None:
        c = MagicMock(spec=DagnamClient)
        c.get_training_logs.return_value = {"items": []}

        assert training_logs("j1", log_level="error", client=c) == {"items": []}
        c.get_training_logs.assert_called_once_with(
            "j1",
            log_level="error",
            source=None,
            page=1,
            limit=100,
        )

    def test_metrics_delegate_filters(self) -> None:
        c = MagicMock(spec=DagnamClient)
        c.get_training_metrics.return_value = {"items": []}

        assert training_metrics("j1", metric_type="train_loss", epoch_summary=True, client=c) == {
            "items": []
        }
        c.get_training_metrics.assert_called_once_with(
            "j1",
            metric_type="train_loss",
            epoch_start=None,
            epoch_end=None,
            epoch_summary=True,
            page=1,
            limit=100,
        )

    def test_metrics_summary_delegates(self) -> None:
        c = MagicMock(spec=DagnamClient)
        c.get_training_metrics_summary.return_value = {"best_epoch": 2}

        assert training_metrics_summary("j1", client=c) == {"best_epoch": 2}
        c.get_training_metrics_summary.assert_called_once_with("j1")
