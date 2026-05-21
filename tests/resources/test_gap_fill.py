"""Fill remaining coverage gaps in resources/ wrappers and _core tail."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
import requests

from dagnam import deployments, hub, projects
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import (
    CheckpointNotFoundError,
    ChecksumError,
    StreamError,
)
from dagnam._core.lro import LongRunningOperation, _freeze
from dagnam.resources import checkpoints as ck_module, training as tr_module
from dagnam.resources.training import _parse_event

# ---------------------------------------------------------------- deployments


def test_deployments_stringify_uuid_branch():
    client = MagicMock(spec=DagnamClient)
    client.get_deployment.return_value = {"id": "x"}
    dep_id = uuid4()
    deployments.get(dep_id, client=client)
    client.get_deployment.assert_called_once_with(str(dep_id))


def test_deployments_health():
    client = MagicMock(spec=DagnamClient)
    client.get_deployment_health_full.return_value = {"status": "healthy"}
    assert deployments.health("dep1", client=client) == {"status": "healthy"}


def test_deployments_metrics():
    client = MagicMock(spec=DagnamClient)
    client.get_deployment_metrics.return_value = {"qps": 1}
    assert deployments.metrics("dep1", client=client, time_range="1h") == {"qps": 1}


def test_deployments_create_with_optional_fields():
    client = MagicMock(spec=DagnamClient)
    client.create_deployment.return_value = {"id": "dep1", "status": "deploying"}
    deployments.create(
        client=client,
        name="x",
        project_id="p1",
        checkpoint_path="ck",
        platform="aws",
        deployment_type="production",
        instance_type="small",
        training_job_id="job1",
        checkpoint_id="ck1",
        min_instances=1,
        max_instances=3,
        region="us-east-1",
        config={"x": 1},
    )
    payload = client.create_deployment.call_args[0][0]
    assert payload["training_job_id"] == "job1"
    assert payload["checkpoint_id"] == "ck1"
    assert payload["min_instances"] == 1
    assert payload["max_instances"] == 3
    assert payload["region"] == "us-east-1"
    assert payload["config"] == {"x": 1}


def test_deployments_update_with_config():
    client = MagicMock(spec=DagnamClient)
    client.update_deployment.return_value = {"id": "dep1"}
    deployments.update("dep1", client=client, name="new", config={"k": "v"})
    payload = client.update_deployment.call_args[0][1]
    assert payload["name"] == "new"
    assert payload["config"] == {"k": "v"}


def test_deployments_delete():
    client = MagicMock(spec=DagnamClient)
    client.delete_deployment.return_value = None
    assert deployments.delete("dep1", client=client) is None


def test_deployments_resume_returns_lro():
    client = MagicMock(spec=DagnamClient)
    client.resume_deployment.return_value = {"id": "dep1", "status": "running"}
    op = deployments.resume("dep1", client=client)
    assert isinstance(op, LongRunningOperation)


def test_deployments_stream_events_delegates_to_iter_with_reconnect():
    """stream_events builds an _open callback and calls iter_with_reconnect."""
    client = MagicMock(spec=DagnamClient)
    fake_response = MagicMock()
    fake_response.close = MagicMock()
    client.open_deployment_stream.return_value = fake_response

    with patch("dagnam.resources.deployments.iter_with_reconnect") as patched:
        patched.return_value = iter([])
        result = deployments.stream_events("dep1", client=client, include_heartbeats=True)
        # Force the generator to materialize so the function body runs.
        list(result)
    assert patched.called
    kwargs = patched.call_args.kwargs
    assert kwargs["include_heartbeats"] is True
    # Verify _open closure forwards the cursor.
    _open = patched.call_args.args[0]
    _open("cursor-x")
    client.open_deployment_stream.assert_called_with("dep1", last_event_id="cursor-x")


# ---------------------------------------------------------------- projects


def test_projects_stringify_uuid_branch():
    client = MagicMock(spec=DagnamClient)
    client.get_project.return_value = {"id": "x"}
    pid = uuid4()
    projects.get(pid, client=client)
    client.get_project.assert_called_once_with(str(pid))


def test_projects_create_with_description():
    client = MagicMock(spec=DagnamClient)
    client.create_project.return_value = {"id": "p1"}
    projects.create("title", client=client, description="d")
    payload = client.create_project.call_args[0][0]
    assert payload["description"] == "d"


def test_projects_import_dag_with_description_tags_commit():
    # NOTE: resources.projects calls client.import_project_dag(), which is not
    # part of the strict DagnamClient interface — use a loose mock.
    client = MagicMock()
    client.import_project_dag.return_value = {"id": "p1"}
    projects.import_dag(
        {"ir": "..."},
        "title",
        client=client,
        description="d",
        tags=["t1"],
        commit_message="msg",
    )
    payload = client.import_project_dag.call_args[0][0]
    assert payload["description"] == "d"
    assert payload["tags"] == ["t1"]
    assert payload["commit_message"] == "msg"


def test_projects_import_dag_existing_with_commit():
    client = MagicMock()
    client.import_project_dag_existing.return_value = {"id": "p1"}
    projects.import_dag_existing("p1", {"ir": "..."}, client=client, commit_message="msg")
    payload = client.import_project_dag_existing.call_args[0][1]
    assert payload["commit_message"] == "msg"


def test_projects_save_architecture_with_commit():
    client = MagicMock()
    client.save_project_architecture.return_value = {"version_id": "v1"}
    projects.save_architecture("p1", {"d": 1}, {"a": 1}, client=client, commit_message="m")
    payload = client.save_project_architecture.call_args[0][1]
    assert payload["commit_message"] == "m"


# ---------------------------------------------------------------- hub


def test_hub_stringify_uuid_branch():
    client = MagicMock()
    client.hub_get.return_value = {"id": "m1"}
    mid = uuid4()
    hub.get(mid, client=client)
    client.hub_get.assert_called_once_with(str(mid))


def test_hub_create_with_metadata():
    client = MagicMock()
    client.hub_create.return_value = {"id": "m1"}
    hub.create(
        client=client,
        name="n",
        description="d",
        task_type="t",
        framework="pt",
        metadata={"x": 1},
    )
    payload = client.hub_create.call_args[0][0]
    assert payload["metadata"] == {"x": 1}


# ---------------------------------------------------------------- training stream gaps


def test_parse_event_invalid_retry():
    raw = SimpleNamespace(event="x", data="{}", id=None, retry="not-int")
    ev = _parse_event(raw)
    assert ev.retry is None


class _FakeSSE:
    def __init__(self, events):
        self._events = events

    def events(self):
        for e in self._events:
            if isinstance(e, Exception):
                raise e
            yield e


def _setup_fake_sseclient(monkeypatch, scripts):
    import sys

    iterator = iter(scripts)

    class _SSEClient:
        def __init__(self, response):
            self._client = _FakeSSE(next(iterator))

        def events(self):
            return self._client.events()

    monkeypatch.setitem(sys.modules, "sseclient", SimpleNamespace(SSEClient=_SSEClient))


def test_stream_training_skips_heartbeats(monkeypatch):
    client = MagicMock(spec=DagnamClient)
    fake_resp = MagicMock()
    fake_resp.close = MagicMock()
    client.open_training_stream.return_value = fake_resp

    _setup_fake_sseclient(
        monkeypatch,
        [
            [
                SimpleNamespace(event="heartbeat", data="{}", id=None, retry=None),
                SimpleNamespace(event="complete", data="{}", id=None, retry=None),
            ]
        ],
    )

    events = list(tr_module.stream_training("job1", client=client))
    assert [e.event for e in events] == ["complete"]


def test_stream_training_reconnects_on_transport_error(monkeypatch):
    client = MagicMock(spec=DagnamClient)
    fake_resp = MagicMock()
    fake_resp.close = MagicMock()
    client.open_training_stream.return_value = fake_resp

    _setup_fake_sseclient(
        monkeypatch,
        [
            [
                SimpleNamespace(event="metric", data='{"loss":1}', id="e1", retry=None),
                requests.exceptions.ConnectionError("boom"),
            ],
            [SimpleNamespace(event="complete", data="{}", id=None, retry=None)],
        ],
    )
    monkeypatch.setattr(tr_module.time, "sleep", lambda _s: None)
    monkeypatch.setattr(tr_module.random, "uniform", lambda _a, _b: 0.0)

    events = list(tr_module.stream_training("job1", client=client))
    assert [e.event for e in events] == ["metric", "complete"]
    # Second open should use the cursor captured from event id.
    assert client.open_training_stream.call_args_list[1].kwargs == {"last_event_id": "e1"}


def test_stream_training_close_error_swallowed(monkeypatch):
    client = MagicMock(spec=DagnamClient)
    fake_resp = MagicMock()
    fake_resp.close = MagicMock(side_effect=RuntimeError("close blew up"))
    client.open_training_stream.return_value = fake_resp

    _setup_fake_sseclient(
        monkeypatch,
        [[SimpleNamespace(event="complete", data="{}", id=None, retry=None)]],
    )

    events = list(tr_module.stream_training("job1", client=client))
    assert [e.event for e in events] == ["complete"]


def test_stream_training_gives_up_after_max_attempts(monkeypatch):
    client = MagicMock(spec=DagnamClient)
    fake_resp = MagicMock()
    fake_resp.close = MagicMock()
    client.open_training_stream.return_value = fake_resp

    _setup_fake_sseclient(
        monkeypatch,
        [
            [requests.exceptions.ConnectionError("a")],
            [requests.exceptions.ConnectionError("b")],
            [requests.exceptions.ConnectionError("c")],
        ],
    )
    monkeypatch.setattr(tr_module.time, "sleep", lambda _s: None)
    monkeypatch.setattr(tr_module.random, "uniform", lambda _a, _b: 0.0)

    with pytest.raises(StreamError, match="dropped after 2"):
        list(tr_module.stream_training("job1", client=client, max_reconnects=2))


# ---------------------------------------------------------------- checkpoints gaps


def test_pick_latest_raises_on_empty():
    with pytest.raises(CheckpointNotFoundError):
        ck_module._pick_latest([])


def test_download_checkpoint_checksum_mismatch_unlinks(tmp_path, monkeypatch):
    cache = tmp_path / "ck"
    client = MagicMock(spec=DagnamClient)

    def fake_stream(_job, _ck, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"bad-bytes")
        return dest, "expected-sha256"

    client.download_checkpoint_stream.side_effect = fake_stream
    monkeypatch.setattr(ck_module, "compute_file_checksum", lambda _path: "actual-different")

    with pytest.raises(ChecksumError):
        ck_module.download_checkpoint("job1", "ck1", client=client, cache_dir=cache)


def test_download_checkpoint_checksum_mismatch_swallows_unlink_failure(tmp_path, monkeypatch):
    cache = tmp_path / "ck"
    client = MagicMock(spec=DagnamClient)

    def fake_stream(_job, _ck, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"bad")
        return dest, "expected-sha"

    client.download_checkpoint_stream.side_effect = fake_stream
    monkeypatch.setattr(ck_module, "compute_file_checksum", lambda _p: "different")

    # Patch Path.unlink to raise — exception path must be swallowed.
    original_unlink = Path.unlink

    def _raising_unlink(self, *args, **kwargs):
        raise OSError("locked")

    monkeypatch.setattr(Path, "unlink", _raising_unlink)
    try:
        with pytest.raises(ChecksumError):
            ck_module.download_checkpoint("job1", "ck1", client=client, cache_dir=cache)
    finally:
        monkeypatch.setattr(Path, "unlink", original_unlink)


def test_download_checkpoint_eviction_failure_is_logged(tmp_path, monkeypatch, caplog):
    cache = tmp_path / "ck"
    client = MagicMock(spec=DagnamClient)

    def fake_stream(_job, _ck, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"ok")
        return dest, None

    client.download_checkpoint_stream.side_effect = fake_stream
    monkeypatch.setattr(ck_module, "get_config_value", lambda key, default=None: 1024)
    monkeypatch.setattr(ck_module, "evict_lru", MagicMock(side_effect=OSError("disk full")))

    import logging

    with caplog.at_level(logging.WARNING):
        ck_module.download_checkpoint("job1", "ck1", client=client, cache_dir=cache)
    assert any("eviction failed" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------- LRO tail


def test_lro_requires_at_least_one_success_state():
    with pytest.raises(ValueError, match="success_state"):
        LongRunningOperation(
            poll=lambda: {"status": "x"},
            success_states=(),
        )


def test_lro_name_property():
    op = LongRunningOperation(
        poll=lambda: {"status": "running"},
        success_states={"running"},
        name="testop",
    )
    assert op.name == "testop"


def test_lro_done_false_when_not_polled():
    op = LongRunningOperation(
        poll=lambda: {"status": "running"},
        success_states={"running"},
    )
    assert op.done() is False


def test_lro_result_with_explicit_failure_message():
    op = LongRunningOperation(
        poll=lambda: {"status": "failed", "error_message": "OOM"},
        success_states={"running"},
    )
    op.status()
    with pytest.raises(Exception, match="OOM"):
        op.result()


def test_lro_status_force_polls():
    poll = MagicMock(return_value={"status": "running"})
    op = LongRunningOperation(poll=poll, success_states={"running"})
    op.status()
    assert poll.call_count == 1


# ---------------------------------------------------------------- _core/__init__ no-aio branch


def test_core_init_handles_missing_aio(monkeypatch):
    """When the aio subpackage fails to import, _core falls back to None."""
    import sys

    # Snapshot every dagnam._core* entry so we can restore exactly what was
    # there before. Otherwise, re-importing dagnam._core mid-suite drops the
    # cached submodule attributes (e.g. dagnam._core.resolver) that later
    # tests rely on via mock.patch("dagnam._core.resolver.X") — on Python
    # 3.10 the attribute-lookup-after-import dance in unittest.mock then
    # fails. See unittest.mock._dot_lookup.
    snapshot = {
        k: v for k, v in sys.modules.items() if k == "dagnam._core" or k.startswith("dagnam._core.")
    }

    monkeypatch.setitem(sys.modules, "dagnam._core.aio", None)
    monkeypatch.delitem(sys.modules, "dagnam._core", raising=False)
    mod = importlib.import_module("dagnam._core")
    assert mod is not None

    # Restore the original module objects so subsequent tests see the same
    # dagnam._core / dagnam._core.* tree they had before this test ran.
    for k in list(sys.modules):
        if k == "dagnam._core" or k.startswith("dagnam._core."):
            sys.modules.pop(k, None)
    sys.modules.update(snapshot)


# ---------------------------------------------------------------- _freeze


def test_freeze_handles_none():
    assert _freeze(None) == frozenset()
