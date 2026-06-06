"""training.init gates uploads and stamps reporter events."""

from __future__ import annotations

import importlib
import json
import threading

import pytest


@pytest.fixture
def training_mod(tmp_path, monkeypatch):
    monkeypatch.setenv("DAGNAM_METRICS_PATH", str(tmp_path / "metrics.jsonl"))
    import dagnam.training as training

    importlib.reload(training)
    training._reset()
    yield training
    training._reset()


def test_internal_mode_never_uploads(training_mod, monkeypatch):
    started = {"count": 0}
    monkeypatch.setattr(
        training_mod,
        "_start_uploader",
        lambda *args, **kwargs: started.__setitem__("count", started["count"] + 1),
    )
    monkeypatch.setenv("DAGNAM_INTERNAL", "1")

    training_mod.init(project_id="proj_1")

    assert started["count"] == 0


def test_offline_falls_back_to_file_only(training_mod, monkeypatch):
    from dagnam._core import auth
    from dagnam._core.exceptions import AuthError

    started = {"count": 0}
    monkeypatch.setattr(
        training_mod,
        "_start_uploader",
        lambda *args, **kwargs: started.__setitem__("count", started["count"] + 1),
    )
    monkeypatch.delenv("DAGNAM_INTERNAL", raising=False)

    def no_key():
        raise AuthError("no key")

    monkeypatch.setattr(auth, "get_api_key", no_key)
    training_mod.init(project_id="proj_1")

    assert started["count"] == 0


def test_online_starts_uploader_and_stamps_schema(training_mod, monkeypatch):
    from dagnam._core import auth

    started = {"count": 0}
    monkeypatch.setattr(
        training_mod,
        "_start_uploader",
        lambda *args, **kwargs: started.__setitem__("count", started["count"] + 1),
    )
    monkeypatch.delenv("DAGNAM_INTERNAL", raising=False)
    monkeypatch.setattr(auth, "get_api_key", lambda: "sk_test")

    training_mod.init(project_id="proj_1", name="run-fixed")
    training_mod.report_metric(epoch=0, step=1, metrics={"loss": 1.0})

    assert started["count"] == 1
    with open(training_mod._metrics_path, encoding="utf-8") as metrics_file:
        last = [json.loads(line) for line in metrics_file if line.strip()][-1]
    assert last["schema_version"] == "1"


def test_invalid_mode_raises(training_mod):
    with pytest.raises(ValueError, match="mode must be"):
        training_mod.init(project_id="p1", mode="bogus")


def test_mode_off_never_uploads(training_mod, monkeypatch):
    started = {"count": 0}
    monkeypatch.setattr(
        training_mod,
        "_start_uploader",
        lambda *_a, **_kw: started.__setitem__("count", started["count"] + 1),
    )
    monkeypatch.delenv("DAGNAM_INTERNAL", raising=False)
    training_mod.init(project_id="p1", mode="off")
    assert started["count"] == 0


def test_project_id_env_overrides_arg(training_mod, monkeypatch):
    monkeypatch.setenv("DAGNAM_PROJECT_ID", "env-proj")
    monkeypatch.setattr(training_mod, "_start_uploader", lambda *_a, **_kw: None)
    monkeypatch.delenv("DAGNAM_INTERNAL", raising=False)
    training_mod.init(project_id="arg-proj", mode="off")
    assert training_mod._project_id == "env-proj"


# ---------------------------------------------------------------- reporter branches


def test_report_system_includes_all_fields(training_mod):
    training_mod.report_system(
        gpu_utilization=0.5, gpu_memory_used=10, gpu_memory_total=20, cpu_percent=0.3
    )
    with open(training_mod._metrics_path, encoding="utf-8") as fh:
        event = [json.loads(line) for line in fh if line.strip()][-1]
    assert event["type"] == "system"
    assert event["gpu_utilization"] == 0.5
    assert event["gpu_memory_used"] == 10
    assert event["gpu_memory_total"] == 20
    assert event["cpu_percent"] == 0.3


def test_report_system_omits_none_fields(training_mod):
    training_mod.report_system()
    with open(training_mod._metrics_path, encoding="utf-8") as fh:
        event = [json.loads(line) for line in fh if line.strip()][-1]
    assert event == {"type": "system", "timestamp": event["timestamp"]}


def test_report_log_writes_level_and_message(training_mod):
    training_mod.report_log(level="warning", message="heads up")
    with open(training_mod._metrics_path, encoding="utf-8") as fh:
        event = [json.loads(line) for line in fh if line.strip()][-1]
    assert event == {
        "type": "log",
        "timestamp": event["timestamp"],
        "level": "warning",
        "message": "heads up",
    }


def test_report_error_includes_epoch_and_step(training_mod):
    training_mod.report_error(category="user_code", technical_summary="boom", epoch=3, step=7)
    with open(training_mod._metrics_path, encoding="utf-8") as fh:
        event = [json.loads(line) for line in fh if line.strip()][-1]
    assert event["epoch"] == 3
    assert event["step"] == 7
    assert "traceback" not in event
    assert training_mod._run_failed is True


def test_report_error_omits_epoch_step_and_includes_traceback(training_mod):
    # No epoch/step (191->193, 193->195 false legs) but a traceback (196).
    training_mod.report_error(
        category="user_code", technical_summary="boom", traceback="Traceback ..."
    )
    with open(training_mod._metrics_path, encoding="utf-8") as fh:
        event = [json.loads(line) for line in fh if line.strip()][-1]
    assert "epoch" not in event
    assert "step" not in event
    assert event["traceback"] == "Traceback ..."


def test_report_progress_writes_counters(training_mod):
    training_mod.report_progress(epoch=1, total_epochs=4, step=2, total_steps=8)
    with open(training_mod._metrics_path, encoding="utf-8") as fh:
        event = [json.loads(line) for line in fh if line.strip()][-1]
    assert event["type"] == "progress"
    assert event["epoch"] == 1
    assert event["total_epochs"] == 4
    assert event["step"] == 2
    assert event["total_steps"] == 8


def test_report_metric_writes_metrics(training_mod):
    training_mod.report_metric(epoch=0, step=1, metrics={"loss": 0.25})
    with open(training_mod._metrics_path, encoding="utf-8") as fh:
        event = [json.loads(line) for line in fh if line.strip()][-1]
    assert event["type"] == "metric"
    assert event["metrics"] == {"loss": 0.25}


# ---------------------------------------------------------------- helper internals


def test_configured_metrics_path_swallows_errors(training_mod, monkeypatch):
    import dagnam._core.config as config_mod

    def _boom(_key):
        raise RuntimeError("config unavailable")

    monkeypatch.setattr(config_mod, "get_config_value", _boom)
    assert training_mod._configured_metrics_path() is None


def test_configured_metrics_path_returns_configured_string(training_mod, monkeypatch):
    # Non-empty string -> returned verbatim (line 48 true leg).
    import dagnam._core.config as config_mod

    monkeypatch.setattr(config_mod, "get_config_value", lambda _key: "/tmp/configured.jsonl")
    assert training_mod._configured_metrics_path() == "/tmp/configured.jsonl"


def test_configured_metrics_path_none_for_non_string(training_mod, monkeypatch):
    # Falsy/non-string value -> None (line 48 false leg).
    import dagnam._core.config as config_mod

    monkeypatch.setattr(config_mod, "get_config_value", lambda _key: 123)
    assert training_mod._configured_metrics_path() is None


def test_resolve_metrics_path_uses_configured_when_no_env(training_mod, monkeypatch):
    # No DAGNAM_METRICS_PATH -> config path wins, not a fallback (lines 56-58).
    monkeypatch.delenv("DAGNAM_METRICS_PATH", raising=False)
    monkeypatch.setattr(training_mod, "_configured_metrics_path", lambda: "/tmp/cfg.jsonl")
    assert training_mod._resolve_metrics_path() == ("/tmp/cfg.jsonl", False)


def test_resolve_metrics_path_falls_back_to_default(training_mod, monkeypatch):
    # No env, no config -> default path flagged as fallback (lines 56, 60).
    monkeypatch.delenv("DAGNAM_METRICS_PATH", raising=False)
    monkeypatch.setattr(training_mod, "_configured_metrics_path", lambda: None)
    path, is_fallback = training_mod._resolve_metrics_path()
    assert path == training_mod._DEFAULT_METRICS_PATH
    assert is_fallback is True


def test_get_file_warns_on_fallback_path(training_mod, monkeypatch, tmp_path, capsys):
    # Exercise the `if _using_fallback_path: _warn_fallback_once()` arm (line 86).
    training_mod._close_file()
    target = tmp_path / "fallback.jsonl"
    training_mod._metrics_path = str(target)
    training_mod._using_fallback_path = True
    training_mod._fallback_warning_emitted = False
    training_mod._file = None
    training_mod._get_file()
    training_mod._close_file()
    assert "training_metrics_path" in capsys.readouterr().err


def test_warn_fallback_once_is_idempotent(training_mod, capsys):
    training_mod._fallback_warning_emitted = False
    training_mod._warn_fallback_once()
    training_mod._warn_fallback_once()
    assert capsys.readouterr().err.count("training_metrics_path") == 1


def test_warn_fallback_swallows_stderr_failure(training_mod, monkeypatch):
    training_mod._fallback_warning_emitted = False

    class _BadStderr:
        def write(self, _msg):
            raise OSError("closed stream")

    monkeypatch.setattr(training_mod.sys, "stderr", _BadStderr())
    training_mod._warn_fallback_once()  # must not raise


def test_close_file_swallows_close_error(training_mod, monkeypatch):
    class _BadFile:
        closed = False

        def close(self):
            raise OSError("cannot close")

    training_mod._file = _BadFile()
    training_mod._close_file()  # must not raise
    assert training_mod._file is None


def test_write_event_swallows_serialization_failure(training_mod, monkeypatch):
    def _boom(*_a, **_kw):
        raise RuntimeError("no file")

    monkeypatch.setattr(training_mod, "_get_file", _boom)
    training_mod._write_event({"type": "metric"})  # must not raise


def test_get_file_resolves_path_when_unset(training_mod, tmp_path, monkeypatch):
    # Force the lazy `if _metrics_path is None` resolution leg (line 83).
    target = tmp_path / "lazy.jsonl"
    monkeypatch.setenv("DAGNAM_METRICS_PATH", str(target))
    training_mod._close_file()
    training_mod._metrics_path = None
    training_mod._get_file()
    training_mod._close_file()
    assert training_mod._metrics_path == str(target)


def test_get_file_skips_makedirs_for_empty_parent(training_mod, tmp_path, monkeypatch):
    # Defensive false leg of `if parent:` (branch 88->90): os.path.dirname can
    # only be "" if abspath yields no directory, which never happens normally,
    # so stub dirname to "" to pin the guard.
    target = tmp_path / "noparent.jsonl"
    monkeypatch.setenv("DAGNAM_METRICS_PATH", str(target))
    training_mod._close_file()
    training_mod._metrics_path = str(target)
    training_mod._file = None
    real_dirname = training_mod.os.path.dirname
    monkeypatch.setattr(
        training_mod.os.path,
        "dirname",
        lambda p: "" if str(p).endswith("noparent.jsonl") else real_dirname(p),
    )
    training_mod._get_file()
    training_mod._close_file()
    assert target.is_file()


def test_write_training_state_returns_without_dir(training_mod, monkeypatch):
    monkeypatch.delenv("DAGNAM_TRAINING_DIR", raising=False)
    training_mod.write_training_state(epoch=1, step=1, latest_checkpoint_path=None)  # no-op


def test_write_training_state_swallows_errors(training_mod, monkeypatch, tmp_path):
    monkeypatch.setenv("DAGNAM_TRAINING_DIR", str(tmp_path))

    def _boom(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr(training_mod.os, "replace", _boom)
    training_mod.write_training_state(
        epoch=1, step=1, latest_checkpoint_path=None
    )  # must not raise


def test_sdk_version_returns_string(training_mod):
    assert isinstance(training_mod._sdk_version(), str)


def test_sdk_version_unknown_when_package_missing(training_mod, monkeypatch):
    from importlib import metadata

    def _missing(_name):
        raise metadata.PackageNotFoundError("dagnam")

    monkeypatch.setattr(metadata, "version", _missing)
    assert training_mod._sdk_version() == "0+unknown"


def test_sdk_version_unknown_on_unexpected_error(training_mod, monkeypatch):
    import importlib

    real_import = importlib.import_module

    def _boom(name, *args, **kwargs):
        if name == "importlib.metadata":
            raise RuntimeError("broken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _boom)
    # _sdk_version imports importlib.metadata via a from-import; force failure by
    # poisoning the cached module so the import inside the function raises.
    monkeypatch.setitem(__import__("sys").modules, "importlib.metadata", None)
    assert training_mod._sdk_version() == "0+unknown"


def test_generated_name_delegates_to_naming(training_mod, monkeypatch):
    import dagnam._core.naming as naming_mod

    monkeypatch.setattr(naming_mod, "generate_run_name", lambda: "happy-otter")
    assert training_mod._generated_name() == "happy-otter"


# ---------------------------------------------------------------- uploader lifecycle


class _FakeRun(dict):
    pass


def _wire_uploader_deps(training_mod, monkeypatch, *, run_loop=None):
    """Stub the lazily-imported _core modules used by _start_uploader."""
    import dagnam._core.auth as auth_mod

    monkeypatch.setattr(auth_mod, "get_api_url", lambda: "https://api.test")
    monkeypatch.setattr(auth_mod, "get_api_key", lambda: "sk_test")

    class _FakeClient:
        def __init__(self, _url, _key):
            pass

        def register_local_run(self, *, project_id, framework, config):
            assert project_id
            assert framework
            assert config is not None
            return _FakeRun(id="run-1")

        def mint_run_token(self, _run_id):
            return {"token": "tok"}

    import dagnam._core.client as client_mod

    monkeypatch.setattr(client_mod, "DagnamClient", _FakeClient)

    import dagnam._core.metrics_uploader as up_mod

    def _default_loop(**_kw):
        return 0

    monkeypatch.setattr(up_mod, "run_upload_loop", run_loop or _default_loop)
    monkeypatch.setattr(up_mod, "is_terminal_upload_error", lambda _exc: True)

    class _FakeSink:
        def __init__(self, *_a, **_kw):
            pass

    monkeypatch.setattr(up_mod, "HTTPSink", _FakeSink)


def test_start_uploader_runs_loop_and_finalizes(training_mod, monkeypatch):
    monkeypatch.delenv("DAGNAM_INTERNAL", raising=False)
    loop_done = threading.Event()

    def _loop(**_kw):
        loop_done.set()
        return 0

    _wire_uploader_deps(training_mod, monkeypatch, run_loop=_loop)

    training_mod._start_uploader("p1", "pytorch", "run-x")
    assert training_mod._uploader_thread is not None
    training_mod._uploader_thread.join(timeout=5.0)
    assert loop_done.is_set()
    assert training_mod._finalize_registered is True

    # _finalize_stream emits one terminal event and joins the thread.
    training_mod._finalize_stream()
    assert training_mod._stream_finalized is True
    with open(training_mod._metrics_path, encoding="utf-8") as fh:
        events = [json.loads(line) for line in fh if line.strip()]
    assert events[-1]["type"] == "complete"


def test_start_uploader_only_registers_atexit_once(training_mod, monkeypatch):
    monkeypatch.delenv("DAGNAM_INTERNAL", raising=False)
    registered = {"count": 0}
    monkeypatch.setattr(
        training_mod.atexit,
        "register",
        lambda *_a, **_kw: registered.__setitem__("count", registered["count"] + 1),
    )
    _wire_uploader_deps(training_mod, monkeypatch)

    training_mod._start_uploader("p1", "pytorch", "run-a")
    training_mod._uploader_thread.join(timeout=5.0)
    first = registered["count"]
    # Second call: thread finished -> not alive -> starts a new one, but the
    # finalize atexit is already registered so it must NOT register again.
    training_mod._start_uploader("p1", "pytorch", "run-b")
    training_mod._uploader_thread.join(timeout=5.0)
    assert registered["count"] == first  # no additional _finalize_stream registration


def test_start_uploader_skips_when_thread_alive(training_mod, monkeypatch):
    monkeypatch.delenv("DAGNAM_INTERNAL", raising=False)

    class _AliveThread:
        def is_alive(self):
            return True

    training_mod._uploader_thread = _AliveThread()  # type: ignore[assignment]
    _wire_uploader_deps(training_mod, monkeypatch)
    training_mod._start_uploader("p1", "pytorch", "run-x")
    # Still the same fake alive thread; _start_uploader returned early.
    assert isinstance(training_mod._uploader_thread, _AliveThread)


def test_start_uploader_register_failure_disables_streaming(training_mod, monkeypatch, capsys):
    monkeypatch.delenv("DAGNAM_INTERNAL", raising=False)
    import dagnam._core.auth as auth_mod

    monkeypatch.setattr(auth_mod, "get_api_url", lambda: "https://api.test")
    monkeypatch.setattr(auth_mod, "get_api_key", lambda: "sk_test")

    class _FailClient:
        def __init__(self, _url, _key):
            pass

        def register_local_run(self, **_kw):
            raise RuntimeError("tier limit")

    import dagnam._core.client as client_mod

    monkeypatch.setattr(client_mod, "DagnamClient", _FailClient)
    import dagnam._core.metrics_uploader as up_mod

    monkeypatch.setattr(up_mod, "HTTPSink", object)
    monkeypatch.setattr(up_mod, "run_upload_loop", lambda **_kw: 0)
    monkeypatch.setattr(up_mod, "is_terminal_upload_error", lambda _e: True)

    training_mod._start_uploader("p1", "pytorch", "run-x")
    assert training_mod._uploader_thread is None
    assert "could not register local run" in capsys.readouterr().err


def test_loop_surfaces_unexpected_error(training_mod, monkeypatch, capsys):
    monkeypatch.delenv("DAGNAM_INTERNAL", raising=False)

    def _loop(**_kw):
        raise RuntimeError("network blip")

    _wire_uploader_deps(training_mod, monkeypatch, run_loop=_loop)
    # Non-terminal error must be surfaced on stderr.
    monkeypatch.setattr(
        __import__("dagnam._core.metrics_uploader", fromlist=["is_terminal_upload_error"]),
        "is_terminal_upload_error",
        lambda _exc: False,
    )
    training_mod._start_uploader("p1", "pytorch", "run-x")
    training_mod._uploader_thread.join(timeout=5.0)
    assert "streaming stopped unexpectedly" in capsys.readouterr().err


def test_loop_terminal_error_is_quiet(training_mod, monkeypatch, capsys):
    monkeypatch.delenv("DAGNAM_INTERNAL", raising=False)

    def _loop(**_kw):
        raise RuntimeError("409 conflict")

    _wire_uploader_deps(training_mod, monkeypatch, run_loop=_loop)
    training_mod._start_uploader("p1", "pytorch", "run-x")
    training_mod._uploader_thread.join(timeout=5.0)
    assert "streaming stopped unexpectedly" not in capsys.readouterr().err


def test_loop_refresh_client_mints_token(training_mod, monkeypatch):
    monkeypatch.delenv("DAGNAM_INTERNAL", raising=False)
    captured: dict[str, object] = {}

    def _loop(**kw):
        # Exercise the refresh_client closure passed into HTTPSink.
        captured["refresh"] = kw
        return 0

    import dagnam._core.auth as auth_mod

    monkeypatch.setattr(auth_mod, "get_api_url", lambda: "https://api.test")
    monkeypatch.setattr(auth_mod, "get_api_key", lambda: "sk_test")

    class _FakeClient:
        def __init__(self, _url, _key):
            pass

        def register_local_run(self, **_kw):
            return _FakeRun(id="run-1")

        def mint_run_token(self, _run_id):
            return {"token": "tok"}

    import dagnam._core.client as client_mod

    monkeypatch.setattr(client_mod, "DagnamClient", _FakeClient)
    import dagnam._core.metrics_uploader as up_mod

    refresh_holder: dict[str, object] = {}

    class _CapturingSink:
        def __init__(self, _client, _run_id, *, source=None, refresh_client=None):
            refresh_holder["refresh_client"] = refresh_client

    monkeypatch.setattr(up_mod, "HTTPSink", _CapturingSink)
    monkeypatch.setattr(up_mod, "run_upload_loop", _loop)
    monkeypatch.setattr(up_mod, "is_terminal_upload_error", lambda _e: True)

    training_mod._start_uploader("p1", "pytorch", "run-x")
    training_mod._uploader_thread.join(timeout=5.0)
    refresh = refresh_holder["refresh_client"]
    assert callable(refresh)
    refreshed = refresh()  # type: ignore[operator]
    assert isinstance(refreshed, _FakeClient)


# ---------------------------------------------------------------- _finalize_stream guards


def test_finalize_stream_noop_when_internal(training_mod, monkeypatch):
    monkeypatch.setenv("DAGNAM_INTERNAL", "1")
    training_mod._uploader_thread = object()  # type: ignore[assignment]
    training_mod._stream_finalized = False
    training_mod._finalize_stream()
    assert training_mod._stream_finalized is False  # early return


def test_finalize_stream_noop_without_thread(training_mod, monkeypatch):
    monkeypatch.delenv("DAGNAM_INTERNAL", raising=False)
    training_mod._uploader_thread = None
    training_mod._finalize_stream()  # early return, no error


def test_finalize_stream_noop_when_already_finalized(training_mod, monkeypatch):
    monkeypatch.delenv("DAGNAM_INTERNAL", raising=False)
    training_mod._uploader_thread = object()  # type: ignore[assignment]
    training_mod._stream_finalized = True
    training_mod._finalize_stream()  # early return


def test_finalize_stream_writes_failed_when_run_failed(training_mod, monkeypatch):
    monkeypatch.delenv("DAGNAM_INTERNAL", raising=False)

    class _DoneThread:
        def join(self, timeout=None):
            return None

    training_mod._uploader_thread = _DoneThread()  # type: ignore[assignment]
    training_mod._uploader_stop = None
    training_mod._stream_finalized = False
    training_mod._run_failed = True
    training_mod._finalize_stream()
    with open(training_mod._metrics_path, encoding="utf-8") as fh:
        events = [json.loads(line) for line in fh if line.strip()]
    assert events[-1]["type"] == "failed"


def test_finalize_stream_sets_stop_event(training_mod, monkeypatch):
    monkeypatch.delenv("DAGNAM_INTERNAL", raising=False)
    stop = threading.Event()

    class _DoneThread:
        def join(self, timeout=None):
            return None

    training_mod._uploader_thread = _DoneThread()  # type: ignore[assignment]
    training_mod._uploader_stop = stop
    training_mod._stream_finalized = False
    training_mod._run_failed = False
    training_mod._finalize_stream()
    assert stop.is_set()
