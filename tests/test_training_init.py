"""training.init gates uploads and stamps reporter events."""

from __future__ import annotations

import importlib
import json

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
