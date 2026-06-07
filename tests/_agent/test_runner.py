"""Tests for dagnam._agent.runner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from dagnam._agent import runner

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, StrCapture


def _patch_entitlements(monkeypatch: PytestMonkeyPatch, value: object) -> None:
    """Point ``dagnam.account.entitlements`` at a static value (or raise on a sentinel)."""
    import dagnam
    from dagnam._core.exceptions import DagnamError

    def fake() -> object:
        if isinstance(value, BaseException):
            raise DagnamError("entitlements unavailable")
        return value

    monkeypatch.setattr(dagnam.account, "entitlements", fake)


# --- plan_preview -----------------------------------------------------------


def test_plan_preview_deploy_fetch_failure_is_swallowed(
    monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    _patch_entitlements(monkeypatch, RuntimeError())  # sentinel: raise APIError
    code = runner.plan_preview(
        "deploy",
        {"project_id": "p1", "instance_type": "gpu.small", "replicas": 2, "platform": "k8s"},
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "DRY RUN" in out
    assert "deploy" in out.lower()
    assert "gpu.small" in out
    assert "replicas: 2" in out
    assert "could not fetch entitlements" in out
    assert "Nothing was executed" in out


def test_plan_preview_train_surfaces_limits(
    monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    _patch_entitlements(
        monkeypatch,
        {
            "plan": {"display_name": "Pro"},
            "limits": [{"key": "gpu_hours", "current": 3, "limit": 10}],
        },
    )
    code = runner.plan_preview(
        "train", {"project_id": "p1", "epochs": 5, "batch_size": 32, "training_dataset_id": "d1"}
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "epochs: 5" in out
    assert "plan: Pro" in out
    assert "gpu_hours: 3/10" in out


def test_plan_preview_deploy_non_dict_entitlements(
    monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    _patch_entitlements(monkeypatch, {"plan": "n/a", "limits": "n/a"})
    runner.plan_preview("deploy", {"project_id": "p1"})
    assert "plan: unknown" in capsys.readouterr().out


def test_plan_preview_train_non_dict_limit_item(
    monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    _patch_entitlements(monkeypatch, {"plan": {"code": "free"}, "limits": [123]})
    runner.plan_preview("train", {"epochs": 1})
    out = capsys.readouterr().out
    assert "plan: free" in out
    assert "123" not in out  # non-dict limit entries are skipped


def test_plan_preview_deploy_empty_limits(
    monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    _patch_entitlements(monkeypatch, {"plan": {}, "limits": []})
    runner.plan_preview("deploy", {"project_id": "p1"})
    assert "plan: unknown" in capsys.readouterr().out


def test_plan_preview_publish_skips_entitlements_and_empty_params(capsys: StrCapture) -> None:
    code = runner.plan_preview("publish", {})
    out = capsys.readouterr().out
    assert code == 0
    assert "DRY RUN: publish" in out
    assert "Plan/usage impact" not in out  # publish is not a spend action


def test_plan_preview_unknown_action_errors() -> None:
    with pytest.raises(ValueError, match="Unsupported action"):
        runner.plan_preview("frobnicate", {})


# --- plan_main --------------------------------------------------------------


def test_plan_main_parses_params(capsys: StrCapture) -> None:
    code = runner.plan_main(["--action", "delete", "--param", "project_id=p9"])
    assert code == 0
    out = capsys.readouterr().out
    assert "project_id: p9" in out


def test_plan_main_without_params(capsys: StrCapture) -> None:
    code = runner.plan_main(["--action", "publish"])
    assert code == 0
    assert "DRY RUN: publish" in capsys.readouterr().out


# --- watch_training ---------------------------------------------------------


class _FakeEvent:
    def __init__(self, event: str, data: object) -> None:
        self.event = event
        self.data = data


def _patch_stream(monkeypatch: PytestMonkeyPatch, events: list[_FakeEvent]) -> None:
    import dagnam

    monkeypatch.setattr(dagnam, "stream_training", lambda *_a, **_k: iter(events))


def test_watch_training_summarizes_and_returns_zero_on_complete(
    monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    _patch_stream(
        monkeypatch,
        [
            _FakeEvent("metric", {"name": "loss", "value": 0.5, "step": 1}),
            _FakeEvent("log", "non-dict payload is tolerated"),
            _FakeEvent("complete", {"job_id": "j1"}),
        ],
    )
    code = runner.watch_training("j1")
    out = capsys.readouterr().out
    assert code == 0
    assert "loss=0.5" in out
    assert "complete" in out.lower()


def test_watch_training_returns_one_on_failure(
    monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    _patch_stream(monkeypatch, [_FakeEvent("failed", {"error": "OOM"})])
    code = runner.watch_training("j1")
    out = capsys.readouterr().out
    assert code == 1
    assert "OOM" in out


def test_watch_training_returns_one_on_cancel(monkeypatch: PytestMonkeyPatch) -> None:
    _patch_stream(monkeypatch, [_FakeEvent("cancelled", {})])
    assert runner.watch_training("j1") == 1


def test_watch_training_no_terminal_event_returns_zero(
    monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    _patch_stream(monkeypatch, [])
    code = runner.watch_training("j1")
    assert code == 0
    assert "stream ended after 0 event" in capsys.readouterr().out


def test_watch_main_parses_job_id(monkeypatch: PytestMonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake(job_id: str) -> int:
        captured["job"] = job_id
        return 0

    monkeypatch.setattr(runner, "watch_training", fake)
    assert runner.watch_main(["j-42"]) == 0
    assert captured["job"] == "j-42"
