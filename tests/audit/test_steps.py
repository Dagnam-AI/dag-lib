"""The shared step ground: the client protocol, the context, the helpers, the waits."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path

import pytest
from tests.audit._platform import Clock, FakePlatform, as_client

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import LROFailedError, LROTimeoutError, PlatformAnswerError
from dagnam.audit.candidates import SFT_SMALL, CandidateKind
from dagnam.audit.state import AuditState, StepState
from dagnam.audit.steps import (
    PlatformClient,
    StepContext,
    answer_field,
    error_code,
    replay_file,
    required,
    string_field,
    wait_for,
    wait_task,
)
from dagnam.audit.workspace import UnsafeWorkloadsError


def test_the_real_client_and_the_fake_both_satisfy_the_protocol() -> None:
    real: PlatformClient = DagnamClient("https://x", "key")
    fake: PlatformClient = as_client(FakePlatform())
    assert real.api_url == fake.api_url == "https://x"


def test_context_paths_label_and_step(
    make_ctx: Callable[..., StepContext], audit_dir: Path
) -> None:
    ctx = make_ctx(workload_id="w2", spec=SFT_SMALL)
    assert ctx.workload_dir == audit_dir / "workloads" / "w2"
    assert ctx.label == "w2/sft_small"
    assert ctx.meta()["stats"]["format_key"] == "chat-messages"
    state = AuditState()
    step = ctx.step(state)
    assert step is state.workloads["w2"][SFT_SMALL.kind]


def test_required_names_the_missing_key() -> None:
    assert required("x", "dataset_id") == "x"
    with pytest.raises(RuntimeError, match="dataset_id is not set yet"):
        required(None, "dataset_id")


def test_string_field_and_error_code() -> None:
    assert string_field({"a": "s", "b": 1}, "a") == "s"
    assert string_field({"a": "s", "b": 1}, "b") is None
    assert string_field({}, "a") is None
    assert error_code(StepState()) is None
    assert error_code(StepState(error="pii_disagreement: details")) == "pii_disagreement"
    assert error_code(StepState(error="bare")) == "bare"


def test_wait_for_polls_on_the_context_clock(
    make_ctx: Callable[..., StepContext], clock: Clock
) -> None:
    ctx = make_ctx()
    statuses = iter(["pending", "pending", "done"])
    payload = wait_for(
        ctx,
        lambda: {"status": next(statuses)},
        success={"done"},
        failure={"failed"},
        timeout=60,
        name="thing",
    )
    assert payload == {"status": "done"}
    assert clock.sleeps == [2.0, 4.0]


def test_wait_for_raises_on_failure_and_timeout(make_ctx: Callable[..., StepContext]) -> None:
    ctx = make_ctx()
    with pytest.raises(LROFailedError) as failed:
        wait_for(
            ctx,
            lambda: {"status": "failed", "error_message": "oom"},
            success={"done"},
            failure={"failed"},
            timeout=60,
            name="thing",
        )
    assert failed.value.detail == "oom"
    with pytest.raises(LROTimeoutError):
        wait_for(
            ctx, lambda: {"status": "pending"}, success={"done"}, failure=set(), timeout=5, name="t"
        )


def test_wait_task_returns_the_result_object_or_nothing(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = make_ctx()
    platform.memberships = {"train": [0]}
    assert wait_task(ctx, "split:ds-1-1")["status"] == "completed"

    monkeypatch.setattr(
        platform, "get_dataset_task_status", lambda task_id: {"status": "SUCCESS", "result": "x"}
    )
    assert wait_task(ctx, "t") == {}

    monkeypatch.setattr(
        platform, "get_dataset_task_status", lambda task_id: {"status": "FAILURE", "error": "disk"}
    )
    with pytest.raises(LROFailedError) as failed:
        wait_task(ctx, "t")
    assert failed.value.detail == "disk"


def test_a_context_never_names_a_folder_outside_workloads(
    make_ctx: Callable[..., StepContext],
) -> None:
    """The id comes from a scan report a person can edit: it is one plain folder name or nothing."""
    ctx = make_ctx(workload_id="../outside")
    with pytest.raises(ValueError, match="is not a plain name"):
        _ = ctx.workload_dir
    with pytest.raises(ValueError, match="is not a plain name"):
        ctx.meta()
    with pytest.raises(ValueError, match="is not a plain name"):
        replay_file(ctx.audit_dir, "a/b", CandidateKind.HEAD_TUNE)
    assert (
        replay_file(ctx.audit_dir, "w1", CandidateKind.HEAD_TUNE).name == "replay-head_tune.jsonl"
    )
    assert make_ctx().meta()["rows"] == 20


@pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege on Windows")
def test_a_context_never_reads_or_writes_through_a_folder_that_is_a_link(
    make_ctx: Callable[..., StepContext], tmp_path: Path
) -> None:
    ctx = make_ctx()
    moved = tmp_path / "moved"
    (ctx.audit_dir / "workloads" / "w1").rename(moved)
    (ctx.audit_dir / "workloads" / "w1").symlink_to(moved, target_is_directory=True)
    with pytest.raises(UnsafeWorkloadsError):
        _ = ctx.workload_dir
    with pytest.raises(UnsafeWorkloadsError):
        ctx.meta()
    with pytest.raises(UnsafeWorkloadsError):
        replay_file(ctx.audit_dir, "w1", CandidateKind.HEAD_TUNE)


def test_a_field_the_platforms_answer_lacks_is_said_so_not_a_key_error() -> None:
    assert answer_field({"id": 7}, "id", "the create") == "7"
    with pytest.raises(
        PlatformAnswerError, match="the platform's answer to the create had no 'id'"
    ):
        answer_field({"status": "ok"}, "id", "the create")
    with pytest.raises(PlatformAnswerError):
        answer_field({"id": None}, "id", "the create")
