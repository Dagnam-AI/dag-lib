"""Unit tests for the public ``dagnam.foundation`` resource surface."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from dagnam import foundation
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import DagnamError, RunFailedError

# The served recipe document, schema and all. Reused by the pass-through test
# below, which asserts deep equality so a dropped bound or a filtered key is a
# visible diff rather than a silently narrower form.
RECIPE: dict[str, Any] = {
    "key": "qlora-sft-chat@1.0",
    "display_name": "QLoRA SFT (chat)",
    "compatible_families": ["llama", "qwen"],
    "hyperparameters": {
        "type": "object",
        "properties": {
            "max_seq_length": {"type": "integer", "minimum": 128, "maximum": 4096, "default": 1024},
            "learning_rate": {"type": "number", "exclusiveMinimum": 0.0, "default": 0.0002},
        },
        "required": [],
    },
    "future_field": {"nested": ["a recipe grew a key the SDK does not know"]},
}

# The three ids every submit needs, as the request body spells them.
SUBMIT_IDS: dict[str, Any] = {
    "project_id": "p1",
    "base_catalog_entry_id": "b1",
    "dataset_version_id": "dv1",
}


def test_list_bases_delegates_with_defaults() -> None:
    entries = [{"id": "b1", "family": "llama"}]
    c = MagicMock(spec=DagnamClient, list_foundation_catalog=MagicMock(return_value=entries))
    assert foundation.list_bases(client=c) == entries
    c.list_foundation_catalog.assert_called_once_with(page=1, limit=20)


def test_list_bases_forwards_pagination() -> None:
    c = MagicMock(spec=DagnamClient, list_foundation_catalog=MagicMock(return_value=[]))
    foundation.list_bases(page=2, limit=50, client=c)
    c.list_foundation_catalog.assert_called_once_with(page=2, limit=50)


def test_list_recipes_returns_the_served_document_unchanged() -> None:
    """No re-declaration, no allow-list: what the API served is what returns.

    A caller renders its form from ``hyperparameters``; the moment this layer
    reshapes a recipe, the form's bounds and the API's validator are two
    sources that can disagree.
    """
    c = MagicMock(spec=DagnamClient, list_training_recipes=MagicMock(return_value=[RECIPE]))
    result = foundation.list_recipes(client=c)
    assert result == [RECIPE]
    assert result[0] is RECIPE


def test_submit_builds_the_request_body() -> None:
    c = MagicMock(spec=DagnamClient, create_foundation_run=MagicMock(return_value={"run_id": "r1"}))
    assert foundation.submit(
        project_id="p1",
        base_catalog_entry_id="b1",
        dataset_version_id="dv1",
        recipe_key="qlora-sft-chat@1.0",
        hyperparameters={"max_seq_length": 2048},
        dataset_field_bindings={"messages": "conversation"},
        client=c,
    ) == {"run_id": "r1"}
    c.create_foundation_run.assert_called_once_with(
        {
            **SUBMIT_IDS,
            "recipe_key": "qlora-sft-chat@1.0",
            "hyperparameters": {"max_seq_length": 2048},
            "dataset_field_bindings": {"messages": "conversation"},
        }
    )


def test_submit_passes_hyperparameters_through_verbatim() -> None:
    """An unknown hyperparameter reaches the API, which owns the verdict.

    The recipe's schema is the only place a hyperparameter is declared, so
    this layer must not filter to the keys it happens to know -- a recipe that
    grows a field would otherwise be unusable until the SDK is re-released.
    """
    c = MagicMock(spec=DagnamClient, create_foundation_run=MagicMock(return_value={}))
    foundation.submit(
        project_id="p1",
        base_catalog_entry_id="b1",
        dataset_version_id="dv1",
        recipe_key="qlora-sft-chat@1.0",
        hyperparameters={"a_field_added_after_this_release": 7},
        client=c,
    )
    sent = c.create_foundation_run.call_args.args[0]
    assert sent["hyperparameters"] == {"a_field_added_after_this_release": 7}


def test_submit_omitted_optionals_become_empty_objects() -> None:
    """The API declares both as objects, so ``null`` would be rejected."""
    c = MagicMock(spec=DagnamClient, create_foundation_run=MagicMock(return_value={}))
    foundation.submit(
        project_id="p1",
        base_catalog_entry_id="b1",
        dataset_version_id="dv1",
        recipe_key="qlora-sft-chat@1.0",
        client=c,
    )
    sent = c.create_foundation_run.call_args.args[0]
    assert sent["hyperparameters"] == {}
    assert sent["dataset_field_bindings"] == {}


def test_get_run_delegates() -> None:
    body = {"run_id": "r1", "status": "running"}
    c = MagicMock(spec=DagnamClient, get_foundation_run=MagicMock(return_value=body))
    assert foundation.get_run("r1", client=c) == body
    c.get_foundation_run.assert_called_once_with("r1")


# ------------------------------------------------------------------ evaluate

EVALUATION_RUN: dict[str, Any] = {
    "run_id": "e1",
    "training_job_id": "j1",
    "resolved_params": {"accuracy@1": {}},
}


def test_evaluate_builds_the_request_body() -> None:
    c = MagicMock(spec=DagnamClient, create_evaluation=MagicMock(return_value=EVALUATION_RUN))
    assert (
        foundation.evaluate(
            project_id="p1",
            subject_version_id="v1",
            baseline_version_id="v0",
            dataset_version_id="dv1",
            dataset_split="test",
            scorer_keys=["accuracy@1", "macro-f1@1"],
            scorer_params={"macro-f1@1": {"average": "macro"}},
            confirm_resource_warning=True,
            client=c,
        )
        == EVALUATION_RUN
    )
    c.create_evaluation.assert_called_once_with(
        {
            "project_id": "p1",
            "subject_version_id": "v1",
            "baseline_version_id": "v0",
            "dataset_version_id": "dv1",
            "dataset_split": "test",
            "scorer_keys": ["accuracy@1", "macro-f1@1"],
            "scorer_params": {"macro-f1@1": {"average": "macro"}},
            "confirm_resource_warning": True,
        }
    )


def test_evaluate_omitted_optionals_use_their_declared_defaults() -> None:
    """``baseline_version_id`` stays ``None`` (the API's own optional default);
    ``scorer_params`` becomes ``{}`` (the API declares it as an object, so
    ``null`` would be rejected); ``confirm_resource_warning`` defaults False."""
    c = MagicMock(spec=DagnamClient, create_evaluation=MagicMock(return_value={}))
    foundation.evaluate(
        project_id="p1",
        subject_version_id="v1",
        dataset_version_id="dv1",
        dataset_split="test",
        scorer_keys=["accuracy@1"],
        client=c,
    )
    sent = c.create_evaluation.call_args.args[0]
    assert sent["baseline_version_id"] is None
    assert sent["scorer_params"] == {}
    assert sent["confirm_resource_warning"] is False


def test_evaluate_passes_scorer_params_through_verbatim() -> None:
    """A scorer's params are declared by the scorer, not this layer -- an
    unrecognised key must still reach the API, which owns the verdict."""
    c = MagicMock(spec=DagnamClient, create_evaluation=MagicMock(return_value={}))
    foundation.evaluate(
        project_id="p1",
        subject_version_id="v1",
        dataset_version_id="dv1",
        dataset_split="test",
        scorer_keys=["accuracy@1"],
        scorer_params={"accuracy@1": {"a_param_added_after_this_release": 7}},
        client=c,
    )
    sent = c.create_evaluation.call_args.args[0]
    assert sent["scorer_params"] == {"accuracy@1": {"a_param_added_after_this_release": 7}}


def test_get_evaluation_delegates() -> None:
    c = MagicMock(spec=DagnamClient, get_evaluation=MagicMock(return_value=EVALUATION_RUN))
    assert foundation.get_evaluation("e1", client=c) == EVALUATION_RUN
    c.get_evaluation.assert_called_once_with("e1")


def test_list_evaluations_delegates() -> None:
    results = [{"scorer_key": "accuracy@1", "value": 0.9}]
    c = MagicMock(spec=DagnamClient, list_version_evaluations=MagicMock(return_value=results))
    assert foundation.list_evaluations("v1", client=c) == results
    c.list_version_evaluations.assert_called_once_with("v1")


# ------------------------------------------------------------------ wait_run


class _Clock:
    """Deterministic ``now``/``sleep`` pair: sleeping advances the clock."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def _run_client(*statuses: dict[str, Any]) -> MagicMock:
    return MagicMock(spec=DagnamClient, get_foundation_run=MagicMock(side_effect=list(statuses)))


def test_wait_run_returns_the_completed_run() -> None:
    clock = _Clock()
    c = _run_client({"status": "pending"}, {"status": "running"}, {"status": "completed", "x": 1})
    run = foundation.wait_run("r1", poll_seconds=15, client=c, sleep=clock.sleep, now=clock.now)
    assert run == {"status": "completed", "x": 1}
    assert clock.sleeps == [15, 15]
    c.get_foundation_run.assert_called_with("r1")


def test_wait_run_returns_immediately_when_already_terminal() -> None:
    clock = _Clock()
    c = _run_client({"status": "completed"})
    foundation.wait_run("r1", client=c, sleep=clock.sleep, now=clock.now)
    assert clock.sleeps == []


@pytest.mark.parametrize("status", ["failed", "cancelled", "timeout"])
def test_wait_run_raises_run_failed_with_the_server_reason(status: str) -> None:
    c = _run_client({"status": status, "error_message": "OOM on step 3"})
    with pytest.raises(RunFailedError, match="OOM on step 3") as exc_info:
        foundation.wait_run("r1", client=c, sleep=lambda _s: None, now=lambda: 0.0)
    assert exc_info.value.run_id == "r1"
    assert exc_info.value.status == status
    assert exc_info.value.reason == "OOM on step 3"
    assert isinstance(exc_info.value, DagnamError)


def test_wait_run_failed_without_a_reason_still_names_the_status() -> None:
    c = _run_client({"status": "cancelled", "error_message": None})
    with pytest.raises(RunFailedError, match="cancelled") as exc_info:
        foundation.wait_run("r1", client=c, sleep=lambda _s: None, now=lambda: 0.0)
    assert exc_info.value.reason is None


def test_wait_run_times_out_without_oversleeping() -> None:
    clock = _Clock()
    c = MagicMock(
        spec=DagnamClient, get_foundation_run=MagicMock(return_value={"status": "running"})
    )
    with pytest.raises(TimeoutError, match="r1"):
        foundation.wait_run(
            "r1", poll_seconds=15, timeout=40, client=c, sleep=clock.sleep, now=clock.now
        )
    # 15 + 15 + the 10s remainder, then one last poll at the deadline.
    assert clock.sleeps == [15, 15, 10]
    assert c.get_foundation_run.call_count == 4
