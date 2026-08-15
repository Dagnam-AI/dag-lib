"""Unit tests for the public ``dagnam.foundation`` resource surface."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from dagnam import foundation
from dagnam._core.client import DagnamClient

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
