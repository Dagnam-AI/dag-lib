"""Unit tests for dagnam.datasets preview/update/delete/roles resource helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dagnam._core.client import DagnamClient
from dagnam.resources import datasets


def test_preview_dataset_delegates() -> None:
    c = MagicMock(spec=DagnamClient, preview_dataset=MagicMock(return_value={"samples": []}))
    assert datasets.preview_dataset("ds-1", rows=5, client=c) == {"samples": []}
    c.preview_dataset.assert_called_once_with("ds-1", rows=5)


def test_update_dataset_delegates() -> None:
    c = MagicMock(spec=DagnamClient, update_dataset=MagicMock(return_value={"id": "ds-1"}))
    datasets.update_dataset("ds-1", name="n", description="d", visibility="public", client=c)
    c.update_dataset.assert_called_once_with("ds-1", name="n", description="d", visibility="public")


def test_delete_dataset_delegates() -> None:
    c = MagicMock(spec=DagnamClient, delete_dataset=MagicMock(return_value=None))
    assert datasets.delete_dataset("ds-1", client=c) is None
    c.delete_dataset.assert_called_once_with("ds-1")


def test_update_dataset_roles_delegates() -> None:
    c = MagicMock(
        spec=DagnamClient,
        update_dataset_roles=MagicMock(return_value={"roles_confirmed": True}),
    )
    result = datasets.update_dataset_roles(
        "ds-1", {"a": "target"}, task_type_hint="classification", client=c
    )
    assert result == {"roles_confirmed": True}
    c.update_dataset_roles.assert_called_once_with(
        "ds-1", {"a": "target"}, task_type_hint="classification"
    )


def test_update_dataset_propagates_valueerror() -> None:
    c = MagicMock(spec=DagnamClient)
    c.update_dataset.side_effect = ValueError("at least one of")
    with pytest.raises(ValueError, match="at least one of"):
        datasets.update_dataset("ds-1", client=c)
