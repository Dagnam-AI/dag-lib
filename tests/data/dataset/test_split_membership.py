"""
Tests for server-declared split membership — the client half of "generated
training scripts pull exactly the split the platform created".

A percentage is not a consumable split. Before this, `split="train"` made the
SDK re-derive membership from its own seeded shuffle, a partition with no
relationship to the split the user created — so the eval holdout a
contamination guard had cleared was not the one a run evaluated against.

Covers:
- `/meta`'s `splits` payload parses into `DagnamDataset.split_membership`
- `select_split_indices` prefers membership over the ratio partition
- out-of-range indices are dropped (a stale cached file can be shorter than
  the version the membership was recorded against)
- `train`/`val`/`test` alias onto a platform `eval_holdout` split
- a split that resolves to nothing **raises** instead of silently falling back
  to a ratio slice over rows the declared splits already own
- a dataset with no membership keeps the exact previous ratio behaviour
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dagnam._types import JsonObject, JsonValue
from dagnam.data.dataset.base import DagnamDataset
from dagnam.data.loaders.media import (
    resolve_membership_name,
    select_split_indices,
)

BASE_META: JsonObject = {
    "id": "ds-1",
    "name": "ds",
    "format": "json",
    "dataset_type": "tabular",
    "num_samples": 5,
    "num_classes": 2,
}


def _dataset(splits: JsonValue | None = None) -> DagnamDataset:
    meta: JsonObject = dict(BASE_META)
    if splits is not None:
        meta["splits"] = splits
    return DagnamDataset(meta, Path())


class TestParseSplitMembership:
    def test_parses_declared_membership(self) -> None:
        ds = _dataset(
            [
                {"split_name": "train", "num_samples": 3, "member_row_indices": [0, 2, 4]},
                {"split_name": "eval_holdout", "num_samples": 2, "member_row_indices": [1, 3]},
            ]
        )
        assert ds.split_membership == {"train": [0, 2, 4], "eval_holdout": [1, 3]}

    def test_absent_splits_key_yields_no_membership(self) -> None:
        assert _dataset().split_membership == {}

    def test_legacy_split_without_membership_is_omitted(self) -> None:
        # `member_row_indices: null` is what makes the loaders fall back.
        ds = _dataset([{"split_name": "train", "num_samples": 5, "member_row_indices": None}])
        assert ds.split_membership == {}

    def test_malformed_entries_are_skipped_not_fatal(self) -> None:
        ds = _dataset(
            [
                "not a dict",
                {"num_samples": 1},
                {"split_name": 7, "member_row_indices": [0]},
                {"split_name": "train", "member_row_indices": [0, "x", True, 2]},
            ]
        )
        # A metadata surprise must never make a dataset unloadable; bools are
        # excluded because `isinstance(True, int)` would smuggle in index 1.
        assert ds.split_membership == {"train": [0, 2]}

    def test_non_list_splits_value_yields_no_membership(self) -> None:
        assert _dataset({"train": [0]}).split_membership == {}


class TestSelectSplitIndicesWithMembership:
    def test_membership_wins_over_the_ratio_partition(self) -> None:
        membership = {"train": [4, 1, 0], "eval_holdout": [2, 3]}
        assert select_split_indices(5, "train", membership=membership) == [4, 1, 0]
        assert select_split_indices(5, "eval_holdout", membership=membership) == [2, 3]

    def test_membership_order_is_preserved(self) -> None:
        assert select_split_indices(5, "train", membership={"train": [3, 0, 1]}) == [3, 0, 1]

    def test_out_of_range_indices_are_dropped(self) -> None:
        # A cached file shorter than the version the membership was recorded
        # against would otherwise IndexError deep inside a framework loader.
        assert select_split_indices(3, "train", membership={"train": [0, 5, 2, -1]}) == [0, 2]

    def test_empty_membership_mapping_falls_back_to_ratios(self) -> None:
        assert select_split_indices(10, "train", membership={}) == select_split_indices(10, "train")

    def test_none_membership_falls_back_to_ratios(self) -> None:
        assert select_split_indices(10, "test", membership=None) == select_split_indices(10, "test")


class TestSplitAliasing:
    @pytest.mark.parametrize("requested", ["val", "validation", "test"])
    def test_evaluation_names_resolve_to_eval_holdout(self, requested: str) -> None:
        membership = {"train": [0, 1], "eval_holdout": [2, 3]}
        assert select_split_indices(4, requested, membership=membership) == [2, 3]

    def test_exact_name_beats_an_alias(self) -> None:
        membership = {"val": [0], "eval_holdout": [1]}
        assert select_split_indices(2, "val", membership=membership) == [0]

    def test_resolve_membership_name_returns_none_when_undeclared(self) -> None:
        assert resolve_membership_name("holdout", {"train": [0]}) is None

    def test_unresolvable_split_raises_instead_of_ratio_slicing(self) -> None:
        # Silently ratio-slicing here would hand back rows `train` already
        # owns — the exact leakage server-side splits exist to prevent.
        with pytest.raises(ValueError, match="not declared by this dataset"):
            select_split_indices(4, "holdout", membership={"train": [0, 1]})


class TestRatioFallbackUnchanged:
    def test_partition_is_disjoint_and_complete(self) -> None:
        parts = [select_split_indices(20, name) for name in ("train", "val", "test")]
        combined = [index for part in parts for index in part]
        assert sorted(combined) == list(range(20))

    def test_unknown_split_without_membership_raises(self) -> None:
        with pytest.raises(ValueError, match="declares no server-side splits"):
            select_split_indices(4, "eval_holdout")


class TestDatasetIndicesForSplit:
    def test_instance_resolver_uses_declared_membership(self) -> None:
        ds = _dataset([{"split_name": "train", "member_row_indices": [4, 0]}])
        assert ds.indices_for_split(5, "train", 0.1, 0.1, 42) == [4, 0]

    def test_instance_resolver_falls_back_without_membership(self) -> None:
        ds = _dataset()
        assert ds.indices_for_split(20, "train", 0.1, 0.1, 42) == select_split_indices(
            20, "train", val_ratio=0.1, test_ratio=0.1, seed=42
        )

    def test_static_wrapper_stays_ratio_only(self) -> None:
        # The static entry point has no dataset to read membership from, so it
        # must keep its documented ratio-only contract.
        assert DagnamDataset.split_indices(20, "train", 0.1, 0.1, 42) == select_split_indices(
            20, "train", val_ratio=0.1, test_ratio=0.1, seed=42
        )
