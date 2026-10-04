"""A published run adopts a lost upload only when the listing's own row proves it is that one."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from tests.audit._platform import FakePlatform

from dagnam._core.exceptions import APIError
from dagnam._types import JsonObject, JsonValue
from dagnam.audit.state import AuditState
from dagnam.audit.steps import StepContext
from dagnam.audit.steps_data import upload


class TestAdoptionNeverTrustsTheFilter:
    """A listing may hold anything the caller can see; only what a row says of itself counts.

    A second audit of the same workload, into a new directory, names its dataset exactly as the
    first did, and the ``unstructured-*`` names are the same for every account: adopting by name
    would train this audit on last month's rows, or on another account's public dataset, and
    upload nothing.
    """

    def _lost_upload(
        self, make_ctx: Callable[..., StepContext], platform: FakePlatform, audit_id: str
    ) -> StepContext:
        platform.lost_uploads = 1
        with pytest.raises(APIError):
            upload(AuditState(audit_id=audit_id), make_ctx())
        return make_ctx(floor=0.5)

    def _foreign(self, platform: FakePlatform, ctx: StepContext, **fields: JsonValue) -> None:
        size = (ctx.workload_dir / "dataset.jsonl").stat().st_size
        platform.other_datasets.append(
            {
                "id": "ds-OTHER",
                "name": "audit-w1-head_tune",
                "description": "",
                "size_bytes": size,
                "num_samples": 0,
                "audit_id": "audit-2",
                "owner_id": platform.owner_id,
                **fields,
            }
        )

    def test_the_first_audits_dataset_in_a_listing_that_ignores_the_filter_is_not_taken(
        self, make_ctx: Callable[..., StepContext], platform: FakePlatform
    ) -> None:
        platform.ignores_audit_filter = True
        ctx = make_ctx()
        self._foreign(platform, ctx)  # same name, same rows, another audit's

        step = ctx.step(upload(AuditState(audit_id="audit-1"), ctx))

        assert step.dataset_id == "ds-1"  # this run's own upload, not ds-OTHER
        assert platform.dataset_tags["ds-1"] == "audit-1"
        assert len(platform.uploads) == 1

    def test_another_accounts_public_dataset_of_the_same_name_is_not_taken(
        self, make_ctx: Callable[..., StepContext], platform: FakePlatform
    ) -> None:
        platform.ignores_audit_filter = True
        ctx = make_ctx()
        self._foreign(platform, ctx, audit_id=None, owner_id="somebody-else")

        assert ctx.step(upload(AuditState(audit_id="audit-1"), ctx)).dataset_id == "ds-1"

    def test_a_row_tagged_with_this_audit_but_owned_by_another_account_is_not_taken(
        self, make_ctx: Callable[..., StepContext], platform: FakePlatform
    ) -> None:
        platform.ignores_audit_filter = True
        ctx = make_ctx()
        self._foreign(platform, ctx, audit_id="audit-1", owner_id="somebody-else")

        assert ctx.step(upload(AuditState(audit_id="audit-1"), ctx)).dataset_id == "ds-1"

    def test_a_platform_that_returns_no_audit_id_never_adopts(
        self, make_ctx: Callable[..., StepContext], platform: FakePlatform
    ) -> None:
        platform.omit_provenance = True
        ctx = self._lost_upload(make_ctx, platform, "audit-1")

        step = ctx.step(upload(AuditState(audit_id="audit-1"), ctx))

        assert step.dataset_id == "ds-2"  # uploaded afresh: the lost one cannot be proved ours
        assert len(platform.uploads) == 2

    def test_a_listing_that_ignores_the_filter_still_finds_this_audits_own_lost_upload(
        self, make_ctx: Callable[..., StepContext], platform: FakePlatform
    ) -> None:
        platform.ignores_audit_filter = True
        ctx = self._lost_upload(make_ctx, platform, "audit-1")
        self._foreign(platform, ctx)

        step = ctx.step(upload(AuditState(audit_id="audit-1"), ctx))

        assert step.dataset_id == "ds-1"
        assert len(platform.uploads) == 1

    def test_the_same_name_with_other_rows_is_not_the_lost_upload(
        self, make_ctx: Callable[..., StepContext], platform: FakePlatform
    ) -> None:
        ctx = self._lost_upload(make_ctx, platform, "audit-1")
        platform.dataset_sizes["ds-1"] += 1  # the platform holds a different file under the name

        assert ctx.step(upload(AuditState(audit_id="audit-1"), ctx)).dataset_id == "ds-2"

    def test_a_row_with_no_size_cannot_be_compared_and_is_not_taken(
        self,
        make_ctx: Callable[..., StepContext],
        platform: FakePlatform,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ctx = self._lost_upload(make_ctx, platform, "audit-1")
        real = platform.list_datasets

        def without_size(**kwargs: Any) -> list[JsonObject]:
            return [{k: v for k, v in row.items() if k != "size_bytes"} for row in real(**kwargs)]

        monkeypatch.setattr(platform, "list_datasets", without_size)
        assert ctx.step(upload(AuditState(audit_id="audit-1"), ctx)).dataset_id == "ds-2"

    @pytest.mark.parametrize(("more", "adopted"), [(5, "ds-2"), (0, "ds-1")])
    def test_an_analysed_dataset_must_also_hold_this_runs_rows(
        self,
        make_ctx: Callable[..., StepContext],
        platform: FakePlatform,
        monkeypatch: pytest.MonkeyPatch,
        more: int,
        adopted: str,
    ) -> None:
        ctx = self._lost_upload(make_ctx, platform, "audit-1")
        real = platform.list_datasets
        rows = len((ctx.workload_dir / "dataset.jsonl").read_text(encoding="utf-8").splitlines())

        def analysed(**kwargs: Any) -> list[JsonObject]:
            return [
                {**row, "analysis_status": "completed", "num_samples": rows + more}
                for row in real(**kwargs)
            ]

        monkeypatch.setattr(platform, "list_datasets", analysed)
        assert ctx.step(upload(AuditState(audit_id="audit-1"), ctx)).dataset_id == adopted

    def test_a_platform_that_names_no_owner_for_the_audit_never_adopts(
        self, make_ctx: Callable[..., StepContext], platform: FakePlatform
    ) -> None:
        platform.omit_provenance = True
        ctx = make_ctx()
        self._foreign(platform, ctx, audit_id="audit-1")
        platform.other_datasets[0].pop(
            "owner_id"
        )  # no owner on the row either: None is not a match
        assert ctx.step(upload(AuditState(audit_id="audit-1"), ctx)).dataset_id == "ds-1"
