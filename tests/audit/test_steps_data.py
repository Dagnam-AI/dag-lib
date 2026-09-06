"""Data steps: each skips itself once done, uploads exactly Task 6's files, and checks the server's scan."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from tests.audit._platform import FakePlatform
from tests.audit.conftest import HOLDOUT, TRAIN

from dagnam.audit.state import AuditState
from dagnam.audit.steps import StepContext
from dagnam.audit.steps_data import (
    pii_scan,
    resolve_version,
    split,
    upload,
    wait_pii,
    wait_split,
)


def _through_split(state: AuditState, ctx: StepContext) -> AuditState:
    for step in (upload, resolve_version, split, wait_split):
        state = step(state, ctx)
    return state


def test_upload_sends_the_derived_file_as_json_and_skips_when_done(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    ctx = make_ctx()
    state = upload(AuditState(), ctx)
    step = ctx.step(state)
    assert step.dataset_id == "ds-1"
    assert platform.uploads == {"ds-1": "labeled-example"}
    upload(state, ctx)
    assert platform.call_log == ["upload_dataset"]


def test_resolve_version_waits_for_the_sniffed_format(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    platform.version_polls = 3
    ctx = make_ctx()
    state = resolve_version(upload(AuditState(), ctx), ctx)
    assert ctx.step(state).version_id == "ds-1-v1"
    assert platform.call_log.count("list_dataset_versions") == 3
    resolve_version(state, ctx)
    assert platform.call_log.count("list_dataset_versions") == 3


def test_resolve_version_handles_no_versions_yet(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = make_ctx()
    state = upload(AuditState(), ctx)
    listings = iter(
        [
            [],
            [
                {
                    "id": "ds-1-v1",
                    "version_number": 1,
                    "num_samples": 5,
                    "data_format": "labeled-example",
                }
            ],
        ]
    )
    monkeypatch.setattr(platform, "list_dataset_versions", lambda dataset_id: next(listings))
    assert ctx.step(resolve_version(state, ctx)).version_id == "ds-1-v1"


def test_resolve_version_records_a_format_mismatch(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    ctx = make_ctx()
    state = upload(AuditState(), ctx)
    platform.uploads["ds-1"] = "chat-messages"
    step = ctx.step(resolve_version(state, ctx))
    assert step.version_id is None
    assert step.error == (
        "format_mismatch: platform sniffed 'chat-messages', the recipe needs 'labeled-example'"
    )


def test_split_renames_the_holdout_and_the_new_version_replaces_the_old(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    ctx = make_ctx()
    state = _through_split(AuditState(), ctx)
    step = ctx.step(state)
    assert platform.memberships == {"train": TRAIN, "validation": HOLDOUT}
    assert step.split_task_id == "split:ds-1-1"
    assert step.split_done is True
    assert step.version_id == "ds-1-v2"
    calls = len(platform.call_log)
    _through_split(state, ctx)
    assert len(platform.call_log) == calls


def test_split_rejection_is_recorded(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    platform.split_result = {"status": "rejected", "reason": "row 3 repeats across splits"}
    ctx = make_ctx()
    step = ctx.step(_through_split(AuditState(), ctx))
    assert step.split_done is None
    assert step.error == "split_rejected: row 3 repeats across splits"


def test_split_without_a_new_version_keeps_the_uploaded_one(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    platform.split_result = {"status": "completed"}
    ctx = make_ctx()
    step = ctx.step(_through_split(AuditState(), ctx))
    assert step.split_done is True
    assert step.version_id == "ds-1-v1"


def test_pii_scan_targets_the_split_version_and_agrees_when_clean(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    ctx = make_ctx()
    state = _through_split(AuditState(), ctx)
    state = wait_pii(pii_scan(state, ctx), ctx)
    step = ctx.step(state)
    assert platform.pii_version == "ds-1-v2"
    assert step.pii_task_id == "pii:ds-1-1"
    assert step.pii_agrees is True
    assert step.error is None
    calls = len(platform.call_log)
    wait_pii(pii_scan(state, ctx), ctx)
    assert len(platform.call_log) == calls


def test_pii_residual_finding_is_a_disagreement(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    platform.pii_counts = {"ds-1": {"PII_EMAIL": 2, "PII_PHONE": 0}}
    ctx = make_ctx()
    state = _through_split(AuditState(), ctx)
    step = ctx.step(wait_pii(pii_scan(state, ctx), ctx))
    assert step.pii_agrees is False
    assert step.error == (
        "pii_disagreement: server found {'PII_EMAIL': 2} after client redaction; "
        "classes the server did not scan: []"
    )


def test_pii_class_the_server_did_not_scan_is_a_disagreement(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    platform.pii_pass_list = ["PII_EMAIL"]
    ctx = make_ctx()
    state = _through_split(AuditState(), ctx)
    step = ctx.step(wait_pii(pii_scan(state, ctx), ctx))
    assert step.pii_agrees is False
    assert step.error is not None
    assert step.error.endswith("['PII_NATIONAL_ID', 'PII_PAYMENT_CARD', 'PII_PHONE']")


def test_pii_result_without_counts_or_pass_list_cannot_agree(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = make_ctx()
    state = pii_scan(_through_split(AuditState(), ctx), ctx)
    monkeypatch.setattr(
        platform,
        "get_dataset_task_status",
        lambda task_id: {
            "status": "SUCCESS",
            "result": {"counts_by_code": None, "pass_list": None},
        },
    )
    step = ctx.step(wait_pii(state, ctx))
    assert step.pii_agrees is False
    assert step.error is not None
    assert "server found {} after" in step.error
