"""Data steps: each skips itself once done, uploads exactly the scan's files, and checks the server's scan."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
import os
from pathlib import Path
from typing import Any

import pytest
from tests.audit._platform import FakePlatform
from tests.audit._records import T0, make_record
from tests.audit.conftest import HOLDOUT, TRAIN

from dagnam._core.exceptions import APIError
from dagnam._types import JsonObject
from dagnam.audit import build_dataset, write_workload
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
from dagnam.audit.steps_serve import holdout
from dagnam.audit.workspace import NotRegularFileError, UnsafeWorkloadsError


def _through_split(state: AuditState, ctx: StepContext) -> AuditState:
    for step in (upload, resolve_version, split, wait_split):
        state = step(state, ctx)
    return state


def test_a_tool_argument_secret_is_redacted_before_it_is_uploaded(
    make_ctx: Callable[..., StepContext],
    platform: FakePlatform,
    audit_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool-calling workload trains on tool arguments, so a `login` call's password was in the rows.

    The assignment pattern needed an unquoted `key:`/`key=`, and inside JSON the
    key and the value were redacted as separate strings -- `{"password": ...}`
    matched neither way and went up to the platform as it was.
    """
    call = {
        "function": {"name": "login", "arguments": '{"user": "ana", "password": "hunter2secret"}'}
    }
    records = [
        make_record(response="", tool_calls=(call,), ts=T0 + timedelta(minutes=i)) for i in range(5)
    ]
    dataset = build_dataset(records, structure_class="json_object", max_seq_length=2_048)
    write_workload(audit_dir, "w9", dataset.rows, dataset.split, dataset.stats)
    sent: list[str] = []
    real_upload = platform.upload_dataset

    def upload_and_keep(file_path: str | Path, **kwargs: Any) -> JsonObject:
        sent.append(Path(file_path).read_text(encoding="utf-8"))
        return real_upload(file_path, **kwargs)

    monkeypatch.setattr(platform, "upload_dataset", upload_and_keep)
    upload(AuditState(), make_ctx(workload_id="w9"))
    assert sent
    assert "hunter2secret" not in sent[0]
    assert "<SECRET>" in sent[0]
    assert dataset.stats["redact"]["counts"]["PII_SECRET"] == 5


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


def test_resolve_version_halts_when_processing_already_failed(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    """The platform made the upload's processing run terminal before the first version poll."""
    ctx = make_ctx()
    state = upload(AuditState(), ctx)
    platform.analysis = {"analysis_status": "failed", "analysis_error": "unsupported file"}
    step = ctx.step(resolve_version(state, ctx))
    assert step.version_id is None
    assert step.error == "upload_failed: unsupported file"
    assert "list_dataset_versions" not in platform.call_log


def test_resolve_version_halts_when_processing_fails_mid_wait(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two pending polls, then the dataset goes terminal; a row with no reason still names one."""
    platform.version_polls = 9
    ctx = make_ctx()
    state = upload(AuditState(), ctx)
    rows = iter(
        [
            {"analysis_status": "processing"},
            {"analysis_status": "processing"},
            {"analysis_status": "failed"},
        ]
    )
    monkeypatch.setattr(platform, "get_dataset", lambda dataset_id: next(rows))
    step = ctx.step(resolve_version(state, ctx))
    assert step.version_id is None
    assert step.error == "upload_failed: the platform gave no reason"
    assert platform.call_log.count("list_dataset_versions") == 2


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


def test_an_upload_the_platform_committed_and_the_client_never_heard_back_is_adopted(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    """Ctrl+C or a timeout after the multipart POST landed, then the same command again.

    The platform cannot replay a multipart create, so a rerun used to upload the customer's rows
    a second time and leave the first dataset under no record in ``state.json`` -- where ``audit
    delete`` never looks. The rerun changes its flags and still finds it: the key is the
    directory's nonce and the candidate, nothing a flag moves.
    """
    platform.lost_uploads = 1
    with pytest.raises(APIError):
        upload(AuditState(project_nonce="nonce-a"), make_ctx())
    assert platform.call_log == ["list_datasets", "upload_dataset"]

    ctx = make_ctx(floor=0.5, max_credits=999)
    step = ctx.step(upload(AuditState(project_nonce="nonce-a"), ctx))

    assert step.dataset_id == "ds-1"
    assert list(platform.uploads) == ["ds-1"]  # one dataset on the platform, and the state names it
    assert platform.call_log[2:] == ["list_datasets"]


def test_the_key_is_in_the_description_not_the_name(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    upload(AuditState(project_nonce="nonce-a"), make_ctx())
    name, description = platform.dataset_rows["ds-1"]
    assert name == "audit-w1-head_tune"
    assert (
        description == "workload audit w1/head_tune: derived, redacted rows [nonce-a/w1/head_tune]"
    )


@pytest.mark.parametrize(
    ("nonce", "workload"),
    [("nonce-b", "w1"), ("nonce-a", "w2")],
    ids=["another directory", "another candidate"],
)
def test_another_directorys_or_candidates_upload_is_never_adopted(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, nonce: str, workload: str
) -> None:
    upload(AuditState(project_nonce="nonce-a"), make_ctx())
    ctx = make_ctx(workload_id=workload, structure_class="enum_label")
    state = upload(AuditState(project_nonce=nonce), ctx)
    assert ctx.step(state).dataset_id == "ds-2"
    assert len(platform.uploads) == 2


def test_a_dataset_that_only_mentions_the_key_is_not_adopted(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    """The search is a substring match on the name and description; the bracketed key is the proof."""
    platform.dataset_rows["ds-9"] = ("audit-w1-head_tune", "see nonce-a/w1/head_tune for details")
    platform.dataset_rows["ds-8"] = ("somebody-elses", "workload audit [nonce-a/w1/head_tune]")
    ctx = make_ctx()
    assert ctx.step(upload(AuditState(project_nonce="nonce-a"), ctx)).dataset_id == "ds-1"


def test_the_newest_of_several_uploads_is_the_one_adopted(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    tag = "workload audit w1/head_tune: derived, redacted rows [nonce-a/w1/head_tune]"
    platform.dataset_rows["ds-old"] = ("audit-w1-head_tune", tag)
    platform.dataset_rows["ds-new"] = ("audit-w1-head_tune", tag)
    ctx = make_ctx()
    assert ctx.step(upload(AuditState(project_nonce="nonce-a"), ctx)).dataset_id == "ds-new"
    assert platform.uploads == {}  # nothing was uploaded


class TestPublished:
    """A published audit's upload is tagged by the platform at creation; the audit id is the key."""

    def test_the_upload_names_the_audit_and_carries_no_key_in_its_description(
        self, make_ctx: Callable[..., StepContext], platform: FakePlatform
    ) -> None:
        state = AuditState(project_nonce="nonce-a", audit_id="audit-1")

        step = make_ctx().step(upload(state, make_ctx()))

        assert step.dataset_id == "ds-1"
        assert platform.dataset_tags == {"ds-1": "audit-1"}
        assert (
            platform.dataset_rows["ds-1"][1]
            == "workload audit w1/head_tune: derived, redacted rows"
        )

    def test_a_lost_upload_is_adopted_by_the_audits_tag_and_the_name(
        self, make_ctx: Callable[..., StepContext], platform: FakePlatform
    ) -> None:
        platform.lost_uploads = 1
        with pytest.raises(APIError):
            upload(AuditState(audit_id="audit-1"), make_ctx())

        ctx = make_ctx(floor=0.5)
        step = ctx.step(upload(AuditState(audit_id="audit-1"), ctx))

        assert step.dataset_id == "ds-1"
        assert platform.call_log[3:] == ["get_audit", "list_datasets"]
        assert len(platform.uploads) == 1

    def test_another_audits_dataset_of_the_same_name_is_never_adopted(
        self, make_ctx: Callable[..., StepContext], platform: FakePlatform
    ) -> None:
        platform.dataset_rows["ds-9"] = ("audit-w1-head_tune", "")
        platform.dataset_tags["ds-9"] = "audit-2"
        ctx = make_ctx()
        assert ctx.step(upload(AuditState(audit_id="audit-1"), ctx)).dataset_id == "ds-1"


@pytest.mark.parametrize(
    "fresh",
    [lambda: AuditState(project_nonce="nonce-a"), lambda: AuditState(audit_id="audit-1")],
    ids=["unpublished", "published"],
)
def test_a_dataset_the_state_already_records_is_never_adopted_by_another_candidate(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, fresh: Callable[[], AuditState]
) -> None:
    """A forced rescan retires the old candidate (its id stays in ``state.retired``): its dataset
    holds the old rows, so the new candidate of the same name must upload its own."""
    first = make_ctx()
    state = fresh()
    first.step(upload(state, first))
    old = first.step(state)
    state.retired.append(old)
    state.workloads.clear()

    ctx = make_ctx()
    again = ctx.step(upload(state, ctx))

    assert (old.dataset_id, again.dataset_id) == ("ds-1", "ds-2")
    assert len(platform.uploads) == 2


@pytest.mark.skipif(os.name == "nt", reason="a named pipe needs POSIX")
def test_an_upload_file_that_is_not_a_regular_file_is_refused_before_it_is_opened(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    def must_not_be_reached(*_args: object, **_kwargs: object) -> JsonObject:
        raise AssertionError("the upload was reached: a pipe would have blocked it for ever")

    monkeypatch.setattr(platform, "upload_dataset", must_not_be_reached)
    ctx = make_ctx()
    target = ctx.workload_dir / "dataset.jsonl"
    target.unlink()
    os.mkfifo(target)  # a pipe would block the upload for ever, under the audit's lock

    with pytest.raises(NotRegularFileError):
        upload(AuditState(project_nonce="n"), ctx)

    assert "upload_dataset" not in platform.call_log


@pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege on Windows")
@pytest.mark.parametrize("name", ["split.json", "meta.json", "dataset.jsonl"])
def test_a_file_a_step_reads_back_from_the_audit_directory_is_never_read_through_a_link(
    make_ctx: Callable[..., StepContext], tmp_path: Path, name: str
) -> None:
    """Whatever a link points at (another user's file, a secret) must not be read into a step."""
    ctx = make_ctx()
    outside = tmp_path / "outside.txt"
    outside.write_text("{}", encoding="utf-8")
    target = ctx.workload_dir / name
    target.unlink()
    target.symlink_to(outside)

    readers: dict[str, Callable[[], object]] = {
        "split.json": lambda: split(AuditState(), ctx),
        "meta.json": ctx.meta,
        "dataset.jsonl": lambda: holdout(ctx),
    }
    with pytest.raises(UnsafeWorkloadsError):
        readers[name]()
