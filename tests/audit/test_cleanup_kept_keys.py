"""The key of an endpoint the platform kept survives every route to ``forget_locally``.

The ids the platform kept are recorded from a cancel as well as a delete, saved BEFORE the first
local file or key is removed, and read again by every later delete: a delete that crashed half
way, a replay the platform answers with a 404, ``--already-deleted``. Each case runs against the
file backend and a fake keyring; the secrets are fakes.
"""

from __future__ import annotations

from pathlib import Path
import sys

import keyring
import pytest
from tests.audit import _receipts as r
from tests.audit._cleanup import FakeCleanup, as_cleanup_client
from tests.audit._recorded import HEAD, SFT, recorded_state

from dagnam._core.exceptions import APIError
from dagnam._types import JsonObject
from dagnam.audit.cleanup import cancel_audit, delete_audit
from dagnam.audit.secrets import SecretStore
from dagnam.audit.state import load_state, save_state

KEPT_KEY, OTHER_KEY = "w1/head_tune", "w2/sft_small"


@pytest.fixture(params=["file", "keyring"])
def backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """The secret store's backend: the 0600 file, or a fake in-memory keyring."""
    if request.param == "keyring":
        vault: dict[tuple[str, str], str] = {}
        monkeypatch.setitem(sys.modules, "keyring", keyring)
        monkeypatch.setattr(keyring, "set_password", lambda s, u, p: vault.__setitem__((s, u), p))
        monkeypatch.setattr(keyring, "get_password", lambda s, u: vault.get((s, u)))
        monkeypatch.setattr(keyring, "delete_password", lambda s, u: vault.pop((s, u), None))
    return str(request.param)


@pytest.fixture
def platform() -> FakeCleanup:
    return FakeCleanup(
        deployment=["dep-1", "dep-2"], model=["mv-1", "mv-2"], job=["job-1", "job-2"],
        dataset=["ds-1", "ds-2"], project=["proj-1"],
    )  # fmt: skip


@pytest.fixture
def prepared(audit_dir: Path, backend: str) -> Path:
    """A published audit whose two endpoints each have a stored key."""
    state = recorded_state()
    state.audit_id = "audit-1"
    state.workloads["w2"][SFT].key_ref = OTHER_KEY
    save_state(audit_dir, state)
    store = SecretStore(audit_dir)
    assert store.store(KEPT_KEY, "dk-kept") == backend
    store.store(OTHER_KEY, "dk-other")
    return audit_dir


def _cancelled(*rows: JsonObject) -> JsonObject:
    return r.designed(*rows, status="halted", schema=r.SCHEMA_CANCELLED)


def _deleted_around_the_kept_endpoint() -> JsonObject:
    return r.designed(
        r.row("training_job", "job-1", "deleted"),
        r.row("deployment", "dep-2", "deleted"),
        r.ADOPTED_ROW,
    )


def _keys(audit_dir: Path) -> tuple[str | None, str | None]:
    store = SecretStore(audit_dir)
    return store.load(KEPT_KEY), store.load(OTHER_KEY)


def test_a_cancel_records_what_the_platform_kept(prepared: Path, platform: FakeCleanup) -> None:
    platform.server_receipt = _cancelled(r.ADOPTED_ROW)

    cancel_audit(prepared, as_cleanup_client(platform))

    assert load_state(prepared).kept_ids == ["dep-1"]
    assert _keys(prepared) == ("dk-kept", "dk-other")  # a cancel removes no key


def test_cancel_then_a_delete_the_platform_cannot_see_keeps_the_key_of_the_kept_endpoint(
    prepared: Path, platform: FakeCleanup
) -> None:
    """``--already-deleted`` after a cancel: the only record of the keep is the cancel's."""
    platform.server_receipt = _cancelled(r.ADOPTED_ROW)
    cancel_audit(prepared, as_cleanup_client(platform))
    platform.account_error = APIError(404, "not found")

    delete_audit(prepared, as_cleanup_client(platform), assume_gone=True)

    assert _keys(prepared) == ("dk-kept", None)
    assert load_state(prepared).halted == {"reason": "deleted"}


def test_the_kept_ids_are_saved_before_the_first_thing_is_removed_locally(
    prepared: Path, platform: FakeCleanup, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A delete that dies in the middle of its local removal, then is replayed with a 404."""
    platform.server_receipt = _deleted_around_the_kept_endpoint()

    def crash(audit_dir: Path) -> None:
        raise RuntimeError("the machine went down")

    with monkeypatch.context() as crashing:
        crashing.setattr("dagnam.audit.cleanup_local._remove_workloads", crash)
        with pytest.raises(RuntimeError, match="went down"):
            delete_audit(prepared, as_cleanup_client(platform))
    assert load_state(prepared).kept_ids == ["dep-1"]  # on disk, not only in memory
    assert load_state(prepared).halted is None

    platform.account_error = APIError(404, "not found")  # the platform has nothing to say now
    delete_audit(prepared, as_cleanup_client(platform), assume_gone=True)

    assert _keys(prepared)[0] == "dk-kept"
    assert load_state(prepared).halted == {"reason": "deleted"}


def test_a_halted_delete_then_a_replayed_one_keeps_only_the_kept_endpoints_key(
    prepared: Path, platform: FakeCleanup
) -> None:
    platform.server_receipt = r.designed(r.ADOPTED_ROW, r.REFUSED_ROW, status="halted")
    delete_audit(prepared, as_cleanup_client(platform))
    assert _keys(prepared) == ("dk-kept", "dk-other")  # a halted delete removes nothing

    platform.server_receipt = _deleted_around_the_kept_endpoint()
    delete_audit(prepared, as_cleanup_client(platform))

    assert _keys(prepared) == ("dk-kept", None)
    delete_audit(prepared, as_cleanup_client(platform))  # and a repeat of the finished delete
    assert _keys(prepared) == ("dk-kept", None)


def test_an_endpoint_the_owner_stopped_keeping_since_the_cancel_loses_its_key_and_its_record(
    prepared: Path, platform: FakeCleanup
) -> None:
    platform.server_receipt = _cancelled(r.ADOPTED_ROW)
    cancel_audit(prepared, as_cleanup_client(platform))
    platform.server_receipt = r.designed(r.row("deployment", "dep-1", "deleted"))

    delete_audit(prepared, as_cleanup_client(platform))

    assert load_state(prepared).kept_ids == []
    assert _keys(prepared) == (None, None)
    assert load_state(prepared).workloads["w1"][HEAD].deployment_id == "dep-1"
