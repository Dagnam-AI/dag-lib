"""``audit delete`` while an endpoint is serving: all or nothing, and nothing local is touched."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.audit import _receipts as r
from tests.audit._cleanup import FakeCleanup, as_cleanup_client
from tests.audit._recorded import HEAD, recorded_state

from dagnam._core.exceptions import APIError, EndpointsServingError, TeardownInProgressError
from dagnam._types import JsonObject
from dagnam.audit.cleanup import DELETED_FILE, cancel_audit, delete_audit
from dagnam.audit.secrets import SecretStore
from dagnam.audit.state import load_state, save_state

SERVING: list[JsonObject] = [
    {
        "id": "dep-1",
        "name": "tickets-ft-small",
        "status": "running",
        "last_request_at": "2026-10-07T14:03:11Z",
    }
]


@pytest.fixture
def platform() -> FakeCleanup:
    return FakeCleanup(
        deployment=["dep-1", "dep-2"],
        model=["mv-1", "mv-2"],
        job=["job-1", "job-2"],
        dataset=["ds-1", "ds-2"],
        project=["proj-1"],
    )


def _prepare(audit_dir: Path, *, audit_id: str | None, unclaimed: list[str] | None = None) -> Path:
    state = recorded_state()
    state.audit_id = audit_id
    state.tagged = audit_id is not None and unclaimed is None
    state.unclaimed_ids = unclaimed or []
    save_state(audit_dir, state)
    SecretStore(audit_dir).store("w1/head_tune", "dk-secret")
    return audit_dir


def _snapshot(audit_dir: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(audit_dir)): p.read_bytes() for p in audit_dir.rglob("*") if p.is_file()
    }


def _never_sleep(_: float) -> None:
    raise AssertionError("a refusal is not waited on")


class TestThePlatformRefuses:
    """``409 endpoints_serving``: one ask, nothing deleted, nothing local touched."""

    def test_the_refusal_is_raised_after_exactly_one_ask_and_changes_nothing(
        self, audit_dir: Path, platform: FakeCleanup
    ) -> None:
        _prepare(audit_dir, audit_id="audit-1")
        platform.serving_refusal = EndpointsServingError("Nothing was deleted: ...", SERVING)
        before = _snapshot(audit_dir)
        present = {kind: set(ids) for kind, ids in platform.present.items()}

        with pytest.raises(EndpointsServingError) as exc:
            delete_audit(audit_dir, as_cleanup_client(platform), sleep=_never_sleep)

        assert exc.value.endpoints == SERVING
        assert platform.call_log == [("delete_audit", "audit-1")]
        assert platform.include_endpoints == [False]
        assert _snapshot(audit_dir) == before  # state.json, rows, keys: byte for byte
        assert not (audit_dir / DELETED_FILE).exists()
        assert SecretStore(audit_dir).load("w1/head_tune") == "dk-secret"
        assert load_state(audit_dir).halted is None
        assert platform.present == present

    def test_a_second_ask_is_made_only_by_the_person_and_is_refused_the_same_way(
        self, audit_dir: Path, platform: FakeCleanup
    ) -> None:
        """Nothing in the client retries or deletes on its own after a refusal: no loop to strand."""
        _prepare(audit_dir, audit_id="audit-1")
        platform.serving_refusal = EndpointsServingError("Nothing was deleted: ...", SERVING)

        for _ in range(2):
            with pytest.raises(EndpointsServingError):
                delete_audit(audit_dir, as_cleanup_client(platform), sleep=_never_sleep)

        assert platform.call_log == [("delete_audit", "audit-1")] * 2

    def test_the_override_is_sent_and_the_platforms_receipt_is_taken_as_it_comes(
        self, audit_dir: Path, platform: FakeCleanup
    ) -> None:
        _prepare(audit_dir, audit_id="audit-1")
        platform.serving_refusal = EndpointsServingError("Nothing was deleted: ...", SERVING)
        was_serving = {**r.row("deployment", "dep-1", "deleted"), "was_serving": True, "later": 1}
        platform.server_receipt = r.designed(
            was_serving,
            r.row("deployment", "dep-2", "deleted"),
            r.row("project", "proj-1", "deleted"),
        )

        receipt = delete_audit(audit_dir, as_cleanup_client(platform), include_endpoints=True)

        assert platform.include_endpoints == [True]
        written = json.loads((audit_dir / DELETED_FILE).read_text(encoding="utf-8"))
        assert written["entries"][0] == was_serving  # the field it sent and the one it will add
        assert receipt["entries"][0]["was_serving"] is True
        assert load_state(audit_dir).halted == {"reason": "deleted"}

    def test_a_busy_teardown_is_still_waited_on_and_not_mistaken_for_a_refusal(
        self, audit_dir: Path, platform: FakeCleanup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _prepare(audit_dir, audit_id="audit-1")
        calls: list[str] = []
        real = platform.delete_audit

        def busy_then_real(audit_id: str, *, include_endpoints: bool = False) -> JsonObject:
            calls.append(audit_id)
            if len(calls) == 1:
                raise TeardownInProgressError(409, "another walk", retry_after_header="1")
            return real(audit_id, include_endpoints=include_endpoints)

        monkeypatch.setattr(platform, "delete_audit", busy_then_real)
        slept: list[float] = []

        delete_audit(audit_dir, as_cleanup_client(platform), sleep=slept.append)

        assert (calls, slept) == (["audit-1", "audit-1"], [1.0])


class TestTheClientsOwnDirectWalk:
    """The ids this client deletes itself are read before the first destructive call."""

    def test_an_unpublished_audit_with_a_serving_endpoint_deletes_nothing(
        self, audit_dir: Path, platform: FakeCleanup
    ) -> None:
        _prepare(audit_dir, audit_id=None)
        platform.statuses = {"dep-1": "running", "dep-2": "paused"}
        before = _snapshot(audit_dir)

        with pytest.raises(EndpointsServingError) as exc:
            delete_audit(audit_dir, as_cleanup_client(platform))

        assert [e["id"] for e in exc.value.endpoints] == ["dep-1"]
        assert {name for name, _ in platform.call_log} == {
            "get_deployment"
        }  # not one destructive call
        assert _snapshot(audit_dir) == before
        assert not (audit_dir / DELETED_FILE).exists()
        assert all(platform.present[kind] for kind in ("deployment", "model", "job", "dataset"))

    def test_a_rolling_out_endpoint_counts_because_a_read_cannot_tell_if_it_ever_served(
        self, audit_dir: Path, platform: FakeCleanup
    ) -> None:
        _prepare(audit_dir, audit_id=None)
        platform.statuses = {"dep-2": "deploying"}

        with pytest.raises(EndpointsServingError) as exc:
            delete_audit(audit_dir, as_cleanup_client(platform))

        assert [(e["id"], e["status"]) for e in exc.value.endpoints] == [("dep-2", "deploying")]

    def test_the_override_deletes_them_without_reading_them(
        self, audit_dir: Path, platform: FakeCleanup
    ) -> None:
        _prepare(audit_dir, audit_id=None)
        platform.statuses = {"dep-1": "running"}

        receipt = delete_audit(audit_dir, as_cleanup_client(platform), include_endpoints=True)

        assert receipt["audit_status"] == "deleted"
        assert platform.present["deployment"] == set()
        assert ("get_deployment", "dep-1") in platform.call_log  # only the walk's own re-read

    def test_paused_and_gone_endpoints_do_not_block(
        self, audit_dir: Path, platform: FakeCleanup
    ) -> None:
        _prepare(audit_dir, audit_id=None)
        platform.present["deployment"].discard("dep-2")

        receipt = delete_audit(audit_dir, as_cleanup_client(platform))

        assert receipt["audit_status"] == "deleted"

    def test_a_published_audit_checks_what_the_platform_refused_to_claim_before_it_asks(
        self, audit_dir: Path, platform: FakeCleanup
    ) -> None:
        _prepare(audit_dir, audit_id="audit-1", unclaimed=["dep-1", "ds-1"])
        platform.statuses = {"dep-1": "running"}
        before = _snapshot(audit_dir)

        with pytest.raises(EndpointsServingError) as exc:
            delete_audit(audit_dir, as_cleanup_client(platform))

        assert [e["id"] for e in exc.value.endpoints] == ["dep-1"]
        assert platform.call_log == [("get_deployment", "dep-1")]  # the platform was never asked
        assert _snapshot(audit_dir) == before

    def test_what_the_platform_owns_is_left_to_the_platforms_own_check(
        self, audit_dir: Path, platform: FakeCleanup
    ) -> None:
        """A tagged audit has no unclaimed ids: its endpoints are not read here."""
        _prepare(audit_dir, audit_id="audit-1")
        platform.statuses = {"dep-1": "running", "dep-2": "running"}
        platform.server_receipt = r.designed(r.row("project", "proj-1", "deleted"))

        delete_audit(audit_dir, as_cleanup_client(platform))

        assert platform.call_log == [("delete_audit", "audit-1")]

    def test_the_override_reaches_the_platform_and_the_direct_walk(
        self, audit_dir: Path, platform: FakeCleanup
    ) -> None:
        _prepare(audit_dir, audit_id="audit-1", unclaimed=["dep-1"])
        platform.statuses = {"dep-1": "running"}
        platform.server_receipt = r.designed(
            r.row("deployment", "dep-1", "kept", "not_created_here"),
            r.row("project", "proj-1", "deleted"),
        )

        delete_audit(audit_dir, as_cleanup_client(platform), include_endpoints=True)

        assert platform.include_endpoints == [True]
        assert ("delete_deployment", "dep-1") in platform.call_log
        assert HEAD in load_state(audit_dir).workloads["w1"]


class TestTheRemedyClearsTheRefusal:
    """``audit cancel`` reads the endpoints' live status, not the one it saved after an earlier cancel."""

    def test_an_endpoint_the_owner_resumed_after_a_cancel_is_paused_by_the_next_cancel(
        self, audit_dir: Path, platform: FakeCleanup
    ) -> None:
        _prepare(audit_dir, audit_id=None)
        platform.statuses = {"dep-1": "running", "dep-2": "running"}
        cancel_audit(audit_dir, as_cleanup_client(platform))
        assert load_state(audit_dir).workloads["w1"][HEAD].deploy_status == "paused"
        platform.statuses["dep-1"] = "running"  # the owner resumed it: production calls it again
        with pytest.raises(EndpointsServingError):
            delete_audit(audit_dir, as_cleanup_client(platform))
        platform.call_log.clear()

        cancel_audit(audit_dir, as_cleanup_client(platform))

        assert ("pause_deployment", "dep-1") in platform.call_log
        assert ("pause_deployment", "dep-2") not in platform.call_log  # it reads paused: left alone
        assert delete_audit(audit_dir, as_cleanup_client(platform))["audit_status"] == "deleted"

    def test_the_same_for_an_endpoint_the_platform_refused_to_claim(
        self, audit_dir: Path, platform: FakeCleanup
    ) -> None:
        _prepare(audit_dir, audit_id="audit-1", unclaimed=["dep-1"])
        platform.statuses = {"dep-1": "running"}
        platform.server_receipt = r.designed(status="halted", schema=r.SCHEMA_CANCELLED)
        cancel_audit(audit_dir, as_cleanup_client(platform))
        platform.statuses["dep-1"] = "running"  # resumed after that cancel
        platform.call_log.clear()

        cancel_audit(audit_dir, as_cleanup_client(platform))

        assert ("pause_deployment", "dep-1") in platform.call_log
        assert ("pause_deployment", "dep-2") not in platform.call_log

    def test_an_endpoint_that_cannot_be_read_is_tried_for_a_pause_and_one_that_is_gone_is_not(
        self, audit_dir: Path, platform: FakeCleanup
    ) -> None:
        _prepare(audit_dir, audit_id=None)
        cancel_audit(audit_dir, as_cleanup_client(platform))  # both read paused: nothing to stop
        platform.call_log.clear()
        platform.unreadable = {"dep-1": APIError(503, "unavailable")}
        platform.present["deployment"].discard("dep-2")

        cancel_audit(audit_dir, as_cleanup_client(platform))

        assert ("pause_deployment", "dep-1") in platform.call_log
        assert ("pause_deployment", "dep-2") not in platform.call_log


class TestAnEndpointResumedWhileThePlatformWalked:
    """The pre-check ran before the platform's walk; the direct walk re-reads right before it deletes."""

    def _receipt(self, platform: FakeCleanup) -> None:
        platform.server_receipt = r.designed(
            r.row("deployment", "dep-1", "kept", "not_created_here"),
            r.row("project", "proj-1", "deleted"),
        )

    def test_one_that_serves_by_then_is_left_in_place_and_recorded_as_blocked(
        self, audit_dir: Path, platform: FakeCleanup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _prepare(audit_dir, audit_id="audit-1", unclaimed=["dep-1"])
        self._receipt(platform)
        real = platform.delete_audit

        def resumed_meanwhile(audit_id: str, *, include_endpoints: bool = False) -> JsonObject:
            platform.statuses["dep-1"] = "running"
            return real(audit_id, include_endpoints=include_endpoints)

        monkeypatch.setattr(platform, "delete_audit", resumed_meanwhile)

        receipt = delete_audit(audit_dir, as_cleanup_client(platform))

        assert ("delete_deployment", "dep-1") not in platform.call_log
        assert "dep-1" in platform.present["deployment"]
        row = next(e for e in receipt["entries"] if e["id"] == "dep-1" and e["status"] == "blocked")
        assert "serving" in row["reason"]
        assert "dep-1" in load_state(audit_dir).kept_ids  # never walked again
        assert SecretStore(audit_dir).load("w1/head_tune") == "dk-secret"  # the key that calls it

    def test_the_override_deletes_it_anyway(self, audit_dir: Path, platform: FakeCleanup) -> None:
        _prepare(audit_dir, audit_id="audit-1", unclaimed=["dep-1"])
        platform.statuses["dep-1"] = "running"
        self._receipt(platform)

        delete_audit(audit_dir, as_cleanup_client(platform), include_endpoints=True)

        assert ("delete_deployment", "dep-1") in platform.call_log

    def test_one_that_cannot_be_read_by_then_is_left_in_place_too(
        self, audit_dir: Path, platform: FakeCleanup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _prepare(audit_dir, audit_id="audit-1", unclaimed=["dep-1"])
        self._receipt(platform)
        real = platform.delete_audit

        def unreadable_meanwhile(audit_id: str, *, include_endpoints: bool = False) -> JsonObject:
            platform.unreadable = {"dep-1": APIError(503, "unavailable")}
            return real(audit_id, include_endpoints=include_endpoints)

        monkeypatch.setattr(platform, "delete_audit", unreadable_meanwhile)

        delete_audit(audit_dir, as_cleanup_client(platform))

        assert ("delete_deployment", "dep-1") not in platform.call_log
