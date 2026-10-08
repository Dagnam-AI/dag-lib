"""The read-before-delete rule for the endpoints this client deletes on its own."""

from __future__ import annotations

import pytest
from tests.audit._cleanup import FakeCleanup, as_cleanup_client

from dagnam._core.exceptions import APIError, EndpointsServingError
from dagnam.audit.serving import refuse_if_serving, serving_here


def _platform(**statuses: str) -> FakeCleanup:
    fake = FakeCleanup(deployment=list(statuses))
    fake.statuses = dict(statuses)
    return fake


def test_a_running_or_rolling_out_endpoint_may_be_serving_and_the_rest_may_not() -> None:
    fake = _platform(
        up="running",
        rolling="deploying",
        paused="paused",
        stopped="stopped",
        failed="failed",
        fresh="not_provisioned",
    )

    found = serving_here(as_cleanup_client(fake), ["up", "rolling", "paused", "stopped", "failed"])

    assert [(e["id"], e["status"]) for e in found] == [("up", "running"), ("rolling", "deploying")]
    assert serving_here(as_cleanup_client(fake), ["fresh"]) == []


def test_an_endpoint_that_is_gone_is_not_serving_and_a_nameless_one_is_listed_by_its_id() -> None:
    fake = _platform(up="running")

    found = serving_here(as_cleanup_client(fake), ["gone", "up"])

    assert found == [{"id": "up", "name": "up", "status": "running"}]


def test_the_name_the_platform_gives_is_the_one_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _platform(up="running")
    real = fake.get_deployment
    monkeypatch.setattr(fake, "get_deployment", lambda i: {**real(i), "name": "tickets-ft-small"})

    assert serving_here(as_cleanup_client(fake), ["up"])[0]["name"] == "tickets-ft-small"


def test_a_serving_endpoint_refuses_with_nothing_but_reads_and_the_endpoints_it_names() -> None:
    fake = _platform(a="running", b="deploying", c="paused")

    with pytest.raises(EndpointsServingError) as exc:
        refuse_if_serving(as_cleanup_client(fake), ["a", "b", "c"], include=False)

    assert [e["id"] for e in exc.value.endpoints] == ["a", "b"]
    assert exc.value.status_code == 409
    assert exc.value.message.startswith("Nothing was deleted: 2 endpoints of this audit")
    assert "deleting them would break any app that calls them" in exc.value.message
    assert "include_endpoints=True" in exc.value.message
    assert {name for name, _ in fake.call_log} == {"get_deployment"}


def test_the_sentence_is_singular_for_one_endpoint() -> None:
    fake = _platform(a="running")

    with pytest.raises(EndpointsServingError) as exc:
        refuse_if_serving(as_cleanup_client(fake), ["a"], include=False)

    assert "1 endpoint of this audit may still be serving" in exc.value.message
    assert "deleting it would break any app that calls it" in exc.value.message


def test_nothing_serving_passes_and_the_override_does_not_even_read() -> None:
    fake = _platform(a="paused")
    refuse_if_serving(as_cleanup_client(fake), ["a"], include=False)
    assert fake.call_log == [("get_deployment", "a")]

    live = _platform(a="running")
    refuse_if_serving(as_cleanup_client(live), ["a"], include=True)
    assert live.call_log == []


def test_an_endpoint_that_cannot_be_read_is_not_guessed_at() -> None:
    fake = _platform(a="running")
    fake.unreadable = {"a": APIError(503, "unavailable")}

    with pytest.raises(APIError) as exc:
        refuse_if_serving(as_cleanup_client(fake), ["a"], include=False)

    assert not isinstance(exc.value, EndpointsServingError)


@pytest.mark.parametrize("status", ["paused", "stopped", "failed", "not_provisioned"])
def test_only_a_status_known_to_be_quiet_lets_an_endpoint_be_deleted(status: str) -> None:
    fake = _platform(a=status)
    refuse_if_serving(as_cleanup_client(fake), ["a"], include=False)


@pytest.mark.parametrize(
    "answer",
    [{"id": "a"}, {"id": "a", "status": None}, {"id": "a", "status": "updating"}, {"status": 7}],
    ids=["no status", "null status", "a status from a later platform", "not a string"],
)
def test_a_status_it_cannot_read_as_quiet_counts_as_serving(
    monkeypatch: pytest.MonkeyPatch, answer: dict[str, object]
) -> None:
    fake = _platform(a="paused")
    monkeypatch.setattr(fake, "get_deployment", lambda _: answer)

    with pytest.raises(EndpointsServingError) as exc:
        refuse_if_serving(as_cleanup_client(fake), ["a"], include=False)

    assert [e["id"] for e in exc.value.endpoints] == ["a"]


def test_the_sentence_names_the_remedy_that_works_for_what_is_listed() -> None:
    running = _platform(a="running")
    rolling = _platform(a="deploying")
    mixed = _platform(a="running", b="deploying")

    def said(fake: FakeCleanup, ids: list[str]) -> str:
        with pytest.raises(EndpointsServingError) as exc:
            refuse_if_serving(as_cleanup_client(fake), ids, include=False)
        return exc.value.message

    assert "Stop it first with `dagnam audit cancel`." in said(running, ["a"])
    assert "cannot be paused" not in said(running, ["a"])
    only_rolling = said(rolling, ["a"])
    assert "audit cancel" not in only_rolling  # it cannot clear a deploying endpoint
    assert "cannot be paused from here: wait for it to settle" in only_rolling
    assert "include_endpoints=True" in only_rolling
    both = said(mixed, ["a", "b"])
    assert "Stop the running ones first with `dagnam audit cancel`." in both
    assert "wait for it to settle" in both
