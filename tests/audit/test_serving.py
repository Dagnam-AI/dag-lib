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

    assert found == [{"id": "up", "name": "up", "status": "running", "last_request_at": None}]


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
