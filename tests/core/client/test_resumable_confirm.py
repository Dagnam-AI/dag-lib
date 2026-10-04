"""A replayed create is read back before it is trusted, and an "in progress" answer is waited out.

The platform replays a create for 24 hours, and holds an "in progress" marker for up to
two minutes when the process that made the ask died. Both are what a rerun after Ctrl+C
meets: the orphan the first run left may have been deleted since, and the marker may
still be there when the same command is typed again.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
import requests
from tests.core.client._replays import API, LOST, RUN, RUNS, Replays, mount, new_client

from dagnam._core._resume import IN_PROGRESS_POLL, IN_PROGRESS_SECONDS
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import (
    APIError,
    DeploymentStateError,
    DeploymentValidationError,
)
from dagnam._types import JsonObject

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

BODY: JsonObject = {"title": "workload-audit-audit", "framework": "pytorch"}
Create = Callable[[DagnamClient], JsonObject]

CREATES: dict[str, tuple[str, str, Create]] = {
    "run": (RUNS, "run_id", lambda c: c.create_foundation_run(RUN)),
    "project": (
        f"{API}/api/v1/projects",
        "id",
        lambda c: c.create_project(BODY, resume_nonce="nonce-a"),
    ),
    "audit": (
        f"{API}/api/v1/audits",
        "id",
        lambda c: c.create_audit({"project_id": "p1"}, resume_nonce="nonce-a"),
    ),
    "deployment": (
        f"{API}/api/v1/deployments",
        "id",
        lambda c: c.create_deployment({"training_job_id": "job-1"}),
    ),
}


@pytest.fixture
def runs(requests_mock: RequestsMocker) -> Replays:
    return mount(requests_mock, RUNS, Replays())


@pytest.fixture(params=list(CREATES))
def create(request: pytest.FixtureRequest, requests_mock: RequestsMocker) -> tuple[Replays, Create]:
    route, made, call = CREATES[request.param]
    requests_mock.post(
        f"{API}/api/v1/deployments/made-1/rotate-key", json={"key_prefix": "dk", "api_key": "dk-2"}
    )
    requests_mock.post(
        f"{API}/api/v1/deployments/made-2/rotate-key", json={"key_prefix": "dk", "api_key": "dk-3"}
    )
    return mount(requests_mock, route, Replays(made)), call


def test_a_replay_naming_something_the_owner_deleted_is_created_afresh(
    create: tuple[Replays, Create],
) -> None:
    """The run was lost to Ctrl+C, the owner deleted the orphan, the rerun was told it still exists."""
    server, call = create
    server.outcomes = [LOST]
    with pytest.raises(KeyboardInterrupt):
        call(new_client())
    server.deleted.add("made-1")

    made = call(new_client())

    assert "made-2" in made.values()
    assert len(server.created) == 2
    assert server.reads == [
        "made-1"
    ]  # read back once; the fresh create is the platform's own answer
    assert len(set(server.keys)) == 2  # the first key replayed, the next key asked afresh


def test_a_replay_of_something_still_there_is_returned_as_before(
    create: tuple[Replays, Create],
) -> None:
    server, call = create
    server.outcomes = [LOST]
    with pytest.raises(KeyboardInterrupt):
        call(new_client())

    made = call(new_client())

    assert "made-1" in made.values()
    assert len(server.created) == 1
    assert server.reads == ["made-1"]


def test_a_create_that_is_not_a_replay_is_not_read_back(create: tuple[Replays, Create]) -> None:
    server, call = create
    call(new_client())
    assert server.reads == []


def test_a_read_that_fails_is_raised_not_trusted_around(create: tuple[Replays, Create]) -> None:
    """A platform that cannot be asked proves nothing: the replay is neither accepted nor repeated."""
    server, call = create
    server.outcomes = [LOST]
    with pytest.raises(KeyboardInterrupt):
        call(new_client())
    server.read_status = 500

    with pytest.raises(APIError) as exc:
        call(new_client())

    assert exc.value.status_code == 500
    assert len(server.created) == 1


def test_an_answer_that_names_nothing_to_read_is_trusted(requests_mock: RequestsMocker) -> None:
    server = mount(requests_mock, f"{API}/api/v1/audits", Replays("id"))
    server.outcomes = [LOST]
    with pytest.raises(KeyboardInterrupt):
        new_client().create_audit({"project_id": "p1"}, resume_nonce="n")
    requests_mock.post(
        f"{API}/api/v1/audits",
        json=lambda req, ctx: ctx.headers.update({"Idempotency-Replayed": "true"}) or {"no": "id"},
    )

    assert new_client().create_audit({"project_id": "p1"}, resume_nonce="n") == {"no": "id"}
    assert server.reads == []


@pytest.mark.parametrize("with_location", [False, True], ids=["resource_id", "location"])
@pytest.mark.parametrize("name", list(CREATES))
def test_a_replay_that_dropped_its_body_is_read_back_from_its_pointer(
    name: str, with_location: bool, requests_mock: RequestsMocker
) -> None:
    """Same key, the resource exists, and the answer is only the platform's marker pointing at it."""
    route, made, call = CREATES[name]
    requests_mock.post(
        f"{API}/api/v1/deployments/made-1/rotate-key", json={"key_prefix": "dk", "api_key": "dk-2"}
    )
    server = mount(requests_mock, route, Replays(made))
    server.outcomes = [LOST]
    with pytest.raises(KeyboardInterrupt):
        call(new_client())
    server.pointer_only, server.with_location = True, with_location

    answer = call(new_client())

    assert answer[made] == "made-1"  # the resource itself, not the marker that pointed at it
    assert "replayed_without_body" not in answer
    assert len(server.created) == 1
    # The confirm read and the read-back are the same resource, never another.
    assert set(server.reads) == {"made-1"}


@pytest.mark.parametrize("name", list(CREATES))
def test_a_body_less_replay_of_something_the_owner_deleted_steps_to_the_next_key(
    name: str, requests_mock: RequestsMocker
) -> None:
    """The pointer names a resource that is gone: the create is asked afresh, not failed."""
    route, made, call = CREATES[name]
    requests_mock.post(
        f"{API}/api/v1/deployments/made-2/rotate-key", json={"key_prefix": "dk", "api_key": "dk-3"}
    )
    server = mount(requests_mock, route, Replays(made))
    server.outcomes = [LOST]
    with pytest.raises(KeyboardInterrupt):
        call(new_client())
    server.pointer_only = True
    server.deleted.add("made-1")

    answer = call(new_client())

    assert "made-2" in answer.values()
    assert len(server.created) == 2
    assert len(set(server.keys)) == 2


def test_a_training_job_create_replayed_without_its_body_is_read_back(
    requests_mock: RequestsMocker,
) -> None:
    """The first ask timed out after the platform made the job; the retry's replay carries no body."""
    marker = {
        "resource_id": "job-9",
        "status": 201,
        "replayed_without_body": True,
        "detail": "This request already completed with status 201; its response body was not retained.",
    }
    requests_mock.post(
        f"{API}/api/v1/training/jobs",
        [
            {"exc": requests.exceptions.ReadTimeout("timed out")},
            {"json": marker, "status_code": 201, "headers": {"Idempotency-Replayed": "true"}},
        ],
    )
    requests_mock.get(f"{API}/api/v1/training/jobs/job-9", json={"id": "job-9", "status": "queued"})

    assert new_client(resume=False).create_training_job({"project_id": "p1"}) == {
        "id": "job-9",
        "status": "queued",
    }


class TestInProgress:
    """An immediate rerun meets the marker its dead predecessor left; it waits, then asks again."""

    def test_a_run_create_waits_the_marker_out_and_then_finds_its_own_answer(
        self, runs: Replays
    ) -> None:
        runs.in_progress = 9  # the first two sends (four requests each, and one more) hit it
        client = new_client()
        sleeps: list[float] = []
        client._sleep = sleeps.append

        run = client.create_foundation_run(RUN)

        assert run["run_id"] == "made-1"
        assert len(runs.created) == 1
        assert (
            len(set(runs.keys)) == 1
        )  # the same key all along: the platform decides, not a new ask
        assert (
            IN_PROGRESS_POLL in sleeps
        )  # and the wait was this client's, not only the short retries

    def test_the_machine_readable_marker_is_waited_on_as_well_as_the_text(
        self, runs: Replays
    ) -> None:
        runs.in_progress, runs.marked = 5, True
        client = new_client()
        sleeps: list[float] = []
        client._sleep = sleeps.append

        assert client.create_foundation_run(RUN)["run_id"] == "made-1"
        assert IN_PROGRESS_POLL in sleeps

    def test_a_conflict_that_is_not_the_idempotency_marker_is_not_waited_on(
        self, requests_mock: RequestsMocker
    ) -> None:
        requests_mock.post(f"{API}/api/v1/audits", status_code=409, json={"detail": "name taken"})
        client = new_client()
        sleeps: list[float] = []
        client._sleep = sleeps.append

        with pytest.raises(APIError):
            client.create_audit({"project_id": "p1"}, resume_nonce="n")

        assert IN_PROGRESS_POLL not in sleeps

    def test_a_deployment_create_waits_too_although_its_409_is_not_an_api_error(
        self, requests_mock: RequestsMocker
    ) -> None:
        deployments = mount(requests_mock, f"{API}/api/v1/deployments", Replays("id"))
        deployments.in_progress = 5
        client = new_client()
        sleeps: list[float] = []
        client._sleep = sleeps.append

        made = client.create_deployment({"training_job_id": "job-1"})

        assert made["id"] == "made-1"
        assert IN_PROGRESS_POLL in sleeps

    def test_a_marker_that_never_clears_is_an_error_after_the_platforms_own_limit(
        self, requests_mock: RequestsMocker
    ) -> None:
        deployments = mount(requests_mock, f"{API}/api/v1/deployments", Replays("id"))
        deployments.in_progress = 10_000
        client = new_client()
        sleeps: list[float] = []
        client._sleep = sleeps.append

        with pytest.raises(DeploymentStateError):
            client.create_deployment({"training_job_id": "job-1"})

        assert sum(sleeps) >= IN_PROGRESS_SECONDS
        assert len(deployments.created) == 0

    def test_a_refusal_that_is_not_a_conflict_is_never_waited_on(
        self, requests_mock: RequestsMocker
    ) -> None:
        deployments = mount(requests_mock, f"{API}/api/v1/deployments", Replays("id"))
        deployments.outcomes = [422, 422]
        client = new_client()
        sleeps: list[float] = []
        client._sleep = sleeps.append

        with pytest.raises(DeploymentValidationError):
            client.create_deployment({"training_job_id": "job-1"})

        assert sleeps == []
