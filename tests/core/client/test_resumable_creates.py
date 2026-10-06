"""Creates a re-run must not repeat: the key comes from the request, so asking again replays.

The platform caches a create's terminal answer under its ``Idempotency-Key``
for 24 hours and replays it, marked ``Idempotency-Replayed: true``, to the same
key and body. A random key per call makes that protect one call only: a process
that dies after the platform committed -- Ctrl+C on a slow submit -- asks again
under a new key and gets a second, paid, resource. :class:`Replays` is that
cache in memory, so these tests drive the real client against the real rule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from tests.core.client._replays import API, LOST, RUN, RUNS, Replays, mount, new_client

from dagnam._core._resume import RESUME_ATTEMPTS
from dagnam._core.exceptions import APIError, DeploymentValidationError, QuotaExceededError
from dagnam._types import JsonObject

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker


@pytest.fixture
def runs(requests_mock: RequestsMocker) -> Replays:
    return mount(requests_mock, RUNS, Replays())


def test_by_default_every_create_is_its_own(runs: Replays) -> None:
    """An SDK caller who submits the same run twice means two runs."""
    first = new_client(resume=False).create_foundation_run(RUN)
    second = new_client(resume=False).create_foundation_run(RUN)
    assert (first["run_id"], second["run_id"]) == ("made-1", "made-2")
    assert runs.keys[0] != runs.keys[1]


def test_a_create_whose_answer_was_lost_is_replayed_to_the_run_that_asks_again(
    runs: Replays,
) -> None:
    """Ctrl+C after the platform committed the run, then the same command again.

    The second process used to mint a new key, so the platform started -- and
    billed -- a second run, and nothing recorded the first.
    """
    runs.outcomes = [LOST]
    with pytest.raises(KeyboardInterrupt):
        new_client().create_foundation_run(RUN)

    run = new_client().create_foundation_run(RUN)

    assert run["run_id"] == "made-1"
    assert len(runs.created) == 1
    assert runs.keys[0] == runs.keys[1]


def test_a_different_request_is_a_different_create(runs: Replays) -> None:
    """A new dataset version is a new run: a deliberate second one is never swallowed."""
    first = new_client().create_foundation_run(RUN)
    second = new_client().create_foundation_run({**RUN, "dataset_version_id": "ds-2-v2"})
    assert (first["run_id"], second["run_id"]) == ("made-1", "made-2")
    assert runs.keys[0] != runs.keys[1]


def test_a_refusal_the_platform_replays_is_asked_again_afresh(runs: Replays) -> None:
    """A cached refusal answers an earlier ask: the cause may be fixed since (credits topped up).

    Deterministic keys alone would replay the 402 for the whole 24-hour window.
    """
    runs.outcomes = [402]
    with pytest.raises(QuotaExceededError):
        new_client().create_foundation_run(RUN)
    assert len(runs.keys) == 1  # a refusal heard first-hand is the answer: nothing is asked twice

    run = new_client().create_foundation_run(RUN)

    assert run["run_id"] == "made-1"
    assert runs.keys[1] == runs.keys[0]  # the replayed refusal ...
    assert runs.keys[2] != runs.keys[0]  # ... and the next key in the sequence, asked afresh


def test_a_create_lost_after_an_earlier_refusal_is_still_found(runs: Replays) -> None:
    runs.outcomes = [402, LOST]
    with pytest.raises(QuotaExceededError):
        new_client().create_foundation_run(RUN)
    with pytest.raises(KeyboardInterrupt):
        new_client().create_foundation_run(RUN)

    run = new_client().create_foundation_run(RUN)

    assert run["run_id"] == "made-1"
    assert len(runs.created) == 1


def test_a_transient_failure_keeps_its_key_because_the_platform_caches_nothing(
    runs: Replays,
) -> None:
    runs.outcomes = [503, 503]
    run = new_client().create_foundation_run(RUN)
    assert run["run_id"] == "made-1"
    assert len(set(runs.keys)) == 1


def test_a_create_refused_every_time_is_still_asked_afresh(runs: Replays) -> None:
    """The key sequence is bounded; past it a random key keeps the ask from being stuck."""
    runs.outcomes = [402] * RESUME_ATTEMPTS
    for _ in range(RESUME_ATTEMPTS):
        with pytest.raises(QuotaExceededError):
            new_client().create_foundation_run(RUN)

    run = new_client().create_foundation_run(RUN)

    assert run["run_id"] == "made-1"
    assert len(set(runs.keys)) == RESUME_ATTEMPTS + 1


def test_an_audit_and_a_deployment_are_resumed_the_same_way(requests_mock: RequestsMocker) -> None:
    audits = mount(requests_mock, f"{API}/api/v1/audits", Replays("id"))
    deployments = mount(requests_mock, f"{API}/api/v1/deployments", Replays("id"))
    requests_mock.post(
        f"{API}/api/v1/deployments/made-1/rotate-key", json={"key_prefix": "dk", "api_key": "dk-2"}
    )
    audits.outcomes, deployments.outcomes = [LOST], [LOST]
    for create in (
        lambda: new_client().create_audit({"project_id": "p1"}),
        lambda: new_client().create_deployment({"training_job_id": "job-1"}),
    ):
        with pytest.raises(KeyboardInterrupt):
            create()

    assert new_client().create_audit({"project_id": "p1"})["id"] == "made-1"
    deployment = new_client().create_deployment({"training_job_id": "job-1"})

    assert len(audits.created) == len(deployments.created) == 1
    # The replay never carries the one-time key, so the client rotates it into a usable one.
    assert (deployment["id"], deployment["api_key"]) == ("made-1", "dk-2")


def test_a_refused_deployment_is_typed_whether_or_not_it_was_replayed(
    requests_mock: RequestsMocker,
) -> None:
    deployments = Replays("id")
    requests_mock.post(f"{API}/api/v1/deployments", json=deployments)
    deployments.outcomes = [422, 422]
    for _ in range(2):
        with pytest.raises(DeploymentValidationError):
            new_client().create_deployment({"training_job_id": "job-1"})
    assert len(deployments.keys) == 3  # asked, then the replay and a fresh ask


def test_a_project_is_never_keyed_by_its_content(requests_mock: RequestsMocker) -> None:
    """Its body is a title: the same for every audit directory of that name, on any machine."""
    projects = Replays("id")
    requests_mock.post(f"{API}/api/v1/projects", json=projects)
    body: JsonObject = {"title": "workload-audit-audit", "framework": "pytorch"}
    assert new_client().create_project(body)["id"] == "made-1"
    assert new_client().create_project(body)["id"] == "made-2"


def test_a_project_with_a_nonce_of_its_own_replays_for_that_nonce_alone(
    requests_mock: RequestsMocker,
) -> None:
    """The nonce is what only this caller holds, so the title can be keyed through it."""
    projects = mount(requests_mock, f"{API}/api/v1/projects", Replays("id"))
    body: JsonObject = {"title": "workload-audit-audit", "framework": "pytorch"}
    projects.outcomes = [LOST]
    with pytest.raises(KeyboardInterrupt):  # the process died after the platform committed
        new_client().create_project(body, resume_nonce="nonce-a")
    assert (
        new_client().create_project(body, resume_nonce="nonce-a")["id"] == "made-1"
    )  # its own, replayed
    assert (
        new_client().create_project(body, resume_nonce="nonce-b")["id"] == "made-2"
    )  # another audit's
    assert len(projects.created) == 2
    assert len(set(projects.keys)) == 2  # one key per nonce: the replay reused the first


def test_a_nonce_changes_nothing_on_a_client_that_does_not_resume(
    requests_mock: RequestsMocker,
) -> None:
    projects = Replays("id")
    requests_mock.post(f"{API}/api/v1/projects", json=projects)
    body: JsonObject = {"title": "t"}
    new_client(resume=False).create_project(body, resume_nonce="nonce-a")
    new_client(resume=False).create_project(body, resume_nonce="nonce-a")
    assert len(projects.created) == 2


def test_a_caller_that_names_its_own_key_keeps_it(requests_mock: RequestsMocker) -> None:
    deployments = Replays("id")
    requests_mock.post(f"{API}/api/v1/deployments/from-model-version", json=deployments)
    new_client().deploy_model_version("mv-1", idempotency_key="mine")
    assert deployments.keys == ["mine"]


def test_a_failure_with_no_answer_at_all_is_not_a_replay(requests_mock: RequestsMocker) -> None:
    requests_mock.post(RUNS, status_code=404, json={"detail": "Project not found"})
    with pytest.raises(APIError) as exc:
        new_client().create_foundation_run(RUN)
    assert exc.value.status_code == 404
    assert requests_mock.call_count == 1
