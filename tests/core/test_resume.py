"""The key of a resumable create, and the walk past the refusals the platform replays."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from dagnam._core._resume import (
    IN_PROGRESS_MARKER,
    IN_PROGRESS_POLL,
    IN_PROGRESS_SECONDS,
    RESUME_ATTEMPTS,
    confirmed_by,
    content_key,
    created_body,
    gone,
    idempotency_in_progress,
    replay_pointer,
    resume_create,
)
from dagnam._core.exceptions import APIError, DagnamError, ProjectNotFoundError

URL = "https://api.test/api/v1/training/foundation-runs"
BODY = {"project_id": "p1", "dataset_version_id": "ds-1-v2"}


def test_the_same_request_has_the_same_key_and_any_difference_another() -> None:
    key = content_key("POST", URL, BODY, 0)
    assert key == content_key("POST", URL, dict(BODY), 0)
    assert key.startswith("dagnam-")
    assert len(key) == len("dagnam-") + 64
    others = {
        content_key("POST", URL, BODY, 1),
        content_key("POST", URL, {**BODY, "dataset_version_id": "ds-2-v2"}, 0),
        content_key("POST", URL.replace("api.test", "api.other"), BODY, 0),
        content_key("PATCH", URL, BODY, 0),
    }
    assert key not in others
    assert len(others) == 4


def test_a_body_json_cannot_carry_is_refused_not_keyed_loosely() -> None:
    """``NaN`` is not JSON: the request would be refused too, so the key must not paper over it."""
    with pytest.raises(ValueError, match="Out of range float"):
        content_key("POST", URL, {"floor": float("nan")}, 0)


def _walk(
    send: Callable[[str], str],
    seen: Callable[[], tuple[bool, bool]] = lambda: (False, False),
    confirm: Callable[[str], bool] | None = None,
) -> tuple[str, list[float]]:
    sleeps: list[float] = []
    made = resume_create(send, seen, "POST", URL, BODY, sleep=sleeps.append, confirm=confirm)
    return made, sleeps


def test_the_first_answer_ends_the_walk() -> None:
    sent: list[str] = []

    def send(key: str) -> str:
        sent.append(key)
        return "made"

    assert _walk(send) == ("made", [])
    assert sent == [content_key("POST", URL, BODY, 0)]


def test_replayed_refusals_are_stepped_past_and_a_first_hand_one_is_raised() -> None:
    sent: list[str] = []
    replays = [True, True, False]

    def send(key: str) -> str:
        sent.append(key)
        raise APIError(402, "out of credits")

    with pytest.raises(APIError, match="out of credits"):
        _walk(send, lambda: (replays[len(sent) - 1], False))
    assert sent == [content_key("POST", URL, BODY, attempt) for attempt in range(3)]


def test_a_history_of_nothing_but_replayed_refusals_ends_in_a_random_key() -> None:
    sent: list[str] = []

    def send(key: str) -> str:
        sent.append(key)
        if len(sent) <= RESUME_ATTEMPTS:
            raise APIError(402, "out of credits")
        return "made"

    assert _walk(send, lambda: (True, False))[0] == "made"
    assert len(sent) == RESUME_ATTEMPTS + 1
    assert not sent[-1].startswith("dagnam-")


def test_a_replayed_success_the_read_back_refutes_is_stepped_past_like_a_refusal() -> None:
    sent: list[str] = []
    asked: list[str] = []

    def send(key: str) -> str:
        sent.append(key)
        return f"made-{len(sent)}"

    def confirm(made: str) -> bool:
        asked.append(made)
        return made != "made-1"

    assert _walk(send, lambda: (True, False), confirm)[0] == "made-2"
    assert sent == [content_key("POST", URL, BODY, attempt) for attempt in range(2)]
    assert asked == ["made-1", "made-2"]


def test_a_first_hand_answer_is_not_read_back() -> None:
    def confirm(_: str) -> bool:
        raise AssertionError("a create that really ran is the platform's own answer")

    assert _walk(lambda _key: "made", lambda: (False, False), confirm)[0] == "made"


def test_an_in_progress_answer_is_waited_out_under_the_same_key() -> None:
    sent: list[str] = []

    def send(key: str) -> str:
        sent.append(key)
        if len(sent) <= 3:
            raise APIError(409, "in progress")
        return "made"

    made, sleeps = _walk(send, lambda: (False, True))

    assert made == "made"
    assert sleeps == [IN_PROGRESS_POLL] * 3
    assert len(set(sent)) == 1


def test_an_in_progress_answer_that_never_clears_is_raised_after_the_markers_lifetime() -> None:
    def send(_key: str) -> str:
        raise APIError(409, "in progress")

    sleeps: list[float] = []
    with pytest.raises(APIError, match="in progress"):
        resume_create(send, lambda: (False, True), "POST", URL, BODY, sleep=sleeps.append)
    assert sum(sleeps) == IN_PROGRESS_SECONDS


def test_any_other_first_hand_failure_is_raised_at_once() -> None:
    def send(_key: str) -> str:
        raise APIError(500, "boom")

    sleeps: list[float] = []
    with pytest.raises(APIError, match="boom"):
        resume_create(send, lambda: (False, False), "POST", URL, BODY, sleep=sleeps.append)
    assert sleeps == []


class _Reply:
    def __init__(self, body: object) -> None:
        self.body = body
        self.headers: dict[str, str] = {}

    def json(self) -> object:
        return self.body


def _missing_read(known: set[str]) -> Callable[[str], object]:
    def read(item_id: str) -> object:
        if item_id not in known:
            raise ProjectNotFoundError(item_id)
        if item_id == "down":
            raise APIError(503, "down")
        return {}

    return read


@pytest.mark.parametrize(
    ("exc", "is_gone"),
    [
        (ProjectNotFoundError("p"), True),
        (APIError(404, "uniform not found"), True),
        (APIError(500, "boom"), False),
        (DagnamError("other"), False),
    ],
)
def test_a_read_says_gone_by_its_typed_error_or_a_plain_404(
    exc: DagnamError, is_gone: bool
) -> None:
    assert gone(ProjectNotFoundError)(exc) is is_gone


def test_a_confirm_is_true_for_what_is_there_false_for_what_is_gone_and_raises_the_rest() -> None:
    confirm = confirmed_by(_missing_read({"p1", "down"}), gone(ProjectNotFoundError))
    assert confirm(_Reply({"id": "p1"})) is True
    assert confirm(_Reply({"id": "p2"})) is False
    with pytest.raises(APIError, match="down"):
        confirm(_Reply({"id": "down"}))


def test_a_confirm_trusts_an_answer_that_names_nothing_to_read() -> None:
    confirm = confirmed_by(_missing_read(set()), gone(ProjectNotFoundError), id_key="run_id")
    assert confirm(_Reply({"id": "p1"})) is True  # no ``run_id``
    assert confirm(_Reply({"run_id": 7})) is True  # not an id
    assert confirm(_Reply(["not", "an", "object"])) is True


class _Response:
    def __init__(
        self,
        status_code: int = 200,
        body: object = None,
        headers: dict[str, str] | None = None,
        json_error: bool = False,
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}
        self.json_error = json_error

    def json(self) -> object:
        if self.json_error:
            raise ValueError("not json")
        return self.body


@pytest.mark.parametrize(
    ("response", "waits"),
    [
        # A platform that marks it.
        (_Response(409, {"error": IN_PROGRESS_MARKER, "detail": "anything at all"}), True),
        (_Response(409, {"error": IN_PROGRESS_MARKER}), True),
        # An older one, which sends the text alone: exactly this text.
        (_Response(409, {"detail": "A request with this Idempotency-Key is in progress"}), True),
        # Never a looser match: a real conflict, another text, another status, no body.
        (
            _Response(409, {"detail": "A request with this Idempotency-Key is in progress now"}),
            False,
        ),
        (_Response(409, {"detail": "a request with this idempotency-key is in progress"}), False),
        (_Response(409, {"detail": "Deployment name already exists"}), False),
        (_Response(409, {"error": "name_taken"}), False),
        (_Response(409, ["A request with this Idempotency-Key is in progress"]), False),
        (_Response(409, json_error=True), False),
        (_Response(422, {"error": IN_PROGRESS_MARKER}), False),
        (_Response(200, {"error": IN_PROGRESS_MARKER}), False),
    ],
)
def test_an_in_progress_answer_is_the_marker_or_exactly_the_platforms_text(
    response: _Response, waits: bool
) -> None:
    assert idempotency_in_progress(response) is waits


@pytest.mark.parametrize(
    ("headers", "body", "pointer"),
    [
        ({"Location": "/api/v1/audits/a-1"}, {}, "a-1"),
        ({"Location": "https://api.test/api/v1/audits/a-1/"}, {"detail": "replayed"}, "a-1"),
        ({}, {"resource_id": "a-2"}, "a-2"),
        ({"Location": "/api/v1/audits/a-1"}, {"resource_id": "a-2"}, "a-1"),
        ({}, {"resource_id": 7}, None),
        ({}, {}, None),
    ],
)
def test_a_replay_that_dropped_its_body_points_at_the_resource_by_location_or_id(
    headers: dict[str, str], body: dict[str, object], pointer: str | None
) -> None:
    assert replay_pointer(_Response(200, body, headers), body) == pointer


def test_a_pointer_only_replay_is_read_back_and_a_full_answer_is_not() -> None:
    read: list[str] = []

    def fetch(item_id: str) -> dict[str, object]:
        read.append(item_id)
        return {"id": item_id, "status": "running"}

    replayed = {"Idempotency-Replayed": "true"}
    pointer_only = _Response(200, {}, {**replayed, "Location": "/api/v1/audits/a-1"})
    assert created_body(pointer_only, {}, fetch) == {"id": "a-1", "status": "running"}
    full = _Response(200, {"id": "a-1"}, replayed)
    assert created_body(full, {"id": "a-1"}, fetch) == {"id": "a-1"}
    first_hand = _Response(200, {}, {"Location": "/api/v1/audits/a-9"})
    assert created_body(first_hand, {}, fetch) == {}
    assert created_body(_Response(200, {}, replayed), {}, fetch) == {}  # nothing to follow
    assert created_body(
        _Response(200, {}, {**replayed, "Location": "/r/r-1"}), {}, fetch, "run_id"
    ) == {
        "id": "r-1",
        "status": "running",
    }
    assert read == ["a-1", "r-1"]
