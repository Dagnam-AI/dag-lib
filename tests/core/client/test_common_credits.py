"""The platform's insufficient-credits 402, in the bodies it really sends."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from dagnam._core.client import common
from dagnam._core.exceptions import InsufficientCreditsError, QuotaExceededError
from dagnam._types import JsonObject

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

    from dagnam._core.client import DagnamClient

RESTART = "https://api.test/api/v1/training/jobs/j1/restart"
SUPPORT = "https://dagnam.ai/support"
TRAINING_MESSAGE = (
    "You don't have enough credits to start this training run. It needs 120 credits to start "
    "and your balance is 30, so you're 90 credits short. "
    f"Ask us for more credits at {SUPPORT}, or make the run smaller (fewer epochs, a smaller "
    "model or a shorter time limit) and try again."
)
INFERENCE_MESSAGE = (
    "This request wasn't run because the account that owns this deployment is out of "
    "credits. You weren't charged. The account owner needs to add credits to keep it running."
)
TRAINING_BODY: JsonObject = {
    "error": "insufficient_credits",
    "message": TRAINING_MESSAGE,
    "user_message": TRAINING_MESSAGE,
    "operation": "training_run",
    "required_credits": 120,
    "available_credits": 30,
    "shortfall_credits": 90,
    "next_steps": ["request_credits", "reduce_job_size"],
    "detail": {
        "error": "insufficient_credits",
        "user_message": TRAINING_MESSAGE,
        "required_credits": 120,
        "available_credits": 30,
    },
}
INFERENCE_BODY: JsonObject = {
    "error": "insufficient_credits",
    "message": INFERENCE_MESSAGE,
    "user_message": INFERENCE_MESSAGE,
    "operation": "prediction",
    "required_credits": 1,
    "next_steps": ["request_credits"],
    "detail": {
        "error": "insufficient_credits",
        "user_message": INFERENCE_MESSAGE,
        "required_credits": 1,
    },
}


class _Resp:
    status_code = 402
    ok = False

    def __init__(self, body: object) -> None:
        self.text = json.dumps(body)
        self.content = self.text.encode()
        self.headers = {"Content-Type": "application/json"}
        self._content = b""


def test_the_training_refusal_is_typed_and_prints_its_message_exactly_once() -> None:
    with pytest.raises(InsufficientCreditsError) as caught:
        common.raise_for_generic(_Resp(TRAINING_BODY))
    exc = caught.value
    assert str(exc) == TRAINING_MESSAGE
    assert str(exc).count("You don't have enough credits") == 1
    assert "request_credits" not in str(exc)
    assert exc.message == TRAINING_MESSAGE
    assert (exc.required_credits, exc.available_credits) == (120, 30)
    assert exc.next_steps == ("request_credits", "reduce_job_size")


def test_the_inference_refusal_has_no_balance() -> None:
    with pytest.raises(InsufficientCreditsError) as caught:
        common.raise_for_generic(_Resp(INFERENCE_BODY))
    exc = caught.value
    assert (exc.required_credits, exc.available_credits) == (1, None)
    assert exc.next_steps == ("request_credits",)
    assert str(exc) == INFERENCE_MESSAGE


def test_it_is_a_quota_error_so_existing_handlers_still_catch_it() -> None:
    assert issubclass(InsufficientCreditsError, QuotaExceededError)
    with pytest.raises(QuotaExceededError):
        common.raise_for_generic(_Resp(TRAINING_BODY))


def test_every_mapper_types_it() -> None:
    for raiser in (common.raise_for_project, common.raise_for_codegen, common.raise_for_hub):
        with pytest.raises(InsufficientCreditsError):
            raiser(_Resp(TRAINING_BODY))


@pytest.mark.parametrize(
    "extra",
    [
        {"next_steps": "request_credits"},
        {"next_steps": None},
        {"next_steps": ["request_credits", 7, None]},
    ],
    ids=["a string", "absent", "mixed list"],
)
def test_malformed_next_steps_keep_only_the_string_tokens(extra: JsonObject) -> None:
    body = {**TRAINING_BODY, **extra}
    with pytest.raises(InsufficientCreditsError) as caught:
        common.raise_for_generic(_Resp(body))
    expected = ("request_credits",) if extra["next_steps"] == ["request_credits", 7, None] else ()
    assert caught.value.next_steps == expected


@pytest.mark.parametrize("available", ["30", True, None], ids=["string", "bool", "null"])
def test_a_balance_that_is_not_a_number_is_absent(available: object) -> None:
    body = {**TRAINING_BODY, "available_credits": available}
    with pytest.raises(InsufficientCreditsError) as caught:
        common.raise_for_generic(_Resp(body))
    assert caught.value.available_credits is None


@pytest.mark.parametrize("required", ["120", True, None], ids=["string", "bool", "null"])
def test_a_refusal_without_a_numeric_requirement_is_the_old_quota_error(required: object) -> None:
    body = {**TRAINING_BODY, "required_credits": required}
    with pytest.raises(QuotaExceededError) as caught:
        common.raise_for_generic(_Resp(body))
    assert not isinstance(caught.value, InsufficientCreditsError)


def test_a_refusal_without_a_message_still_says_something() -> None:
    body = {**TRAINING_BODY, "message": ""}
    with pytest.raises(InsufficientCreditsError) as caught:
        common.raise_for_generic(_Resp(body))
    assert str(caught.value) == "You don't have enough credits for this."


def test_the_plan_limit_shape_is_unchanged() -> None:
    body = {
        "error": "limit_exceeded",
        "message": "You have reached the Pro plan limit for projects.count.",
        "remediation_hints": ["Upgrade your plan"],
    }
    with pytest.raises(QuotaExceededError) as caught:
        common.raise_for_generic(_Resp(body))
    assert not isinstance(caught.value, InsufficientCreditsError)
    assert str(caught.value) == (
        "You have reached the Pro plan limit for projects.count. (Upgrade your plan)"
    )


def test_a_402_whose_marker_is_another_error_stays_a_plan_limit() -> None:
    with pytest.raises(QuotaExceededError) as caught:
        common.raise_for_generic(_Resp({"error": "something_else", "message": "No."}))
    assert not isinstance(caught.value, InsufficientCreditsError)


def test_the_sync_client_raises_it(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(RESTART, status_code=402, json=TRAINING_BODY)
    with pytest.raises(InsufficientCreditsError) as caught:
        client.restart_training_job("j1")
    assert caught.value.required_credits == 120
