"""Wire-level coverage for the sync FoundationClientMixin.

The fine-tuning discovery + submit surface: curated bases, shipped recipes,
submitting a run, reading one back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import (
    APIError,
    AuthError,
    EvaluationRunNotFoundError,
    FoundationRunNotFoundError,
    ModelNotFoundError,
    ResponseError,
)
from dagnam._types import JsonObject

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

API = "https://api.test"

# Shaped like one entry of the served recipe feed. ``hyperparameters`` is a
# JSON Schema document with a real bound, and ``future_field`` is a key this
# SDK version has never heard of -- both are here so a pass-through failure
# (a re-declared model, a field allow-list) shows up as a diff.
RECIPE: dict[str, Any] = {
    "key": "qlora-sft-chat@1.0",
    "display_name": "QLoRA SFT (chat)",
    "compatible_families": ["llama", "qwen"],
    "hyperparameters": {
        "type": "object",
        "properties": {
            "max_seq_length": {"type": "integer", "minimum": 128, "maximum": 4096, "default": 1024},
            "learning_rate": {"type": "number", "exclusiveMinimum": 0.0, "default": 0.0002},
        },
        "required": [],
    },
    "future_field": {"nested": ["a recipe grew a key the SDK does not know"]},
}


# ------------------------------------------------------------------ catalog


def test_list_foundation_catalog(client: DagnamClient, rmock: RequestsMocker) -> None:
    entry = {
        "id": "b1",
        "family": "llama",
        "a_field_added_after_this_release": {"nested": [1, None, True]},
    }
    rmock.get(f"{API}/api/v1/foundation-catalog", json=[entry])
    assert client.list_foundation_catalog() == [entry]


def test_list_foundation_catalog_sends_default_page_and_limit(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    """The defaults travel on the wire, so a caller gets one bounded page."""
    rmock.get(f"{API}/api/v1/foundation-catalog", json=[])
    client.list_foundation_catalog()
    assert rmock.last_request.qs == {"page": ["1"], "limit": ["20"]}


def test_list_foundation_catalog_paginates(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/foundation-catalog", json=[])
    client.list_foundation_catalog(page=3, limit=100)
    assert rmock.last_request.qs == {"page": ["3"], "limit": ["100"]}


def test_list_foundation_catalog_non_array_body(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.get(f"{API}/api/v1/foundation-catalog", json={"detail": "nope"})
    with pytest.raises(TypeError):
        client.list_foundation_catalog()


def test_list_foundation_catalog_401(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/foundation-catalog", status_code=401, text="bad key")
    with pytest.raises(AuthError):
        client.list_foundation_catalog()


# ------------------------------------------------------------------ recipes


def test_list_training_recipes_returns_the_served_payload_unchanged(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    """The recipe's schema is the contract; the SDK is a pipe, not a filter.

    Deep equality against the served document -- not a spot-check of two
    keys -- is what fails if any layer re-declares a hyperparameter, drops a
    bound, or allow-lists the fields it knows.
    """
    rmock.get(f"{API}/api/v1/training/recipes", json=[RECIPE])
    assert client.list_training_recipes() == [RECIPE]


def test_list_training_recipes_500_stays_an_api_error(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.get(f"{API}/api/v1/training/recipes", status_code=500, text="boom")
    with pytest.raises(APIError) as exc_info:
        client.list_training_recipes()
    assert exc_info.value.status_code == 500


def test_list_training_recipes_non_json_body(client: DagnamClient, rmock: RequestsMocker) -> None:
    """A JSON route that answers HTML is a broken deployment, not an empty list."""
    rmock.get(f"{API}/api/v1/training/recipes", text="<html>gateway</html>")
    with pytest.raises(ResponseError):
        client.list_training_recipes()


# ------------------------------------------------------------------- submit


def test_create_foundation_run(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(
        f"{API}/api/v1/training/foundation-runs",
        json={"run_id": "r1", "training_job_id": "j1"},
        status_code=201,
    )
    # The payload carries a hyperparameter this SDK has never heard of: the
    # recipe's schema is the only place one is declared, so a filter anywhere
    # on the write path would make every recipe published after this release
    # partly unusable. Asserted against the body ON THE WIRE, not the argument.
    payload: JsonObject = {
        "project_id": "p1",
        "recipe_key": "qlora-sft-chat@1.0",
        "hyperparameters": {
            "max_seq_length": 2048,
            "a_field_added_after_this_release": {"nested": [1, None, True]},
        },
    }
    assert client.create_foundation_run(payload) == {"run_id": "r1", "training_job_id": "j1"}
    assert rmock.last_request.json() == payload


def test_create_foundation_run_sends_idempotency_key(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    """A submit that spends GPU credits must replay, not duplicate."""
    rmock.post(f"{API}/api/v1/training/foundation-runs", json={"run_id": "r1"}, status_code=201)
    client.create_foundation_run({"project_id": "p1"})
    assert rmock.last_request.headers.get("Idempotency-Key")


def test_create_foundation_run_404_is_not_a_missing_run(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    """No run exists yet at submit time, so a 404 names no run id.

    The server answers a uniform 404 for any id in the body that did not
    resolve. Raising ``FoundationRunNotFoundError`` here would invent a run
    that was never created and bury the server's message.
    """
    rmock.post(
        f"{API}/api/v1/training/foundation-runs", status_code=404, text="dataset version not found"
    )
    with pytest.raises(APIError) as exc_info:
        client.create_foundation_run({"project_id": "p1"})
    assert not isinstance(exc_info.value, FoundationRunNotFoundError)
    assert "dataset version not found" in str(exc_info.value)


def test_create_foundation_run_400_invalid_hyperparameters(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(
        f"{API}/api/v1/training/foundation-runs",
        status_code=400,
        text="max_seq_length: less than or equal to 4096",
    )
    with pytest.raises(APIError) as exc_info:
        client.create_foundation_run({"project_id": "p1"})
    assert exc_info.value.status_code == 400


# --------------------------------------------------------------- read a run


def test_get_foundation_run(client: DagnamClient, rmock: RequestsMocker) -> None:
    body = {
        "run_id": "r1",
        "status": "running",
        "resolved_hyperparameters": {
            "epochs": 3,
            "a_field_added_after_this_release": {"nested": [1, None, True]},
        },
    }
    rmock.get(f"{API}/api/v1/training/foundation-runs/r1", json=body)
    assert client.get_foundation_run("r1") == body


def test_get_foundation_run_quotes_the_id(client: DagnamClient, rmock: RequestsMocker) -> None:
    """A caller-supplied id is a path segment, never a path."""
    rmock.get(f"{API}/api/v1/training/foundation-runs/..%2Fjobs%2Fj1", json={"run_id": "x"})
    client.get_foundation_run("../jobs/j1")
    # requests_mock lower-cases percent-escapes when it records the path.
    assert rmock.last_request.path.lower() == "/api/v1/training/foundation-runs/..%2fjobs%2fj1"


def test_get_foundation_run_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    """Missing and not-yours are one answer; both surface as the typed error."""
    rmock.get(
        f"{API}/api/v1/training/foundation-runs/missing",
        status_code=404,
        text="foundation run not found",
    )
    with pytest.raises(FoundationRunNotFoundError) as exc_info:
        client.get_foundation_run("missing")
    assert exc_info.value.run_id == "missing"


def test_get_foundation_run_non_object_body(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/training/foundation-runs/r1", json=[1, 2])
    with pytest.raises(ResponseError):
        client.get_foundation_run("r1")


# ------------------------------------------------------------------ evaluate

# The frozen specification an evaluation submit/read returns. A field this
# SDK version has never heard of is included so an allow-list or a
# re-declared response model shows up as a diff, not a silent drop.
EVALUATION_RUN: dict[str, Any] = {
    "run_id": "e1",
    "training_job_id": "j1",
    "subject_version_id": "v1",
    "baseline_version_id": "v0",
    "dataset_version_id": "dv1",
    "dataset_split": "test",
    "scorer_keys": ["accuracy@1", "macro-f1@1"],
    "resolved_params": {"accuracy@1": {}, "macro-f1@1": {"average": "macro"}},
    "a_field_added_after_this_release": {"nested": [1, None, True]},
}


def test_create_evaluation(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(
        f"{API}/api/v1/training/evaluations",
        json=EVALUATION_RUN,
        status_code=201,
    )
    # An unrecognised key travels on the wire unchanged: this layer is a
    # pipe, not a schema -- asserted against the body ON THE WIRE, not the
    # argument, so a filter anywhere on the write path is a visible diff.
    payload: JsonObject = {
        "project_id": "p1",
        "subject_version_id": "v1",
        "dataset_version_id": "dv1",
        "dataset_split": "test",
        "scorer_keys": ["accuracy@1"],
        "a_field_added_after_this_release": {"nested": [1, None, True]},
    }
    assert client.create_evaluation(payload) == EVALUATION_RUN
    assert rmock.last_request.json() == payload


def test_create_evaluation_sends_idempotency_key(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    """A submit that starts dispatch must replay, not duplicate."""
    rmock.post(f"{API}/api/v1/training/evaluations", json={"run_id": "e1"}, status_code=201)
    client.create_evaluation({"project_id": "p1"})
    assert rmock.last_request.headers.get("Idempotency-Key")


def test_create_evaluation_404_is_not_a_missing_run(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    """No run exists yet at submit time, so a 404 names no run id.

    The server answers a uniform 404 for any id in the body that did not
    resolve. Raising ``EvaluationRunNotFoundError`` here would invent a run
    that was never created and bury the server's message.
    """
    rmock.post(
        f"{API}/api/v1/training/evaluations", status_code=404, text="dataset version not found"
    )
    with pytest.raises(APIError) as exc_info:
        client.create_evaluation({"project_id": "p1"})
    assert not isinstance(exc_info.value, EvaluationRunNotFoundError)
    assert "dataset version not found" in str(exc_info.value)


def test_create_evaluation_400_incompatible_scorer(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(
        f"{API}/api/v1/training/evaluations",
        status_code=400,
        text="scorer 'macro-f1@1' is not compatible with task contract 'chat-completions@1.0'",
    )
    with pytest.raises(APIError) as exc_info:
        client.create_evaluation({"project_id": "p1"})
    assert exc_info.value.status_code == 400


def test_get_evaluation(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/training/evaluations/e1", json=EVALUATION_RUN)
    assert client.get_evaluation("e1") == EVALUATION_RUN


def test_get_evaluation_quotes_the_id(client: DagnamClient, rmock: RequestsMocker) -> None:
    """A caller-supplied id is a path segment, never a path."""
    rmock.get(f"{API}/api/v1/training/evaluations/..%2Fjobs%2Fj1", json={"run_id": "x"})
    client.get_evaluation("../jobs/j1")
    assert rmock.last_request.path.lower() == "/api/v1/training/evaluations/..%2fjobs%2fj1"


def test_get_evaluation_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    """Missing and not-yours are one answer; both surface as the typed error."""
    rmock.get(
        f"{API}/api/v1/training/evaluations/missing",
        status_code=404,
        text="evaluation run not found",
    )
    with pytest.raises(EvaluationRunNotFoundError) as exc_info:
        client.get_evaluation("missing")
    assert exc_info.value.run_id == "missing"


def test_get_evaluation_non_object_body(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/training/evaluations/e1", json=[1, 2])
    with pytest.raises(ResponseError):
        client.get_evaluation("e1")


# --------------------------------------------------------- version results

# One version's recorded scorer results, unknown-key-augmented so a filtering
# layer between the wire and the caller shows up as a diff.
VERSION_EVALUATIONS: list[dict[str, Any]] = [
    {
        "id": "r1",
        "evaluation_run_id": "e1",
        "version_id": "v1",
        "scorer_key": "accuracy@1",
        "value": 0.87,
        "sample_count": 512,
        "a_field_added_after_this_release": "future",
    }
]


def test_list_version_evaluations(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/model-versions/v1/evaluations", json=VERSION_EVALUATIONS)
    assert client.list_version_evaluations("v1") == VERSION_EVALUATIONS


def test_list_version_evaluations_empty(client: DagnamClient, rmock: RequestsMocker) -> None:
    """No recorded evaluations is a valid, empty answer -- not an error."""
    rmock.get(f"{API}/api/v1/model-versions/v1/evaluations", json=[])
    assert client.list_version_evaluations("v1") == []


def test_list_version_evaluations_quotes_the_id(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.get(f"{API}/api/v1/model-versions/..%2Fmodels%2Fm1/evaluations", json=[])
    client.list_version_evaluations("../models/m1")
    assert rmock.last_request.path.lower() == "/api/v1/model-versions/..%2fmodels%2fm1/evaluations"


def test_list_version_evaluations_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    """Missing and not-yours are one answer; both surface as the typed error."""
    rmock.get(
        f"{API}/api/v1/model-versions/missing/evaluations",
        status_code=404,
        text="Model version not found",
    )
    with pytest.raises(ModelNotFoundError) as exc_info:
        client.list_version_evaluations("missing")
    assert exc_info.value.model_id == "missing"


def test_list_version_evaluations_non_array_body(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.get(f"{API}/api/v1/model-versions/v1/evaluations", json={"detail": "nope"})
    with pytest.raises(TypeError):
        client.list_version_evaluations("v1")
