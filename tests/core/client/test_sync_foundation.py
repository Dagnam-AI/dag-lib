"""Wire-level coverage for the sync FoundationClientMixin.

The fine-tuning discovery + submit surface: curated bases, shipped recipes,
submitting a run, reading one back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError, AuthError, FoundationRunNotFoundError, ResponseError
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
