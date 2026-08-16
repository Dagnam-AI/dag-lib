"""Wire-level coverage for the async AsyncFoundationMixin.

Async mirror of ``tests/core/client/test_sync_foundation.py``. The routes,
the payload pass-through and the 404 mapping must match the sync surface, so
those assertions are deliberately identical; only the body-shape guard differs
(``ensure_json_object`` raises ``TypeError`` where the sync narrowing raises
``ResponseError``) -- a pre-existing split this module does not widen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
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
    from tests.typing_helpers import RespxMockRouter

API = "https://api.test"

pytestmark = pytest.mark.anyio

# Same document the sync mirror uses: a real JSON Schema bound plus a key this
# SDK version does not know, so any filtering shows up as a diff.
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


async def test_async_list_foundation_catalog(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    entry = {
        "id": "b1",
        "family": "llama",
        "a_field_added_after_this_release": {"nested": [1, None, True]},
    }
    mock.get("/api/v1/foundation-catalog").mock(return_value=httpx.Response(200, json=[entry]))
    assert await client.list_foundation_catalog() == [entry]


async def test_async_list_foundation_catalog_sends_default_page_and_limit(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.get("/api/v1/foundation-catalog").mock(return_value=httpx.Response(200, json=[]))
    await client.list_foundation_catalog()
    assert dict(route.calls[-1].request.url.params) == {"page": "1", "limit": "20"}


async def test_async_list_foundation_catalog_paginates(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.get("/api/v1/foundation-catalog").mock(return_value=httpx.Response(200, json=[]))
    await client.list_foundation_catalog(page=3, limit=100)
    assert dict(route.calls[-1].request.url.params) == {"page": "3", "limit": "100"}


async def test_async_list_foundation_catalog_non_array_body(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/foundation-catalog").mock(
        return_value=httpx.Response(200, json={"detail": "nope"})
    )
    with pytest.raises(TypeError):
        await client.list_foundation_catalog()


async def test_async_list_foundation_catalog_401(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/foundation-catalog").mock(return_value=httpx.Response(401, text="bad key"))
    with pytest.raises(AuthError):
        await client.list_foundation_catalog()


# ------------------------------------------------------------------ recipes


async def test_async_list_training_recipes_returns_the_served_payload_unchanged(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    """The async surface is a pipe too -- schema in, same schema out."""
    mock.get("/api/v1/training/recipes").mock(return_value=httpx.Response(200, json=[RECIPE]))
    assert await client.list_training_recipes() == [RECIPE]


async def test_async_list_training_recipes_500(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/training/recipes").mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(APIError) as exc_info:
        await client.list_training_recipes()
    assert exc_info.value.status_code == 500


async def test_async_list_training_recipes_non_json_body(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/training/recipes").mock(
        return_value=httpx.Response(200, text="<html>gateway</html>")
    )
    with pytest.raises(ResponseError):
        await client.list_training_recipes()


# ------------------------------------------------------------------- submit


async def test_async_create_foundation_run(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post("/api/v1/training/foundation-runs").mock(
        return_value=httpx.Response(201, json={"run_id": "r1", "training_job_id": "j1"})
    )
    # Same unknown-hyperparameter fixture as the sync mirror -- asserted
    # against the serialized request body, not the argument.
    payload: JsonObject = {
        "project_id": "p1",
        "recipe_key": "qlora-sft-chat@1.0",
        "hyperparameters": {
            "max_seq_length": 2048,
            "a_field_added_after_this_release": {"nested": [1, None, True]},
        },
    }
    assert await client.create_foundation_run(payload) == {
        "run_id": "r1",
        "training_job_id": "j1",
    }
    assert route.calls[-1].request.read() == httpx.Request("POST", API, json=payload).read()


async def test_async_create_foundation_run_sends_idempotency_key(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post("/api/v1/training/foundation-runs").mock(
        return_value=httpx.Response(201, json={"run_id": "r1"})
    )
    await client.create_foundation_run({"project_id": "p1"})
    assert route.calls[-1].request.headers.get("Idempotency-Key")


async def test_async_create_foundation_run_404_is_not_a_missing_run(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    """A 404 at submit names an unresolved body id, never a run."""
    mock.post("/api/v1/training/foundation-runs").mock(
        return_value=httpx.Response(404, text="dataset version not found")
    )
    with pytest.raises(APIError) as exc_info:
        await client.create_foundation_run({"project_id": "p1"})
    assert not isinstance(exc_info.value, FoundationRunNotFoundError)
    assert "dataset version not found" in str(exc_info.value)


async def test_async_create_foundation_run_400(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/training/foundation-runs").mock(
        return_value=httpx.Response(400, text="max_seq_length: less than or equal to 4096")
    )
    with pytest.raises(APIError) as exc_info:
        await client.create_foundation_run({"project_id": "p1"})
    assert exc_info.value.status_code == 400


# --------------------------------------------------------------- read a run


async def test_async_get_foundation_run(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    body = {
        "run_id": "r1",
        "status": "running",
        "resolved_hyperparameters": {
            "epochs": 3,
            "a_field_added_after_this_release": {"nested": [1, None, True]},
        },
    }
    mock.get("/api/v1/training/foundation-runs/r1").mock(
        return_value=httpx.Response(200, json=body)
    )
    assert await client.get_foundation_run("r1") == body


async def test_async_get_foundation_run_quotes_the_id(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.get("/api/v1/training/foundation-runs/..%2Fjobs%2Fj1").mock(
        return_value=httpx.Response(200, json={"run_id": "x"})
    )
    await client.get_foundation_run("../jobs/j1")
    assert (
        route.calls[-1].request.url.raw_path.decode()
        == "/api/v1/training/foundation-runs/..%2Fjobs%2Fj1"
    )


async def test_async_get_foundation_run_404(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/training/foundation-runs/missing").mock(
        return_value=httpx.Response(404, text="foundation run not found")
    )
    with pytest.raises(FoundationRunNotFoundError) as exc_info:
        await client.get_foundation_run("missing")
    assert exc_info.value.run_id == "missing"


async def test_async_get_foundation_run_non_object_body(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/training/foundation-runs/r1").mock(
        return_value=httpx.Response(200, json=[1, 2])
    )
    with pytest.raises(TypeError):
        await client.get_foundation_run("r1")


# ------------------------------------------------------------------ evaluate

# Same fixture as the sync mirror -- a field this SDK version has never heard
# of, so a filtering layer between the wire and the caller is a visible diff.
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


async def test_async_create_evaluation(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    route = mock.post("/api/v1/training/evaluations").mock(
        return_value=httpx.Response(201, json=EVALUATION_RUN)
    )
    payload: JsonObject = {
        "project_id": "p1",
        "subject_version_id": "v1",
        "dataset_version_id": "dv1",
        "dataset_split": "test",
        "scorer_keys": ["accuracy@1"],
        "a_field_added_after_this_release": {"nested": [1, None, True]},
    }
    assert await client.create_evaluation(payload) == EVALUATION_RUN
    assert route.calls[-1].request.read() == httpx.Request("POST", API, json=payload).read()


async def test_async_create_evaluation_sends_idempotency_key(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post("/api/v1/training/evaluations").mock(
        return_value=httpx.Response(201, json={"run_id": "e1"})
    )
    await client.create_evaluation({"project_id": "p1"})
    assert route.calls[-1].request.headers.get("Idempotency-Key")


async def test_async_create_evaluation_404_is_not_a_missing_run(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    """A 404 at submit names an unresolved body id, never a run."""
    mock.post("/api/v1/training/evaluations").mock(
        return_value=httpx.Response(404, text="dataset version not found")
    )
    with pytest.raises(APIError) as exc_info:
        await client.create_evaluation({"project_id": "p1"})
    assert not isinstance(exc_info.value, EvaluationRunNotFoundError)
    assert "dataset version not found" in str(exc_info.value)


async def test_async_create_evaluation_400(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/training/evaluations").mock(
        return_value=httpx.Response(
            400,
            text="scorer 'macro-f1@1' is not compatible with task contract 'chat-completions@1.0'",
        )
    )
    with pytest.raises(APIError) as exc_info:
        await client.create_evaluation({"project_id": "p1"})
    assert exc_info.value.status_code == 400


async def test_async_get_evaluation(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/training/evaluations/e1").mock(
        return_value=httpx.Response(200, json=EVALUATION_RUN)
    )
    assert await client.get_evaluation("e1") == EVALUATION_RUN


async def test_async_get_evaluation_quotes_the_id(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.get("/api/v1/training/evaluations/..%2Fjobs%2Fj1").mock(
        return_value=httpx.Response(200, json={"run_id": "x"})
    )
    await client.get_evaluation("../jobs/j1")
    assert (
        route.calls[-1].request.url.raw_path.decode()
        == "/api/v1/training/evaluations/..%2Fjobs%2Fj1"
    )


async def test_async_get_evaluation_404(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/training/evaluations/missing").mock(
        return_value=httpx.Response(404, text="evaluation run not found")
    )
    with pytest.raises(EvaluationRunNotFoundError) as exc_info:
        await client.get_evaluation("missing")
    assert exc_info.value.run_id == "missing"


async def test_async_get_evaluation_non_object_body(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/training/evaluations/e1").mock(return_value=httpx.Response(200, json=[1, 2]))
    with pytest.raises(TypeError):
        await client.get_evaluation("e1")


# --------------------------------------------------------- version results

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


async def test_async_list_version_evaluations(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/model-versions/v1/evaluations").mock(
        return_value=httpx.Response(200, json=VERSION_EVALUATIONS)
    )
    assert await client.list_version_evaluations("v1") == VERSION_EVALUATIONS


async def test_async_list_version_evaluations_empty(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/model-versions/v1/evaluations").mock(
        return_value=httpx.Response(200, json=[])
    )
    assert await client.list_version_evaluations("v1") == []


async def test_async_list_version_evaluations_quotes_the_id(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.get("/api/v1/model-versions/..%2Fmodels%2Fm1/evaluations").mock(
        return_value=httpx.Response(200, json=[])
    )
    await client.list_version_evaluations("../models/m1")
    assert (
        route.calls[-1].request.url.raw_path.decode()
        == "/api/v1/model-versions/..%2Fmodels%2Fm1/evaluations"
    )


async def test_async_list_version_evaluations_404(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/model-versions/missing/evaluations").mock(
        return_value=httpx.Response(404, text="Model version not found")
    )
    with pytest.raises(ModelNotFoundError) as exc_info:
        await client.list_version_evaluations("missing")
    assert exc_info.value.model_id == "missing"


async def test_async_list_version_evaluations_non_array_body(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/model-versions/v1/evaluations").mock(
        return_value=httpx.Response(200, json={"detail": "nope"})
    )
    with pytest.raises(TypeError):
        await client.list_version_evaluations("v1")
