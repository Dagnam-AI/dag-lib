"""Synchronous foundation fine-tuning client methods.

The discovery-and-submit surface for fine-tuning a curated base model with a
shipped training recipe: list the bases, list the recipes, submit a run, read
it back.

Every method returns the decoded response body **as it arrived**. That is a
deliberate design constraint rather than laziness: a recipe publishes its
hyperparameters as a JSON Schema document, and that document is the single
source both the API validates against and a caller renders a form from. An SDK
that re-declared a field, or narrowed the payload to the keys this release
happens to know, would let a client advertise a bound the server does not
enforce -- and would hide any hyperparameter added after this release. So no
response model is declared here, and none should be.
"""

from __future__ import annotations

from dagnam._core.client.base import (
    ALLOW_REDIRECTS,
    DEFAULT_TIMEOUT,
    BaseDagnamClient,
)
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_generic,
    raise_for_model,
    requests_query_params,
    response_json_value,
)
from dagnam._core.exceptions import EvaluationRunNotFoundError, FoundationRunNotFoundError
from dagnam._types import JsonArray, JsonObject, JsonValue, QueryParams, ResponseLike

_CATALOG_PATH = "/api/v1/foundation-catalog"
_RECIPES_PATH = "/api/v1/training/recipes"
_RUNS_PATH = "/api/v1/training/foundation-runs"
_EVALUATIONS_PATH = "/api/v1/training/evaluations"
_MODEL_VERSIONS_PATH = "/api/v1/model-versions"


def raise_for_foundation_run(resp: ResponseLike, run_id: str | None = None) -> None:
    """Map a response to a typed error; 404 names a run only when one exists.

    Shared by the sync and async mixins so both answer a 404 identically.
    ``run_id`` is set only when the request addressed one particular run, so a
    404 from the submit route -- where the API answers one uniform "not found"
    for whichever id in the body did not resolve -- stays an :class:`APIError`
    carrying the server's message, instead of naming a run that was never
    created.
    """
    raise_for_generic(resp, FoundationRunNotFoundError if run_id else None, run_id)


def raise_for_evaluation_run(resp: ResponseLike, run_id: str | None = None) -> None:
    """Map a response to a typed error; 404 names a run only when one exists.

    Same "uniform 404, name only what could exist" rule as
    ``raise_for_foundation_run``: the submit route's 404 means some id in the
    request body did not resolve, never the run itself (none was created), so
    the submit path calls this with ``run_id=None`` and a 404 there stays a
    generic :class:`~dagnam._core.exceptions.APIError`.
    """
    raise_for_generic(resp, EvaluationRunNotFoundError if run_id else None, run_id)


class FoundationClientMixin(BaseDagnamClient):
    """Foundation fine-tuning resource methods for DagnamClient."""

    def _foundation_request(
        self,
        method: str,
        path: str,
        *,
        run_id: str | None = None,
        params: QueryParams | None = None,
        json_body: JsonValue = None,
        idempotent: bool = False,
    ) -> JsonValue:
        """Issue an authenticated foundation request and decode its JSON body.

        Every route here answers JSON on success, so a body that will not
        decode raises :class:`ResponseError` rather than being smuggled
        through as text -- a gateway's HTML error page is not an empty recipe
        list.
        """
        resp = self._request(
            method,
            f"{self.api_url}{path}",
            raise_for=lambda r: raise_for_foundation_run(r, run_id),
            params=requests_query_params(params),
            json=json_body,
            timeout=DEFAULT_TIMEOUT,
            allow_redirects=ALLOW_REDIRECTS,
            idempotent=idempotent,
        )
        return response_json_value(resp)

    def list_foundation_catalog(self, *, page: int = 1, limit: int = 20) -> JsonArray:
        """One bounded page of the curated base models. ``GET /foundation-catalog``.

        ``limit`` is capped server-side; asking for more is rejected rather
        than silently trimmed.
        """
        return self._expect_array(
            self._foundation_request("GET", _CATALOG_PATH, params={"page": page, "limit": limit})
        )

    def list_training_recipes(self) -> JsonArray:
        """Every shipped training recipe, serialized. ``GET /training/recipes``.

        Each entry carries its hyperparameter JSON Schema under
        ``"hyperparameters"``. Returned untouched -- see the module docstring.
        """
        return self._expect_array(self._foundation_request("GET", _RECIPES_PATH))

    def create_foundation_run(self, payload: JsonObject) -> JsonObject:
        """Submit a fine-tuning run. ``POST /training/foundation-runs``.

        Sends an ``Idempotency-Key``: this call reserves GPU credits, so a
        transient failure must retry into a server-side replay rather than
        start a second run.
        """
        return self._expect_object(
            self._foundation_request("POST", _RUNS_PATH, json_body=payload, idempotent=True)
        )

    def get_foundation_run(self, run_id: str) -> JsonObject:
        """Read one run's frozen specification and its job's live status."""
        return self._expect_object(
            self._foundation_request(
                "GET", f"{_RUNS_PATH}/{quote_path_segment(run_id)}", run_id=run_id
            )
        )

    def _evaluation_request(
        self,
        method: str,
        path: str,
        *,
        run_id: str | None = None,
        json_body: JsonValue = None,
        idempotent: bool = False,
    ) -> JsonValue:
        """Issue an authenticated evaluation request and decode its JSON body.

        A sibling of ``_foundation_request`` rather than a shared call: an
        evaluation run's 404 maps to its own typed error
        (``EvaluationRunNotFoundError``), never ``FoundationRunNotFoundError``.
        """
        resp = self._request(
            method,
            f"{self.api_url}{path}",
            raise_for=lambda r: raise_for_evaluation_run(r, run_id),
            json=json_body,
            timeout=DEFAULT_TIMEOUT,
            allow_redirects=ALLOW_REDIRECTS,
            idempotent=idempotent,
        )
        return response_json_value(resp)

    def create_evaluation(self, payload: JsonObject) -> JsonObject:
        """Submit an evaluation run. ``POST /training/evaluations``.

        Sends an ``Idempotency-Key``: like a fine-tuning run, this reserves a
        job slot and starts dispatch, so a transient failure must retry into a
        server-side replay rather than start a second run. The payload is
        forwarded exactly as given and the response exactly as served -- an
        evaluation specification is scored server-side; this layer does not
        interpret it.
        """
        return self._expect_object(
            self._evaluation_request("POST", _EVALUATIONS_PATH, json_body=payload, idempotent=True)
        )

    def get_evaluation(self, run_id: str) -> JsonObject:
        """Read one evaluation run's frozen specification."""
        return self._expect_object(
            self._evaluation_request(
                "GET", f"{_EVALUATIONS_PATH}/{quote_path_segment(run_id)}", run_id=run_id
            )
        )

    def list_version_evaluations(self, version_id: str) -> JsonArray:
        """Every scorer result recorded against this model version, from any run.

        ``GET /model-versions/{id}/evaluations``. Results are forwarded
        exactly as served -- fabricating or dropping a scorer's row here would
        misrepresent what was actually measured.
        """
        resp = self._request(
            "GET",
            f"{self.api_url}{_MODEL_VERSIONS_PATH}/{quote_path_segment(version_id)}/evaluations",
            raise_for=lambda r: raise_for_model(r, version_id),
            timeout=DEFAULT_TIMEOUT,
            allow_redirects=ALLOW_REDIRECTS,
        )
        return self._expect_array(response_json_value(resp))
