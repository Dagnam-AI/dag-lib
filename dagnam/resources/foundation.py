"""Foundation fine-tuning — the public SDK surface.

Fine-tune a curated base model with a shipped training recipe:

.. code-block:: python

    import dagnam

    recipe = dagnam.foundation.list_recipes()[0]
    base = dagnam.foundation.list_bases(limit=5)[0]
    # `recipe["hyperparameters"]` is a JSON Schema document -- the same one the
    # API validates against. Read the bounds from it; never restate them.
    run = dagnam.foundation.submit(
        project_id=project_id,
        base_catalog_entry_id=base["id"],
        dataset_version_id=dataset_version_id,
        recipe_key=recipe["key"],
        hyperparameters={"max_seq_length": 2048},
    )
    dagnam.foundation.get_run(run["run_id"])["status"]

Recipes and catalog entries come back exactly as served, and
``hyperparameters`` is forwarded exactly as given. That is the one design rule
of this module: a recipe declares its own hyperparameters once, and both the
form a caller renders and the validation the API performs must read that same
declaration. A convenience here that re-typed a field, capped a value, or
dropped an unrecognised key would create a second source that can disagree
with the first -- and would make every recipe published after this release
partly unusable. The API owns the verdict; this module owns the round trip.
"""

from __future__ import annotations

from dagnam._core.client import DagnamClient
from dagnam._core.resolver import resolve_client
from dagnam._types import JsonArray, JsonObject


def list_bases(
    *,
    page: int = 1,
    limit: int = 20,
    client: DagnamClient | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
) -> JsonArray:
    """List curated base models, one page at a time.

    Each entry carries the ``family`` a recipe's ``compatible_families``
    matches on, and the ``gated`` flag the submit gate refuses on -- so a
    caller can narrow the list to what it may actually fine-tune.

    ``limit`` is capped by the API; a larger value is rejected, not trimmed.
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.list_foundation_catalog(page=page, limit=limit)


def list_recipes(
    *,
    client: DagnamClient | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
) -> JsonArray:
    """List the shipped training recipes, serialized as the API published them.

    Each entry's ``"hyperparameters"`` is a JSON Schema document describing
    every tunable the recipe accepts, with its type, default and bounds. It is
    returned untouched: render from it, and the form cannot advertise a limit
    the API does not enforce.
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.list_training_recipes()


def submit(
    *,
    project_id: str,
    base_catalog_entry_id: str,
    dataset_version_id: str,
    recipe_key: str,
    hyperparameters: JsonObject | None = None,
    dataset_field_bindings: JsonObject | None = None,
    client: DagnamClient | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
) -> JsonObject:
    """Submit a fine-tuning run and return its frozen specification.

    ``hyperparameters`` is forwarded verbatim and validated by the API against
    the recipe's own schema -- omit it to train on the recipe's defaults.
    ``dataset_field_bindings`` maps the recipe's expected fields onto the
    dataset's actual column names.

    Returns the created run: ``run_id``, ``training_job_id`` (follow the job
    with the training APIs), the ``resolved_hyperparameters`` the run was
    frozen with, and its cost estimate.

    Raises:
        APIError: an id in the request did not resolve to something this
            caller may use (404, uniform for every id so it cannot be used to
            probe for other people's projects or datasets), or the recipe
            rejected the base, the dataset or a hyperparameter (400).
        APIError: the request was rejected before the recipe was consulted
            (422) -- an id that is not a well-formed UUID, or a field the API
            declares but does not honour yet.
        QuotaExceededError: the account is at its concurrent-training limit.
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.create_foundation_run(
        {
            "project_id": project_id,
            "base_catalog_entry_id": base_catalog_entry_id,
            "dataset_version_id": dataset_version_id,
            "recipe_key": recipe_key,
            "hyperparameters": hyperparameters or {},
            "dataset_field_bindings": dataset_field_bindings or {},
        }
    )


def get_run(
    run_id: str,
    *,
    client: DagnamClient | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
) -> JsonObject:
    """Read a run: its frozen specification plus its job's live status.

    Raises:
        FoundationRunNotFoundError: no such run for this caller. The API
            answers the same 404 for a run that does not exist and one that
            belongs to someone else.
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_foundation_run(run_id)


def evaluate(
    *,
    project_id: str,
    subject_version_id: str,
    dataset_version_id: str,
    dataset_split: str,
    scorer_keys: list[str],
    baseline_version_id: str | None = None,
    scorer_params: JsonObject | None = None,
    confirm_resource_warning: bool = False,
    client: DagnamClient | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
) -> JsonObject:
    """Score a registry model version against a held-out dataset split.

    Each entry in ``scorer_keys`` must appear in the subject's (and, when
    given, the baseline's) task contract compatibility list -- the API
    verifies this at submit time rather than silently skipping an
    incompatible scorer. Pass ``baseline_version_id`` to compare the subject
    against a second, contract-compatible version in the same run.

    Returns the created run: ``run_id``, ``training_job_id`` (follow the job
    with the training APIs), and the ``resolved_params`` each scorer was
    frozen with.

    Raises:
        APIError: an id in the request did not resolve to something this
            caller may use (404, uniform for every id so it cannot be used to
            probe for other people's projects, versions, or datasets), or the
            specification did not bind -- an unready subject/baseline, a
            baseline on a different task contract, an unsupported scorer, or
            a dataset that does not satisfy the evaluation's requirements
            (400).
        QuotaExceededError: the account is at its concurrent-training limit.
    """
    resolved = resolve_client(client, api_key, api_url)
    payload: JsonObject = {
        "project_id": project_id,
        "subject_version_id": subject_version_id,
        "baseline_version_id": baseline_version_id,
        "dataset_version_id": dataset_version_id,
        "dataset_split": dataset_split,
        "scorer_keys": [str(key) for key in scorer_keys],
        "scorer_params": scorer_params or {},
        "confirm_resource_warning": confirm_resource_warning,
    }
    return resolved.create_evaluation(payload)


def get_evaluation(
    run_id: str,
    *,
    client: DagnamClient | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
) -> JsonObject:
    """Read back a frozen evaluation specification.

    Raises:
        EvaluationRunNotFoundError: no such run for this caller. The API
            answers the same 404 for a run that does not exist and one that
            belongs to someone else.
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_evaluation(run_id)


def list_evaluations(
    version_id: str,
    *,
    client: DagnamClient | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
) -> JsonArray:
    """List every scorer result recorded against a registry model version.

    One entry per ``(evaluation_run_id, scorer_key)`` pair that has ever been
    recorded for this version, across every run that scored it -- not just
    the most recent. A version with no recorded evaluations returns an empty
    list, not an error.
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.list_version_evaluations(version_id)


__all__ = [
    "evaluate",
    "get_evaluation",
    "get_run",
    "list_bases",
    "list_evaluations",
    "list_recipes",
    "submit",
]
