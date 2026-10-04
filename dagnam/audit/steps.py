"""Shared ground for the audit's platform steps: the client surface, the context, the waits.

A step is a pure function ``(state, ctx) -> state`` guarded by its own
"already done" check, so the orchestrator can run the whole list on every
invocation and only the unfinished work touches the platform. The steps live
by seam in :mod:`steps_data`, :mod:`steps_train` and :mod:`steps_serve`.

The audit drives the SDK's ``DagnamClient`` through :class:`PlatformClient`
-- the exact ``_core`` methods it needs and nothing else -- so a test can hand
it a fake and the layer contract (``audit`` never imports ``resources``) holds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Protocol

from dagnam._core.exceptions import PlatformAnswerError
from dagnam._core.lro import LongRunningOperation
from dagnam._types import JsonArray, JsonMapping, JsonObject
from dagnam.audit.candidates import CandidateKind, CandidateSpec
from dagnam.audit.secrets import SecretStore
from dagnam.audit.state import AuditState, StepState
from dagnam.audit.structure import StructureClass
from dagnam.audit.workspace import (
    check_writable,
    read_regular,
    workload_dir as checked_workload_dir,
)

RUN_TIMEOUT_SECONDS = 7200.0
"""How long ``wait_run`` follows one training run before giving up (resumable)."""
DEPLOY_TIMEOUT_SECONDS = 1200.0
"""A revision not active within this is ``deploy_timeout``."""
TASK_TIMEOUT_SECONDS = 300.0
"""Dataset tasks (sniff, split, PII scan) settle in seconds; this is the ceiling."""

CONFLICT_STATUS = 409
"""The platform's refusal of a state change: a blocked delete, a refused transition, a halt."""

# ``GET /api/v1/datasets/tasks/{id}`` reports the task queue's own status
# under ``status`` -- the same spellings ``dagnam.resources.datasets`` accepts.
TASK_SUCCESS = frozenset({"completed", "ready", "success", "SUCCESS"})
TASK_FAILURE = frozenset({"failed", "failure", "FAILURE", "cancelled", "revoked", "REVOKED"})


class PlatformClient(Protocol):
    """The ``DagnamClient`` methods the audit drives; a fake implements these and no more."""

    api_url: str
    resume_creates: bool
    """Set by the run: a create it repeats finds what the first ask made (see ``dagnam._core._resume``)."""

    def get_platform_build(self) -> JsonObject:
        """``GET /health/build``: the running build, with the ``dagnam-contracts`` version it installed."""
        ...

    def create_project(self, payload: JsonObject, *, resume_nonce: str | None = None) -> JsonObject:
        """``POST /api/v1/projects`` (idempotent; the client retries transients).

        ``resume_nonce`` is what lets a create that was interrupted replay its own project.
        """
        ...

    def get_credit_balance(self) -> int:
        """``GET /api/v1/users/me/credits`` -> the balance the replay's cost is measured against."""
        ...

    def upload_dataset(
        self,
        file_path: str | Path,
        name: str,
        dataset_type: str,
        format: str,
        description: str | None = None,
        visibility: str = "private",
        license: str | None = None,
        progress_cb: object = None,
        audit_id: str | None = None,
    ) -> JsonObject:
        """``POST /api/v1/datasets/`` (multipart); ``audit_id`` has the platform tag the row."""
        ...

    def list_datasets(
        self, type: str = "all", search: str | None = None, audit_id: str | None = None
    ) -> list[JsonObject]:
        """``GET /api/v1/datasets/browse``.

        ``search`` matches the name and the description; ``audit_id`` keeps the datasets the
        platform tagged with that audit.
        """
        ...

    def get_dataset(self, dataset_id: str) -> JsonObject:
        """``GET /api/v1/datasets/{id}`` -> the row, with ``analysis_status``/``analysis_error``."""
        ...

    def list_dataset_versions(self, dataset_id: str) -> list[JsonObject]:
        """``GET /api/v1/datasets/{id}/versions``."""
        ...

    def create_explicit_splits(
        self, dataset_id: str, version_id: str, memberships: Mapping[str, Sequence[int]]
    ) -> JsonObject:
        """``POST .../versions/{vid}/splits/explicit`` -> ``{"task_id"}``."""
        ...

    def scan_pii(
        self, dataset_id: str, version_id: str, policy: Mapping[str, str] | None = None
    ) -> JsonObject:
        """``POST .../versions/{vid}/pii-scan`` -> ``{"task_id"}``."""
        ...

    def get_dataset_task_status(self, task_id: str) -> JsonObject:
        """``GET /api/v1/datasets/tasks/{task_id}``."""
        ...

    def list_foundation_catalog(self, *, page: int = 1, limit: int = 20) -> JsonArray:
        """``GET /api/v1/foundation-catalog``."""
        ...

    def create_foundation_run(self, payload: JsonObject) -> JsonObject:
        """``POST /api/v1/training/foundation-runs`` (idempotent; the client retries transients)."""
        ...

    def get_foundation_run(self, run_id: str) -> JsonObject:
        """``GET /api/v1/training/foundation-runs/{id}``."""
        ...

    def create_deployment(self, payload: JsonObject) -> JsonObject:
        """``POST /api/v1/deployments`` -> the record plus its one-time ``api_key``."""
        ...

    def create_deployment_revision(
        self, deployment_id: str, payload: JsonObject, *, idempotency_key: str | None = None
    ) -> JsonObject:
        """``POST /api/v1/deployments/{id}/revisions``."""
        ...

    def get_deployment_revisions(
        self, deployment_id: str, *, page: int = 1, limit: int = 50
    ) -> JsonArray:
        """``GET /api/v1/deployments/{id}/revisions`` (newest first)."""
        ...

    def pause_deployment(self, deployment_id: str) -> JsonObject:
        """``POST /api/v1/deployments/{id}/pause``."""
        ...

    # -- publishing the audit to the account (dagnam.audit.publish) -------------

    def create_audit(self, payload: JsonObject, *, resume_nonce: str | None = None) -> JsonObject:
        """``POST /api/v1/audits`` -> the audit record (idempotent); its ``id`` is the ``audit_id``.

        ``resume_nonce`` is what lets a create that was interrupted replay its own audit.
        """
        ...

    def claim_audit_resources(self, audit_id: str, entries: JsonArray) -> JsonObject:
        """``POST /api/v1/audits/{id}/claims``: ask the platform to take ids made before the audit."""
        ...

    def create_audit_candidate(self, audit_id: str, payload: JsonObject) -> JsonObject:
        """``POST /api/v1/audits/{id}/candidates`` (idempotent per workload x kind)."""
        ...

    def patch_audit_candidate(
        self, audit_id: str, candidate_id: str, payload: JsonObject
    ) -> JsonObject:
        """``PATCH /api/v1/audits/{id}/candidates/{cid}``: one step of a candidate."""
        ...

    def get_audit(self, audit_id: str) -> JsonObject:
        """``GET /api/v1/audits/{id}``: its ``status``; the uniform 404 once it is deleted."""
        ...

    def get_project(self, project_id: str) -> JsonObject:
        """``GET /api/v1/projects/{id}``: the project, with its ``owner_id``."""
        ...

    def resume_audit(self, audit_id: str) -> JsonObject:
        """``POST /api/v1/audits/{id}/resume``: un-halt it for a run starting again."""
        ...

    def halt_audit(self, audit_id: str, reason: str) -> JsonObject:
        """``POST /api/v1/audits/{id}/halt``."""
        ...


@dataclass(frozen=True, slots=True)
class StepContext:
    """Everything a step needs besides the state: which candidate, through which client, with what clock."""

    audit_dir: Path
    workload_id: str
    structure_class: StructureClass
    spec: CandidateSpec
    client: PlatformClient
    secrets: SecretStore
    floor: float
    max_credits: int
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.monotonic
    run_timeout: float = RUN_TIMEOUT_SECONDS
    deploy_timeout: float = DEPLOY_TIMEOUT_SECONDS
    task_timeout: float = TASK_TIMEOUT_SECONDS
    platform_contracts: str | None = None
    """The ``dagnam-contracts`` version the platform reported before the run; ``None`` if it did not."""

    @property
    def workload_dir(self) -> Path:
        """``audit_dir/workloads/<id>``: ``dataset.jsonl``, ``split.json``, ``meta.json``.

        Through :func:`~dagnam.audit.workspace.workload_dir`: an id that is not one
        plain name, or a folder that is a link, is refused rather than followed.
        """
        return checked_workload_dir(self.audit_dir, self.workload_id)

    @property
    def label(self) -> str:
        """``<workload>/<candidate>``: the key_ref, the idempotency prefix, the artifact names."""
        return f"{self.workload_id}/{self.spec.kind.value}"

    def step(self, state: AuditState) -> StepState:
        """This candidate's step state."""
        return state.candidate(self.workload_id, self.spec.kind)

    def meta(self) -> dict[str, Any]:
        """``meta.json`` as the scan wrote it."""
        return workload_meta(self.audit_dir, self.workload_id)

    def holdout_rows(self) -> int:
        """How many ``eval_holdout`` rows the replay sends: what its metered cost is projected on."""
        return int(self.meta()["splits"]["eval_holdout"])


type Step = Callable[[AuditState, StepContext], AuditState]
type Answer = tuple[str | None, float]
"""One replayed row: the answer (``None`` when the call failed) and its round trip in ms."""


def workload_meta(audit_dir: Path, workload_id: str) -> dict[str, Any]:
    """One workload's ``meta.json``, read through the checked folder (never a link, never outside)."""
    return json.loads(read_regular(checked_workload_dir(audit_dir, workload_id) / "meta.json"))


def regular_path(path: Path) -> Path:
    """``path``, once a link or anything that is not a regular file there has been refused.

    For a file a client opens itself (the upload reads ``dataset.jsonl`` with ``open``): a
    planted pipe would block it for ever while the audit directory is locked.
    """
    check_writable(path)
    return path


def replay_file(audit_dir: Path, workload_id: str, kind: CandidateKind) -> Path:
    """Where a candidate's replay writes each answer as it lands: ``workloads/<id>/replay-<kind>.jsonl``."""
    return checked_workload_dir(audit_dir, workload_id) / f"replay-{kind.value}.jsonl"


def _replay_lines(path: Path) -> tuple[dict[str, Any] | None, dict[int, Answer]]:
    """``(head, answers by row)`` of an answers file; a line a crash cut short is skipped.

    A head that does not parse is ``None``; a later line for a row wins.
    """
    lines = read_regular(path).splitlines() if path.exists() else []
    parsed: list[Any] = []
    for line in lines:
        try:
            parsed.append(json.loads(line))
        except ValueError:
            parsed.append(None)
    head = parsed[0] if parsed and isinstance(parsed[0], dict) else None
    answers = {
        int(row["row"]): (row["answer"], float(row["ms"]))
        for row in parsed[1:]
        if isinstance(row, dict)
    }
    return head, answers


def read_replay(path: Path, deployment_id: str) -> tuple[int | None, dict[int, Answer]] | None:
    """``(balance before the first attempt, answers by row)`` a replay left, or ``None``.

    ``None`` when there is no file, or none this replay can trust: its head
    names another endpoint, or a crash cut the head itself short.
    """
    head, answers = _replay_lines(path)
    if head is None or head.get("deployment_id") != deployment_id:
        return None
    return head.get("balance_before"), answers


def answered_rows(path: Path) -> int:
    """How many rows an answers file shows answered, whatever state its head is in."""
    return sum(1 for answer, _ in _replay_lines(path)[1].values() if answer is not None)


def required(value: str | None, what: str) -> str:
    """A state key an earlier step must have set; missing means the steps ran out of order."""
    if value is None:
        raise RuntimeError(f"step ordering bug: {what} is not set yet")
    return value


def answer_field(payload: Mapping[str, Any], key: str, what: str) -> str:
    """``payload[key]`` of the platform's answer to ``what``, or an error that says what was missing.

    A 200 whose body lacks the field is the platform's fault, not a ``KeyError`` of this client's.
    """
    value = payload.get(key)
    if value is None:
        raise PlatformAnswerError(f"the platform's answer to {what} had no {key!r}")
    return str(value)


def string_field(payload: Mapping[str, Any], key: str) -> str | None:
    """``payload[key]`` when it is a string, else ``None``."""
    value = payload.get(key)
    return value if isinstance(value, str) else None


def error_code(step: StepState) -> str | None:
    """The token before the colon of ``step.error`` (``"pii_disagreement"``), or ``None``."""
    return step.error.partition(":")[0] if step.error else None


def wait_for(
    ctx: StepContext,
    poll: Callable[[], JsonMapping],
    *,
    success: frozenset[str] | set[str],
    failure: frozenset[str] | set[str],
    timeout: float,
    name: str,
    error_key: str | tuple[str, ...] = "error_message",
) -> JsonMapping:
    """Poll until ``poll()``'s ``status`` is terminal, on the context's clock.

    Raises:
        LROFailedError: the status landed in ``failure``.
        LROTimeoutError: ``timeout`` elapsed first; the next ``run`` resumes waiting.
    """
    op = LongRunningOperation(
        poll=poll, success_states=success, failure_states=failure, error_key=error_key, name=name
    )
    return op.wait(timeout, sleep=ctx.sleep, now=ctx.now).result()


def wait_task(ctx: StepContext, task_id: str) -> JsonObject:
    """A dataset task's ``result`` object once it settles (empty when it carried none)."""
    payload = wait_for(
        ctx,
        lambda: ctx.client.get_dataset_task_status(task_id),
        success=TASK_SUCCESS,
        failure=TASK_FAILURE,
        timeout=ctx.task_timeout,
        name=f"dataset task {task_id}",
        error_key=("error", "error_message"),
    )
    result = payload.get("result")
    return result if isinstance(result, dict) else {}


__all__ = [
    "CONFLICT_STATUS",
    "DEPLOY_TIMEOUT_SECONDS",
    "RUN_TIMEOUT_SECONDS",
    "TASK_FAILURE",
    "TASK_SUCCESS",
    "TASK_TIMEOUT_SECONDS",
    "Answer",
    "PlatformClient",
    "Step",
    "StepContext",
    "answer_field",
    "answered_rows",
    "error_code",
    "read_replay",
    "regular_path",
    "replay_file",
    "required",
    "string_field",
    "wait_for",
    "wait_task",
    "workload_meta",
]
