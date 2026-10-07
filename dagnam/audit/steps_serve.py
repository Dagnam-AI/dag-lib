"""Serving steps: deploy the pushed version, wait for it, replay the holdout, score.

The deployment shape is the one proven end to end through the SDK (``vllm`` / ``text``
/ ``modal-serverless`` with a serverless revision of the model version); the
platform renders the app from the version's task contract, so a head-tuned
classifier and a chat adapter take the same path. The replay sends the
holdout's turns and lets the served bridge render them, so what the endpoint
scores is what a customer's client would get -- and the reading back is
``dagnam_contracts.prompts.parse_chat_prompt``, the inverse of the rendering
this module re-exports it beside, so a replay can never send text the
classifier was not trained on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from pathlib import Path
from typing import Any

from dagnam_contracts.audit.verdict import UNRELIABLE_ERROR_SHARE
from dagnam_contracts.prompts import parse_chat_prompt

from dagnam._core.exceptions import (
    APIError,
    DeploymentStateError,
    LROFailedError,
    LROTimeoutError,
)
from dagnam._types import JsonObject
from dagnam.audit.frontier import Endpoint, latency_of, replay_holdout
from dagnam.audit.scoring import Agreement, modal_keys, score_json, score_labels
from dagnam.audit.state import AuditState
from dagnam.audit.steps import (
    StepContext,
    answer_field,
    read_replay,
    replay_file,
    required,
    string_field,
    wait_for,
)
from dagnam.audit.steps_train import over_budget, projected_replay
from dagnam.audit.structure import StructureClass
from dagnam.audit.workspace import open_append, read_regular, write_atomic

PLATFORM = "vllm"
DEPLOYMENT_TYPE = "text"
INSTANCE_TYPE = "modal-serverless"
CAPACITY_MODE = "serverless"
DEPLOY_RUNNING = "running"
"""``StepState.deploy_status`` once ``wait_active`` saw the revision go live."""

type HoldoutRow = tuple[list[dict[str, str]], str]

_HOLDOUT: dict[str, Callable[[dict[str, Any]], HoldoutRow]] = {
    "labeled-example": lambda row: (parse_chat_prompt(str(row["input"])), str(row["label"])),
    "chat-messages": lambda row: (
        list(row["messages"][:-1]),
        str(row["messages"][-1]["content"]),
    ),
}
"""Row format -> (the request's messages, the teacher's answer)."""

_SCORERS: dict[StructureClass, Callable[[Sequence[str], Sequence[str]], Agreement]] = {
    StructureClass.ENUM_LABEL: score_labels,
    StructureClass.SHORT_SPAN: score_labels,
    StructureClass.JSON_OBJECT: lambda pred, truth: score_json(pred, truth, modal_keys(truth)),
}
"""Structure class -> scorer; free text has no candidates and so no scorer."""


def holdout(ctx: StepContext) -> list[HoldoutRow]:
    """The ``eval_holdout`` rows of ``dataset.jsonl`` as (messages, truth) pairs."""
    lines = read_regular(ctx.workload_dir / "dataset.jsonl").splitlines()
    rows = [json.loads(line) for line in lines]
    membership = json.loads(read_regular(ctx.workload_dir / "split.json"))
    build = _HOLDOUT[str(ctx.meta()["stats"]["format_key"])]
    return [build(rows[int(i)]) for i in membership["member_row_indices"]["eval_holdout"]]


def create_deployment(state: AuditState, ctx: StepContext) -> AuditState:
    """Create the deployment and put its one-time key in the secret store; done once ``deployment_id`` is set."""
    step = ctx.step(state)
    if step.deployment_id is not None:
        return state
    job_id = required(step.training_job_id, "training_job_id")
    payload: JsonObject = {
        "name": f"audit-{ctx.workload_id}-{ctx.spec.kind.value}",
        "project_id": required(state.project_id, "project_id"),
        "checkpoint_path": f"training-job://{job_id}",
        "platform": PLATFORM,
        "deployment_type": DEPLOYMENT_TYPE,
        "instance_type": INSTANCE_TYPE,
        "num_instances": 1,
        "auto_scaling_enabled": False,
        "training_job_id": job_id,
    }
    if state.audit_id is not None:
        # Written on the deployment by the platform at creation, so the audit's delete finds it.
        payload["audit_id"] = state.audit_id
    created = ctx.client.create_deployment(payload)
    step.deployment_id = answer_field(created, "id", "the deployment create")
    key = string_field(created, "api_key")
    if key is not None:
        ctx.secrets.store(ctx.label, key)
        step.key_ref = ctx.label
    return state


def create_revision(state: AuditState, ctx: StepContext) -> AuditState:
    """Roll the deployment to the pushed version as a serverless revision; done once ``deploy_status`` is set."""
    step = ctx.step(state)
    if step.deploy_status is not None:
        return state
    version_id = required(step.model_version_id, "model_version_id")
    ctx.client.create_deployment_revision(
        required(step.deployment_id, "deployment_id"),
        {
            "model_version_id": version_id,
            "capacity_mode": CAPACITY_MODE,
        },
        idempotency_key=f"audit-{ctx.label}-{version_id}",
    )
    step.deploy_status = "deploying"
    return state


def _revision_status(ctx: StepContext, deployment_id: str) -> JsonObject:
    """The newest revision (the one this audit created) as a pollable payload."""
    revisions = ctx.client.get_deployment_revisions(deployment_id)
    newest = revisions[0] if revisions and isinstance(revisions[0], dict) else None
    if newest is None:
        return {"status": "deploying"}
    if newest.get("is_active"):
        return {"status": DEPLOY_RUNNING}
    status = string_field(newest, "status") or "deploying"
    return {"status": status, "error_message": string_field(newest, "failure_reason")}


def wait_active(state: AuditState, ctx: StepContext) -> AuditState:
    """Wait for the revision to go active; a failure or timeout is recorded (an unpublished audit's endpoint is paused).

    A candidate that already scored waits for nothing: it was deployed,
    replayed and scored on an earlier run, and ``dagnam audit cancel`` may
    since have paused that endpoint. This is the one step whose "already done"
    guard is not implied by its own key -- a paused deployment is not
    ``running`` -- so without this a resumed run would wake a live endpoint
    back up to re-reach a number the state already holds.
    """
    step = ctx.step(state)
    if step.deploy_status == DEPLOY_RUNNING or step.scored:
        return state
    deployment_id = required(step.deployment_id, "deployment_id")
    try:
        wait_for(
            ctx,
            lambda: _revision_status(ctx, deployment_id),
            success={DEPLOY_RUNNING},
            failure={"failed"},
            timeout=ctx.deploy_timeout,
            name=f"deployment {deployment_id}",
        )
    except LROFailedError as exc:
        step.deploy_status = "failed"
        step.error = f"deploy_failed: {exc.detail or 'no reason given'}"
        return state
    except LROTimeoutError:
        step.deploy_status = "timeout"
        waited = f"deploy_timeout: not active after {ctx.deploy_timeout:.0f}s"
        if state.audit_id is not None:
            # A published audit's endpoint is paused by the audit's own cancel and nothing else.
            step.error = f"{waited}; `dagnam audit cancel` stops it"
            return state
        # A revision that never activated leaves the deployment unpausable; the timeout is
        # recorded either way, and says which it was.
        try:
            ctx.client.pause_deployment(deployment_id)
            step.error = f"{waited}; paused"
        except DeploymentStateError:
            step.error = f"{waited}; it could not be paused"
        return state
    step.deploy_status = DEPLOY_RUNNING
    return state


def _balance(ctx: StepContext) -> int | None:
    """The account's credit balance, or ``None`` when the read fails.

    A balance the platform will not report must never turn a completed replay
    into an unscored candidate, so the cost is left unknown instead.
    """
    try:
        return ctx.client.get_credit_balance()
    except APIError:
        return None


def _replay_file(ctx: StepContext) -> Path:
    """This candidate's answers file (:func:`~dagnam.audit.steps.replay_file`)."""
    return replay_file(ctx.audit_dir, ctx.workload_id, ctx.spec.kind)


def replay_and_score(state: AuditState, ctx: StepContext) -> AuditState:
    """Replay the holdout, record agreement, latency and the replay's credit cost; done once ``scored``.

    Every served prediction is metered, so the replay -- not the training --
    is most of what an audit spends. It is refused outright when its
    projected cost could take the credits spent past the ceiling,
    and its cost is measured from the account balance either side of it
    rather than assumed from a rate card.

    Each answer is written to :func:`_replay_file` as it lands, beside the
    balance read before the first attempt, so an interrupted replay resumes
    with only the rows it never heard back from -- and its cost still counts
    what the interrupted attempt burned.
    """
    step = ctx.step(state)
    if step.scored:
        return state
    key = ctx.secrets.load(step.key_ref) if step.key_ref is not None else None
    if key is None:
        step.error = "no_key: the deployment key is in neither the keyring nor the secrets file"
        return state
    rows = holdout(ctx)
    deployment_id = required(step.deployment_id, "deployment_id")
    path = _replay_file(ctx)
    earlier = read_replay(path, deployment_id)
    before, answers = (None, {}) if earlier is None else earlier
    # A row an earlier attempt never heard back from is sent again -- and so is
    # one that failed: an outage a Ctrl+C cut short must not freeze into
    # `unreliable`. What was answered is already counted as spent.
    todo = [index for index in range(len(rows)) if answers.get(index, (None, 0.0))[0] is None]
    if over_budget(state, ctx, projected_replay(len(todo)), f"{ctx.label} replay"):
        return state
    if earlier is None:
        # A file this replay cannot trust (a crash cut its head, or another
        # endpoint's) spent what nobody can say: the cost stays unknown, and
        # the budget counts it at its projection.
        before = None if path.exists() else _balance(ctx)
        head = {"deployment_id": deployment_id, "balance_before": before}
        write_atomic(path, json.dumps(head) + "\n")
    with open_append(path) as sink:

        def landed(position: int, answer: str | None, ms: float) -> None:
            index = todo[position]
            answers[index] = (answer, ms)
            sink.write(json.dumps({"row": index, "answer": answer, "ms": ms}) + "\n")
            sink.flush()

        replay_holdout(
            Endpoint(ctx.client.api_url, deployment_id, key),
            [{"messages": rows[index][0]} for index in todo],
            on_result=landed,
        )
    after = _balance(ctx)
    if before is not None and after is not None:
        # Clamped: a grant landing mid-replay would otherwise read as a refund.
        step.replay_cost_credits = max(0.0, float(before - after))
    results = [answers[index] for index in range(len(rows))]
    scored = [(a, t) for (a, _), (_, t) in zip(results, rows, strict=True) if a is not None]
    agreement = _SCORERS[ctx.structure_class]([a for a, _ in scored], [t for _, t in scored])
    step.agreement = {
        **agreement.to_json(),
        "floor": ctx.floor,
        "passes_floor": agreement.ci95[0] >= ctx.floor,
    }
    latency = latency_of(results)
    step.latency = latency.to_json()
    step.scored = True
    if latency.calls == 0 or latency.errors / latency.calls > UNRELIABLE_ERROR_SHARE:
        step.error = f"unreliable: {latency.errors} of {latency.calls} replay calls failed"
    return state


__all__ = [
    "CAPACITY_MODE",
    "DEPLOYMENT_TYPE",
    "DEPLOY_RUNNING",
    "INSTANCE_TYPE",
    "PLATFORM",
    "UNRELIABLE_ERROR_SHARE",
    "HoldoutRow",
    "create_deployment",
    "create_revision",
    "holdout",
    "parse_chat_prompt",
    "replay_and_score",
    "wait_active",
]
