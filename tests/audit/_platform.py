"""An in-memory platform the orchestration tests drive instead of the network.

``FakePlatform`` implements exactly :class:`dagnam.audit.steps.PlatformClient`
with a scripted state machine (upload -> ids; run polls -> completed;
revision polls -> active) and a ``call_log`` of every method name, so a test
can assert a second run made zero platform calls. The chat endpoint is a
``requests_mock`` route served by :func:`serve_chat`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from dagnam_contracts.prompts import render_chat_prompt
from tests.typing_helpers import RequestsMocker

from dagnam._core.exceptions import (
    APIError,
    DagnamError,
    DatasetNotFoundError,
    DeploymentNotFoundError,
    ModelNotFoundError,
    ProjectNotFoundError,
    TrainingJobNotFoundError,
)
from dagnam._types import JsonArray, JsonObject, QueryValue
from dagnam.audit.cleanup import CleanupClient
from dagnam.audit.steps import PlatformClient

CHAT_URL = "https://x/v1/chat/completions"

DATASET_IN_USE = "Dataset is referenced by a training run and cannot be deleted"
"""The platform's own 409 wording (``src/datasets/service.py``)."""

BASES: list[JsonObject] = [
    {"id": "bert-big", "family": "bert", "parameter_count": 340_000_000, "gated": False},
    {
        "id": "bert-small",
        "display_name": "BERT small",
        "family": "bert",
        "parameter_count": 110_000_000,
    },
    {"id": "bert-gated", "family": "bert", "parameter_count": 10, "gated": True},
    {"id": "bert-unsized", "family": "bert", "parameter_count": None},
    {"id": "qwen-7b", "family": "qwen2", "parameter_count": 7_000_000_000},
    {"id": "qwen-05b", "family": "qwen2", "parameter_count": 500_000_000},
    {"id": "llama-1b", "family": "llama", "parameter_count": 1_000_000_000},
]


class FakePlatform:
    """The scripted platform; every knob is a plain attribute a test sets before running."""

    api_url = "https://x"

    def __init__(self) -> None:
        self.call_log: list[str] = []
        self.submits = 0
        self.submitted: list[JsonObject] = []
        self.credits_estimate_max: int | None = 100
        self.credit_balance = 1000
        self.replay_charge = 4
        """Credits the balance drops between the two reads that bracket one replay."""
        self.credit_errors: list[BaseException | None] = []
        """Per-read overrides: the i-th balance read raises ``credit_errors[i]`` when set."""
        self.balance_reads = 0
        self.submit_errors: list[BaseException] = []
        self.run_polls = 2
        self.run_final_status = "completed"
        self.run_error: str | None = None
        self.run_extra: JsonObject = {}
        self.pushes_version = True
        self.version_polls = 1
        self.split_result: JsonObject | None = None
        self.pii_counts: dict[str, JsonObject] = {}
        """Residual server findings keyed by dataset id (``"ds-1"``); absent means clean."""
        self.pii_pass_list: list[str] = [
            "PII_EMAIL",
            "PII_PHONE",
            "PII_PAYMENT_CARD",
            "PII_NATIONAL_ID",
        ]
        self.bases: JsonArray = list(BASES)
        self.deployment_key: str | None = "dk-secret"
        self.revision_polls = 2
        self.revision_final = "active"
        self.uploads: dict[str, str] = {}
        self.deployments: list[JsonObject] = []
        self.revisions: list[JsonObject] = []
        self.paused: list[str] = []
        self.audits: list[dict[str, Any]] = []
        """``create_audit`` payloads, in order."""
        self.candidates: list[tuple[str, dict[str, Any]]] = []
        """``(audit_id, payload)`` per ``create_audit_candidate``."""
        self.patches: list[tuple[str, dict[str, Any]]] = []
        """``(candidate_id, payload)`` per ``patch_audit_candidate``."""
        self.halts: list[tuple[str, str]] = []
        self.cancelled: list[str] = []
        self.deleted_audits: list[str] = []
        self.publish_errors: dict[str, list[BaseException]] = {}
        """Per method name: the exception the next call raises, popped in order."""
        self.forbid_publishing = False
        """``--local-only``: any audit route is a test failure, not a recorded call."""
        self.receipt: JsonObject = {
            "schema": "dagnam.audit.deleted/1",
            "deleted_at": "2026-09-07T10:00:00+00:00",
            "entries": [{"kind": "project", "id": "proj-1", "status": "deleted", "reason": None}],
        }
        self._polls: Counter[str] = Counter()
        self._ids: Counter[str] = Counter()

    def _log(self, name: str) -> None:
        self.call_log.append(name)

    def _next(self, prefix: str) -> str:
        self._ids[prefix] += 1
        return f"{prefix}-{self._ids[prefix]}"

    def get_credit_balance(self) -> int:
        """The metered balance: it drops by ``replay_charge`` on every second read."""
        self._log("get_credit_balance")
        self.balance_reads += 1
        error = (
            self.credit_errors[self.balance_reads - 1]
            if self.balance_reads <= len(self.credit_errors)
            else None
        )
        if error is not None:
            raise error
        return self.credit_balance - self.replay_charge * (self.balance_reads // 2)

    # -- projects / datasets ------------------------------------------------

    def create_project(self, payload: JsonObject) -> JsonObject:
        self._log("create_project")
        return {"id": self._next("proj"), "title": payload["title"]}

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
    ) -> JsonObject:
        self._log("upload_dataset")
        first = json.loads(Path(file_path).read_text(encoding="utf-8").splitlines()[0])
        dataset_id = self._next("ds")
        self.uploads[dataset_id] = "labeled-example" if "label" in first else "chat-messages"
        return {"id": dataset_id, "name": name, "format": format, "dataset_type": dataset_type}

    def list_dataset_versions(self, dataset_id: str) -> list[JsonObject]:
        self._log("list_dataset_versions")
        self._polls[dataset_id] += 1
        if self._polls[dataset_id] < self.version_polls:
            return [{"id": f"{dataset_id}-v1", "version_number": 1}]
        return [
            {"id": f"{dataset_id}-v0", "version_number": 0, "num_samples": 1, "data_format": "x"},
            {
                "id": f"{dataset_id}-v1",
                "version_number": 1,
                "num_samples": 20,
                "data_format": self.uploads[dataset_id],
            },
        ]

    def create_explicit_splits(
        self, dataset_id: str, version_id: str, memberships: Mapping[str, Sequence[int]]
    ) -> JsonObject:
        self._log("create_explicit_splits")
        self.memberships = {name: list(rows) for name, rows in memberships.items()}
        return {"task_id": self._next(f"split:{dataset_id}")}

    def scan_pii(
        self, dataset_id: str, version_id: str, policy: Mapping[str, str] | None = None
    ) -> JsonObject:
        self._log("scan_pii")
        self.pii_version = version_id
        return {"task_id": self._next(f"pii:{dataset_id}")}

    def get_dataset_task_status(self, task_id: str) -> JsonObject:
        self._log("get_dataset_task_status")
        self._polls[task_id] += 1
        if self._polls[task_id] < 2:
            return {"status": "PENDING", "state": "pending"}
        if task_id.startswith("split:"):
            dataset_id = task_id.split(":")[1].rsplit("-", 1)[0]
            result: JsonObject = self.split_result or {
                "status": "completed",
                "new_version_id": f"{dataset_id}-v2",
                "splits": {name: len(rows) for name, rows in self.memberships.items()},
            }
            return {"status": "SUCCESS", "state": "completed", "result": result}
        dataset_id = task_id.split(":")[1].rsplit("-", 1)[0]
        return {
            "status": "SUCCESS",
            "result": {
                "status": "completed",
                "counts_by_code": dict(self.pii_counts.get(dataset_id, {})),
                "pass_list": list(self.pii_pass_list),
            },
        }

    # -- foundation runs ------------------------------------------------------

    def list_foundation_catalog(self, *, page: int = 1, limit: int = 20) -> JsonArray:
        self._log("list_foundation_catalog")
        return list(self.bases)

    def create_foundation_run(self, payload: JsonObject) -> JsonObject:
        self._log("create_foundation_run")
        if self.submit_errors:
            raise self.submit_errors.pop(0)
        self.submits += 1
        self.submitted.append(payload)
        run_id = self._next("run")
        return {
            "run_id": run_id,
            "training_job_id": run_id.replace("run", "job"),
            "status": "queued",
            "credits_estimate_max": self.credits_estimate_max,
        }

    def get_foundation_run(self, run_id: str) -> JsonObject:
        self._log("get_foundation_run")
        self._polls[run_id] += 1
        if self._polls[run_id] <= self.run_polls:
            return {"run_id": run_id, "status": "running"}
        return {
            "run_id": run_id,
            "status": self.run_final_status,
            "error_message": self.run_error,
            "created_at": "2026-09-06T10:00:00Z",
            "credits_estimate_max": self.credits_estimate_max,
            **self.run_extra,
        }

    def list_model_entries(self, **filter_params: QueryValue) -> JsonArray:
        self._log("list_model_entries")
        return [{"id": "entry-1"}, {"id": "entry-2"}, "not-an-entry"]

    def list_model_versions(self, model_id: str) -> JsonArray:
        self._log("list_model_versions")
        if model_id != "entry-1":
            return [
                "junk",
                {"id": "mv-undated"},
                {"id": "mv-2020", "created_at": "2020-01-01T00:00:00"},
            ]
        if not self.pushes_version:
            return []
        return [
            {"id": "mv-pushed", "created_at": "2026-09-06T10:05:00+00:00"},
            {"id": "mv-earlier", "created_at": "2026-09-06T10:01:00+00:00"},
        ]

    # -- deployments ------------------------------------------------------------

    def create_deployment(self, payload: JsonObject) -> JsonObject:
        self._log("create_deployment")
        self.deployments.append(payload)
        record: JsonObject = {"id": self._next("dep"), "status": "deploying"}
        if self.deployment_key is not None:
            record["api_key"] = self.deployment_key
        return record

    def create_deployment_revision(
        self, deployment_id: str, payload: JsonObject, *, idempotency_key: str | None = None
    ) -> JsonObject:
        self._log("create_deployment_revision")
        self.revisions.append({**payload, "deployment_id": deployment_id, "key": idempotency_key})
        return {"id": "rev-1", "revision_number": 1, "status": "pending", "is_active": False}

    def get_deployment_revisions(
        self, deployment_id: str, *, page: int = 1, limit: int = 50
    ) -> JsonArray:
        self._log("get_deployment_revisions")
        self._polls[f"rev:{deployment_id}"] += 1
        if self._polls[f"rev:{deployment_id}"] < self.revision_polls:
            return [{"id": "rev-1", "status": "deploying", "is_active": False}]
        if self.revision_final == "active":
            return [{"id": "rev-1", "status": "active", "is_active": True}]
        if self.revision_final == "failed":
            return [{"id": "rev-1", "status": "failed", "failure_reason": "no gpu"}]
        return [{"id": "rev-1", "status": "deploying", "is_active": False}]

    def pause_deployment(self, deployment_id: str) -> JsonObject:
        self._log("pause_deployment")
        self.paused.append(deployment_id)
        return {"id": deployment_id, "status": "paused"}

    # -- publishing the audit ----------------------------------------------------

    def _publish(self, name: str) -> None:
        """Log one audit route, refusing it outright under ``forbid_publishing``."""
        if self.forbid_publishing:
            raise AssertionError("must not be called")
        self._log(name)
        errors = self.publish_errors.get(name)
        if errors:
            raise errors.pop(0)

    def create_audit(self, payload: JsonObject) -> JsonObject:
        self._publish("create_audit")
        self.audits.append(payload)
        return {"id": self._next("audit")}

    def create_audit_candidate(self, audit_id: str, payload: JsonObject) -> JsonObject:
        self._publish("create_audit_candidate")
        self.candidates.append((audit_id, payload))
        return {"id": self._next("cand")}

    def patch_audit_candidate(
        self, audit_id: str, candidate_id: str, payload: JsonObject
    ) -> JsonObject:
        self._publish("patch_audit_candidate")
        self.patches.append((candidate_id, payload))
        return {"id": candidate_id, "status": payload["status"]}

    def halt_audit(self, audit_id: str, reason: str) -> JsonObject:
        self._publish("halt_audit")
        self.halts.append((audit_id, reason))
        return {"id": audit_id, "status": "halted", "halted_reason": reason}

    def cancel_audit(self, audit_id: str) -> JsonObject:
        self._publish("cancel_audit")
        self.cancelled.append(audit_id)
        return dict(self.receipt)

    def delete_audit(self, audit_id: str) -> JsonObject:
        self._publish("delete_audit")
        self.deleted_audits.append(audit_id)
        return dict(self.receipt)


class FakeCleanup:
    """A platform holding ids to delete: each ``get`` raises not-found once its id is gone."""

    def __init__(self, **present: list[str]) -> None:
        self.present: dict[str, set[str]] = {
            kind: set(present.get(kind, []))
            for kind in ("deployment", "model", "job", "dataset", "project")
        }
        self.entry_of: dict[str, str] = {}
        """model version id -> entry id (a deleted entry takes its versions with it)."""
        self.call_log: list[tuple[str, str]] = []
        self.sticky: set[str] = set()
        """Ids whose delete succeeds but which a re-read still finds (a server bug to surface)."""
        self.running: set[str] = set()
        """Job ids the platform refuses to delete: it deletes terminal jobs only."""
        self.held_by_job: dict[str, str] = {}
        """dataset id -> job id: the live FK, a 409 for as long as that job exists."""
        self.dataset_error: DagnamError | None = None
        """Raised by ``delete_dataset`` whatever else is true (a server failure)."""

    def _take(self, kind: str, item_id: str, absent: type[DagnamError]) -> None:
        if item_id not in self.present[kind]:
            raise absent(item_id)
        if item_id not in self.sticky:
            self.present[kind].discard(item_id)

    def _need(self, kind: str, item_id: str, absent: type[DagnamError]) -> JsonObject:
        if item_id not in self.present[kind]:
            raise absent(item_id)
        return {"id": item_id}

    def get_deployment(self, deployment_id: str) -> JsonObject:
        self.call_log.append(("get_deployment", deployment_id))
        return self._need("deployment", deployment_id, DeploymentNotFoundError)

    def delete_deployment(self, deployment_id: str) -> JsonObject | None:
        self.call_log.append(("delete_deployment", deployment_id))
        self._take("deployment", deployment_id, DeploymentNotFoundError)
        return None

    def get_model_version(self, version_id: str) -> JsonObject:
        self.call_log.append(("get_model_version", version_id))
        entry = self.entry_of.get(version_id)
        if entry is None or entry not in self.present["model"]:
            raise ModelNotFoundError(version_id)
        return {"id": version_id, "entry_id": entry}

    def delete_model_entry(self, model_id: str) -> None:
        self.call_log.append(("delete_model_entry", model_id))
        self._take("model", model_id, ModelNotFoundError)

    def get_training_job(self, job_id: str) -> JsonObject:
        self.call_log.append(("get_training_job", job_id))
        return self._need("job", job_id, TrainingJobNotFoundError)

    def bulk_delete_training_jobs(self, job_ids: list[str]) -> JsonObject:
        """The real route answers 200 with a per-id ``errors`` list, never a 404."""
        self.call_log.append(("bulk_delete_training_jobs", ",".join(job_ids)))
        deleted = 0
        errors: JsonArray = []
        for job_id in job_ids:
            if job_id not in self.present["job"]:
                errors.append({"job_id": job_id, "error": "Not found or not authorized"})
            elif job_id in self.running:
                errors.append({"job_id": job_id, "error": "Cannot delete job with status running"})
            else:
                self._take("job", job_id, TrainingJobNotFoundError)
                deleted += 1
        return {"deleted": deleted, "errors": errors}

    def get_dataset_meta(self, dataset_id: str, version: str | None = None) -> JsonObject:
        self.call_log.append(("get_dataset_meta", dataset_id))
        return self._need("dataset", dataset_id, DatasetNotFoundError)

    def delete_dataset(self, dataset_id: str) -> None:
        self.call_log.append(("delete_dataset", dataset_id))
        if self.dataset_error is not None:
            raise self.dataset_error
        if self.held_by_job.get(dataset_id) in self.present["job"]:
            raise APIError(409, DATASET_IN_USE)
        self._take("dataset", dataset_id, DatasetNotFoundError)

    def get_project(self, project_id: str) -> JsonObject:
        self.call_log.append(("get_project", project_id))
        return self._need("project", project_id, ProjectNotFoundError)

    def delete_project(self, project_id: str) -> None:
        self.call_log.append(("delete_project", project_id))
        self._take("project", project_id, ProjectNotFoundError)


def as_cleanup_client(platform: FakeCleanup) -> CleanupClient:
    """The cleanup fake, typed as the protocol ``delete_audit`` takes."""
    return platform


def as_client(platform: FakePlatform) -> PlatformClient:
    """The fake, typed as the protocol the orchestrator takes (a static conformance check)."""
    return platform


def serve_chat(
    requests_mock: RequestsMocker, answer: Callable[[list[dict[str, str]]], str | None]
) -> list[dict[str, Any]]:
    """Serve the OpenAI-compatible route from ``answer(messages)``; ``None`` answers 500."""
    seen: list[dict[str, Any]] = []

    def respond(request: Any, context: Any) -> dict[str, Any]:
        body = json.loads(request.text)
        seen.append({"body": body, "authorization": request.headers.get("Authorization")})
        content = answer(body["messages"])
        if content is None:
            context.status_code = 500
            return {"error": "replica crashed"}
        return {"choices": [{"message": {"role": "assistant", "content": content}}]}

    requests_mock.post(CHAT_URL, json=respond)
    return seen


def last_user(messages: list[dict[str, str]]) -> str:
    """The content of the last ``user`` turn, the way the fake endpoint keys its answers."""
    return next(m["content"] for m in reversed(messages) if m["role"] == "user")


def teacher(messages: list[dict[str, str]]) -> str:
    """The teacher's answer for a fixture prompt: ``ticket N`` -> a/b, ``order N`` -> JSON."""
    kind, _, number = last_user(messages).partition(" ")
    if kind == "ticket":
        return "a" if int(number) % 2 else "b"
    return json.dumps({"order_id": number, "product": "x"})


def label_row(i: int) -> dict[str, Any]:
    """A ``labeled-example`` row exactly as Task 6 derives one."""
    turns = [{"role": "user", "content": f"ticket {i}"}]
    return {
        "input": render_chat_prompt(turns, system="Classify the ticket"),
        "label": teacher(turns),
    }


def json_row(i: int) -> dict[str, Any]:
    """A ``chat-messages`` row exactly as Task 6 derives one."""
    turns = [{"role": "system", "content": "Extract"}, {"role": "user", "content": f"order {i}"}]
    return {"messages": [*turns, {"role": "assistant", "content": teacher(turns)}]}


class Clock:
    """A clock the waits drive: ``sleep`` advances ``now`` and records the request."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds
