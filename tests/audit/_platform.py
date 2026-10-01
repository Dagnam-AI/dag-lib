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

from dagnam._core.exceptions import APIError
from dagnam._types import JsonArray, JsonObject
from dagnam.audit.steps import PlatformClient

CHAT_URL = "https://x/v1/chat/completions"


class PublishLeak(BaseException):
    """An audit route reached under ``--local-only``.

    A ``BaseException`` on purpose: the publisher swallows every ``Exception``
    so that a publish failure can never end a run, which would swallow this
    tripwire too and let a leak pass as a green test.
    """


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
        self.analysis: JsonObject = {"analysis_status": "completed", "analysis_error": None}
        """The dataset row's processing state, as ``get_dataset`` reports it."""
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
        self.resumes: list[tuple[str, bool]] = []
        """(route, K1 ``resume`` flag) per audit create / candidate / patch, in order."""
        self.audit_halted = False
        """The account shows the audit halted (a website cancel, or a halt a run published).

        A ``resume: false`` publish is then a 409, and ``get_audit`` says ``halted``.
        """
        self.audit_deleted = False
        """The audit was deleted in the account: every audit route answers the uniform 404."""
        self.resume_route = True
        """K1b: the platform has ``POST /audits/{id}/resume``; ``False`` is an older one (404)."""
        self.cancelled: list[str] = []
        self.deleted_audits: list[str] = []
        self.publish_errors: dict[str, list[BaseException]] = {}
        """Per method name: the exception the next call raises, popped in order."""
        self.forbid_publishing = False
        """``--local-only``: any audit route is a test failure, not a recorded call."""
        self.forbidden_attempts: list[str] = []
        """Audit routes reached while ``forbid_publishing``; a leak, recorded before it raises."""
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

    def get_dataset(self, dataset_id: str) -> JsonObject:
        self._log("get_dataset")
        return {"id": dataset_id, **self.analysis}

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
        """The run read (K2): a completed run names the version it pushed, unless it pushed none."""
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
            **({"model_version_id": "mv-pushed"} if self.pushes_version else {}),
            **self.run_extra,
        }

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
        """Log one audit route, refusing it outright under ``forbid_publishing``.

        The attempt is recorded BEFORE it raises, so a test can see a leak the
        publisher caught and discarded rather than only one that escaped.
        """
        if self.forbid_publishing:
            self.forbidden_attempts.append(name)
            raise PublishLeak(f"{name} must not be called")
        self._log(name)
        errors = self.publish_errors.get(name)
        if errors:
            raise errors.pop(0)

    def _resume(self, name: str, payload: JsonObject) -> None:
        """K1: every publish says whether it resumes; only a resuming one gets past a halt."""
        self._gone()
        resume = bool(payload.pop("resume"))
        self.resumes.append((name, resume))
        if self.audit_halted and not resume:
            raise APIError(409, "audit is halted")
        self.audit_halted = False

    def _gone(self) -> None:
        if self.audit_deleted:
            raise APIError(404, "Audit not found")

    def get_audit(self, audit_id: str) -> JsonObject:
        self._publish("get_audit")
        self._gone()
        return {"id": audit_id, "status": "halted" if self.audit_halted else "running"}

    def resume_audit(self, audit_id: str) -> JsonObject:
        """K1b: un-halt the audit; an older platform has no such route."""
        self._publish("resume_audit")
        if not self.resume_route:
            raise APIError(404, "Not Found")
        self._gone()
        self.audit_halted = False
        return {"id": audit_id, "status": "running"}

    def create_audit(self, payload: JsonObject) -> JsonObject:
        self._publish("create_audit")
        self._resume("create_audit", payload)
        self.audits.append(payload)
        return {"id": self._next("audit")}

    def create_audit_candidate(self, audit_id: str, payload: JsonObject) -> JsonObject:
        self._publish("create_audit_candidate")
        self._resume("create_audit_candidate", payload)
        self.candidates.append((audit_id, payload))
        return {"id": self._next("cand")}

    def patch_audit_candidate(
        self, audit_id: str, candidate_id: str, payload: JsonObject
    ) -> JsonObject:
        self._publish("patch_audit_candidate")
        self._resume("patch_audit_candidate", payload)
        self.patches.append((candidate_id, payload))
        return {"id": candidate_id, "status": payload["status"]}

    def halt_audit(self, audit_id: str, reason: str) -> JsonObject:
        self._publish("halt_audit")
        self._gone()
        self.halts.append((audit_id, reason))
        self.audit_halted = True
        return {"id": audit_id, "status": "halted", "halted_reason": reason}

    def cancel_audit(self, audit_id: str) -> JsonObject:
        self._publish("cancel_audit")
        self.cancelled.append(audit_id)
        return dict(self.receipt)

    def delete_audit(self, audit_id: str) -> JsonObject:
        self._publish("delete_audit")
        self.deleted_audits.append(audit_id)
        return dict(self.receipt)


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
