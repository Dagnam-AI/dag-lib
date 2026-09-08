"""Mirror a local audit into the account as it runs (spec section 10, Task 11).

The run stays the source of truth: everything here is a best-effort echo of
what ``state.json`` already records, so the website can show a run in progress
without the run depending on the website. Every call is wrapped -- an
``APIError`` is logged and the body queued to go out with the next one -- and
:meth:`Publisher.halt` is the only thing that ever tells the server a run
stopped. ``dagnam audit run --local-only`` builds the publisher with
``client=None`` and every method returns before it can open a socket.

The server's candidate lifecycle is a state machine
(``uploading -> splitting -> pii_check -> submitting -> training -> deploying
-> replaying -> scored``), so :data:`STATUS_BY_STEP` maps each step of
``orchestrate.STEPS`` onto it in order; a step that recorded an error
publishes ``failed`` (or ``pii_disagreement``) instead, which every state
allows.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from importlib.metadata import PackageNotFoundError, version
import json
import logging
from pathlib import Path
from typing import Any

from dagnam._core.exceptions import APIError
from dagnam._types import JsonObject
from dagnam.audit.economics import serving_cost_usd_month
from dagnam.audit.state import AuditState, StepState
from dagnam.audit.steps import PlatformClient, StepContext, error_code

_LOGGER = logging.getLogger("dagnam.audit.publish")

SOURCE = "cli"
"""``AuditCreate.source``: this audit was run by the CLI, not started on the site."""
SCORED_BY = "cli"
UNKNOWN_VERSION = "0+unknown"
"""What a source checkout with no installed distribution reports as its version."""
MEASURED_FROM = "client"
"""The replay's percentiles are the client's own round trips, not the server's."""

STATUS_BY_STEP: Mapping[str, str] = {
    "upload": "uploading",
    "resolve_version": "uploading",
    "split": "splitting",
    "wait_split": "splitting",
    "pii_scan": "pii_check",
    "wait_pii": "pii_check",
    "submit": "submitting",
    "wait_run": "training",
    "resolve_model_version": "training",
    "create_deployment": "deploying",
    "create_revision": "deploying",
    "wait_active": "deploying",
    "replay_and_score": "scored",
}
"""Every step of ``orchestrate.STEPS`` -> the candidate status it leaves behind."""

REPLAY_STEP = "replay"
"""Published just before ``replay_and_score``: the server has no ``deploying -> scored`` edge."""
PII_DISAGREEMENT = "pii_disagreement"
FAILED = "failed"

DROP_STATUSES = frozenset({400, 422})
"""A body this client should never have sent: logged once with the reason, never resent."""
MAX_PENDING = 20
"""Cap on the resend queue; the oldest unsent step is dropped rather than grow it forever."""

_EXCERPT_MAX = 200
_ERROR_MAX = 500
_REASON_MAX = 300
_ID_MAX = 64
_NAME_MAX = 120
_VERSION_MAX = 32


def installed_version() -> str:
    """The installed ``dagnam`` version, or a marker when the package is not installed."""
    try:
        return version("dagnam")
    except PackageNotFoundError:
        return UNKNOWN_VERSION


def _number(value: object) -> float | None:
    """``value`` as a float when it is a real number, else ``None``."""
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _mean(total: object, calls: object) -> int | None:
    """A per-call mean from the scan's totals; the server stores means, not totals."""
    amount, over = _number(total), _number(calls)
    return None if amount is None or not over else round(amount / over)


def _sub(entry: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = entry.get(key)
    return value if isinstance(value, Mapping) else {}


def _pii_counts(audit_dir: Path, workload_id: str) -> dict[str, int]:
    """The redaction counts Task 6 wrote for this workload; empty when it derived none."""
    meta = audit_dir / "workloads" / workload_id / "meta.json"
    if not meta.is_file():
        return {}
    counts = _sub(_sub(json.loads(meta.read_text(encoding="utf-8")), "stats"), "redact").get(
        "counts"
    )
    return {str(code): int(n) for code, n in counts.items()} if isinstance(counts, Mapping) else {}


def workload_body(
    entry: Mapping[str, Any], *, selected: bool, pii_counts: Mapping[str, int]
) -> JsonObject:
    """One ``scan-report.json`` workload as the publish body's ``WorkloadPublish``."""
    verdict = _sub(entry, "verdict")
    tokens = _sub(entry, "tokens")
    return {
        "workload_id": str(entry["id"])[:_ID_MAX],
        "structure_class": str(entry["structure_class"]),
        "template_excerpt": str(entry.get("template_excerpt") or "")[:_EXCERPT_MAX],
        "calls_per_day": _number(entry.get("calls_per_day")) or 0.0,
        "mean_prompt_tokens": _mean(tokens.get("prompt"), entry.get("calls")),
        "mean_completion_tokens": _mean(tokens.get("completion"), entry.get("calls")),
        "spend_usd_month": _number(entry.get("cost_usd_month")),
        "export_p50_ms": _number(_sub(entry, "latency_ms").get("p50")),
        "verdict": str(verdict.get("status") or "unknown_cost"),
        "verdict_reason": str(verdict.get("reason") or "")[:_REASON_MAX],
        "ratio": _number(verdict.get("ratio")),
        "pii_counts": dict(pii_counts),
        "selected": selected,
    }


def status_for(step_name: str, step: StepState) -> str:
    """The status a step leaves the candidate in; a recorded error wins over the map."""
    if step.error is not None and not step.scored:
        return PII_DISAGREEMENT if error_code(step) == PII_DISAGREEMENT else FAILED
    return STATUS_BY_STEP[step_name]


class Publisher:
    """Echoes one audit into the account. Never raises: a publish failure is a warning.

    ``client=None`` (``--local-only``) makes every method a no-op that opens no
    socket, and so does a run whose ``create_audit`` never succeeded: there is
    nothing to attach a candidate or a step to.
    """

    def __init__(self, client: PlatformClient | None, state: AuditState) -> None:
        self._client = client
        self._state = state
        self._entries: dict[str, Mapping[str, Any]] = {}
        """Workload id -> its scan entry, for the serving cost a scored candidate carries."""
        self._pending: list[tuple[str, JsonObject]] = []
        """(candidate id, body) patches a failed call left to resend, oldest first."""

    # -- the audit ------------------------------------------------------------

    def follow(self, state: AuditState) -> None:
        """Mirror ``state`` from here on.

        ``run_audit`` loads the state from disk itself, so the object the CLI
        built this publisher with is not the one the run will save; this points
        the publisher at the live one before the first publish.
        """
        self._state = state

    def start(
        self,
        audit_dir: Path,
        scan: Mapping[str, Any],
        selected: Collection[str],
        *,
        floor: float,
        max_credits: int,
        sdk_version: str,
    ) -> None:
        """Publish the scan (once) and remember its workloads; sets ``state.audit_id``.

        A resumed run calls this too -- the workload numbers a scored candidate
        needs are read here -- but returns before the POST once the state
        already carries an ``audit_id``.
        """
        entries = [e for e in scan.get("workloads", []) if isinstance(e, Mapping) and "id" in e]
        self._entries = {str(entry["id"]): entry for entry in entries}
        client = self._client
        if client is None or self._state.audit_id is not None or not entries:
            return

        def body() -> JsonObject:
            # Built inside the guarded call: a scan report this client cannot
            # turn into a body is a warning like any other publish failure.
            return {
                "project_id": self._state.project_id,
                "source": SOURCE,
                "floor": floor,
                "max_credits": max_credits,
                "price_table_version": str(scan.get("price_table_version") or "unknown")[
                    :_VERSION_MAX
                ],
                "scan_generated_at": scan.get("generated_at"),
                "sdk_version": sdk_version[:_VERSION_MAX],
                "local_dir_name": audit_dir.resolve().name[:_NAME_MAX],
                "workloads": [
                    workload_body(
                        entry,
                        selected=str(entry["id"]) in selected,
                        pii_counts=_pii_counts(audit_dir, str(entry["id"])),
                    )
                    for entry in entries
                ],
            }

        created = self._call(client, "create_audit", lambda c: c.create_audit(body()))
        if created is not None:
            self._state.audit_id = str(created["id"])

    def halt(self, reason: str) -> None:
        """Tell the account the run stopped short, and why."""
        target = self._target()
        if target is None:
            return
        client, audit_id = target
        self._flush(client, audit_id)
        self._call(client, "halt_audit", lambda c: c.halt_audit(audit_id, reason))

    # -- candidates -----------------------------------------------------------

    def candidate(self, ctx: StepContext, step: StepState) -> None:
        """Open this candidate on the server once; the id is remembered in the state."""
        target = self._target()
        if target is None or step.published_candidate_id is not None:
            return
        client, audit_id = target
        body: JsonObject = {
            "workload_id": ctx.workload_id[:_ID_MAX],
            "kind": ctx.spec.kind.value,
            "recipe_key": ctx.spec.recipe_key,
        }
        created = self._call(
            client, "create_audit_candidate", lambda c: c.create_audit_candidate(audit_id, body)
        )
        if created is not None:
            step.published_candidate_id = str(created["id"])

    def step(self, ctx: StepContext, step_name: str, step: StepState) -> None:
        """Record one step of a candidate, resending anything an earlier call could not."""
        target = self._target()
        if target is None:
            return
        self.candidate(ctx, step)
        candidate_id = step.published_candidate_id
        if candidate_id is None:
            return
        if step_name == "replay_and_score":
            self._queue(candidate_id, {"step": REPLAY_STEP, "status": "replaying"})
        self._queue(candidate_id, self._patch(ctx, step_name, step))
        self._flush(*target)

    def _patch(self, ctx: StepContext, step_name: str, step: StepState) -> JsonObject:
        """The ``CandidatePatch`` body for one step: its status plus whatever the step produced."""
        body: JsonObject = {"step": step_name, "status": status_for(step_name, step)}
        if step.error is not None:
            body["error"] = step.error[:_ERROR_MAX]
        for key, value in (
            ("dataset_version_id", step.version_id),
            ("training_job_id", step.training_job_id),
            ("deployment_id", step.deployment_id),
            ("base_display_name", None if step.base is None else step.base[:_NAME_MAX]),
            ("training_cost_credits", step.training_cost_credits),
            ("replay_cost_credits", step.replay_cost_credits),
        ):
            if value is not None:
                body[key] = value
        if step.agreement is not None:
            body["agreement"] = dict(step.agreement)
        if step.latency is not None:
            body["latency"] = {**step.latency, "measured_from": MEASURED_FROM}
        if step.scored:
            body["scored_by"] = SCORED_BY
            cost = self._serving_cost(ctx)
            if cost is not None:
                body["serving_cost_usd_month"] = cost
        return body

    def _serving_cost(self, ctx: StepContext) -> float | None:
        """What serving this candidate costs a month at the workload's own volume."""
        entry = self._entries.get(ctx.workload_id)
        if entry is None or ctx.spec.serving_rate_key is None:
            return None
        tokens = _sub(entry, "tokens")
        return serving_cost_usd_month(
            ctx.spec.serving_rate_key,
            calls_per_day=_number(entry.get("calls_per_day")) or 0.0,
            completion_tokens=int(_number(tokens.get("completion")) or 0),
            calls=int(_number(entry.get("calls")) or 1) or 1,
        )

    # -- transport ------------------------------------------------------------

    def _target(self) -> tuple[PlatformClient, str] | None:
        """The client and audit to publish against, or ``None`` when there is nothing to publish to."""
        client, audit_id = self._client, self._state.audit_id
        return None if client is None or audit_id is None else (client, audit_id)

    def _queue(self, candidate_id: str, body: JsonObject) -> None:
        self._pending.append((candidate_id, body))
        if len(self._pending) > MAX_PENDING:
            dropped = self._pending.pop(0)
            _LOGGER.warning("audit publish: dropping the unsent step %r", dropped[1].get("step"))

    def _flush(self, client: PlatformClient, audit_id: str) -> None:
        """Send the queued patches in order, stopping at the first that will not go.

        Order matters -- the server's transitions are monotonic -- so a failure
        leaves everything behind it queued for the next call rather than
        letting a later step overtake an earlier one. A body the server calls a
        client bug is dropped instead: resending it would only fail again.
        """
        while self._pending:
            candidate_id, body = self._pending[0]
            try:
                client.patch_audit_candidate(audit_id, candidate_id, body)
            except Exception as exc:
                _LOGGER.warning(
                    "audit publish: step %r failed (%s); the run continues", body["step"], exc
                )
                if not isinstance(exc, APIError) or exc.status_code not in DROP_STATUSES:
                    return
            self._pending.pop(0)

    def _call(
        self, client: PlatformClient, what: str, send: Callable[[PlatformClient], JsonObject]
    ) -> JsonObject | None:
        """Run one publish call; a failure is a warning, never the end of the run."""
        try:
            return send(client)
        except Exception as exc:
            # Publishing is an echo of what the run already recorded on disk, so
            # nothing it can go wrong at -- a refusal, a dead network, a body
            # this client should not have built -- is worth ending a run over.
            _LOGGER.warning("audit publish: %s failed (%s); the run continues", what, exc)
            return None


__all__ = [
    "MAX_PENDING",
    "MEASURED_FROM",
    "REPLAY_STEP",
    "SCORED_BY",
    "SOURCE",
    "STATUS_BY_STEP",
    "UNKNOWN_VERSION",
    "Publisher",
    "installed_version",
    "status_for",
    "workload_body",
]
