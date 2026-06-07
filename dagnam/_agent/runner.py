"""Standalone helpers the Agent Skill scripts invoke.

These functions print to stdout (they back the ``skill/scripts/*.py`` shims an
agent runs and reads), so this module is exempt from the T20 (print) lint rule.
It must not import ``dagnam.cli`` -- see the package docstring. ``dagnam`` itself
is imported lazily inside the functions (matching the CLI modules) so importing
this module for ``--help`` stays fast and never triggers the heavy SDK import.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from typing import Any

# Costly / irreversible / public actions the guardrail previews before they run.
_COSTLY_ACTIONS = ("deploy", "train", "publish", "delete")
# Actions that spend metered resources, so the preview surfaces live plan/usage.
_SPEND_ACTIONS = ("deploy", "train")

# Terminal SSE event names emitted by ``dagnam.stream_training`` (see
# ``dagnam._core.sse.TERMINAL_TRAINING_EVENTS``).
_TERMINAL_OK = "complete"
_TERMINAL_FAIL = frozenset({"failed", "cancelled"})


def _entitlement_lines() -> list[str]:
    """Return best-effort live plan/usage context; never raise into plan output."""
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        snapshot = dagnam.account.entitlements()
    except DagnamError:
        return ["  (could not fetch entitlements - run `dagnam login` to see plan/usage impact)"]
    plan = snapshot.get("plan")
    plan_name = (
        plan.get("display_name", plan.get("code", "unknown"))
        if isinstance(plan, dict)
        else "unknown"
    )
    lines = [f"  plan: {plan_name}"]
    limits = snapshot.get("limits")
    if isinstance(limits, list):
        for item in limits:
            if isinstance(item, dict):
                lines.append(
                    f"  {item.get('key', '?')}: {item.get('current', '?')}/{item.get('limit', 'unlimited')}"
                )
    return lines


def plan_preview(action: str, params: Mapping[str, Any]) -> int:
    """Print a dry-run execution plan for a costly action; never execute it.

    Returns 0. Raises ``ValueError`` for an unrecognized action so the caller (and
    the agent) fails loudly rather than silently skipping the guardrail.
    """
    if action not in _COSTLY_ACTIONS:
        raise ValueError(
            f"Unsupported action for plan preview: {action!r} (expected one of {_COSTLY_ACTIONS})"
        )

    print(f"=== DRY RUN: {action} ===")
    print("This is a preview. Nothing was executed.")
    print("Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    if action in _SPEND_ACTIONS:
        print("Plan/usage impact:")
        for line in _entitlement_lines():
            print(line)
    print()
    print(
        f"To proceed, confirm with the user, then run the real `{action}` call. Nothing was executed."
    )
    return 0


def plan_main(argv: Sequence[str] | None = None) -> int:
    """CLI entry for ``scripts/plan.py``: ``--action deploy --param key=value ...``."""
    parser = argparse.ArgumentParser(
        prog="dagnam-plan", description="Preview a costly Dagnam action (dry run)."
    )
    parser.add_argument("--action", required=True, choices=_COSTLY_ACTIONS)
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable action parameter.",
    )
    args = parser.parse_args(argv)
    params: dict[str, str] = {}
    for raw in args.param:
        key, _, value = raw.partition("=")
        params[key] = value
    return plan_preview(args.action, params)


def watch_training(job_id: str) -> int:
    """Stream a training job's SSE events, print a compact summary, return an exit code.

    Returns 0 on the ``complete`` terminal event, 1 on ``failed``/``cancelled``, and
    0 if the stream ends without a terminal event (the job may still be running; the
    agent re-checks).
    """
    import dagnam

    seen = 0
    for event in dagnam.stream_training(job_id):
        name = getattr(event, "event", "")
        raw = getattr(event, "data", None)
        data = raw if isinstance(raw, dict) else {}
        seen += 1
        if name == "metric":
            print(f"[metric] {data.get('name')}={data.get('value')} (step {data.get('step')})")
        elif name == _TERMINAL_OK:
            print(f"[done] training complete for {job_id}")
            return 0
        elif name in _TERMINAL_FAIL:
            print(f"[error] training {name} for {job_id}: {data.get('error', 'unknown')}")
            return 1
    print(
        f"[info] stream ended after {seen} event(s) with no terminal event; job may still be running."
    )
    return 0


def watch_main(argv: Sequence[str] | None = None) -> int:
    """CLI entry for ``scripts/watch_training.py``: ``<job_id>``."""
    parser = argparse.ArgumentParser(
        prog="dagnam-watch", description="Stream a training job to completion."
    )
    parser.add_argument("job_id", help="The training job id to stream.")
    args = parser.parse_args(argv)
    return watch_training(args.job_id)
