"""CLI reports for the refusals a server sends about the caller or its plan.

A plan limit or credit shortfall (402), a deployment state or capacity
conflict (409), and the account-state and permission 403s: each report says
why the call was refused and what to do next. :mod:`dagnam.cli.errors`
dispatches to these by exception type.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from dagnam.cli.presentation import ErrorReport

if TYPE_CHECKING:
    from dagnam._core.exceptions import (
        APIError,
        DeploymentStateError,
        PermissionDeniedError,
        QuotaExceededError,
    )


def quota_report(exc: BaseException) -> ErrorReport:
    # The registry dispatches by isinstance, so exc IS a QuotaExceededError.
    if cast("QuotaExceededError", exc).required_credits is not None:
        return ErrorReport(
            title="not enough credits",
            fields=[("Detail", str(exc))],
            tries=[("", "Top up credits in the web app, then retry.")],
        )
    return ErrorReport(
        title="plan limit reached",
        fields=[("Detail", str(exc))],
        tries=[
            ("dagnam usage", "see your current usage and limits"),
            ("", "Free up resources or upgrade your plan to raise the limit."),
        ],
    )


def state_report(exc: BaseException) -> ErrorReport:
    # The registry dispatches by isinstance, so exc IS a DeploymentStateError.
    if cast("DeploymentStateError", exc).reason == "deployment_limit_reached":
        return ErrorReport(
            title="active deployment limit reached",
            fields=[("Detail", str(exc))],
            tries=[
                ("dagnam deployments list", "see your deployments; paused ones still count"),
                ("dagnam deployments delete <id>", "delete one you no longer need, then retry"),
            ],
        )
    return ErrorReport(
        title="action not allowed in the deployment's current state",
        fields=[("Detail", str(exc))],
        tries=[("dagnam deployments get <id>", "check the current lifecycle state")],
    )


def _detail(exc: BaseException) -> list[tuple[str, str]]:
    message = cast("APIError", exc).message
    return [("Detail", message)] if message else []


def _refusal_report(title: str, hint: str) -> Callable[[BaseException], ErrorReport]:
    """A fixed title and next step for an account-state refusal, plus its detail."""

    def build(exc: BaseException) -> ErrorReport:
        return ErrorReport(title=title, fields=_detail(exc), tries=[("", hint)])

    return build


email_report = _refusal_report(
    "your email address is not verified",
    "Open the verification email we sent you, then retry.",
)
suspended_report = _refusal_report(
    "this account is suspended",
    "Retrying will not help. Contact support to restore access.",
)
pending_deletion_report = _refusal_report(
    "this account is scheduled for deletion",
    "Sign in to the web app and cancel the deletion, then retry.",
)


_PLAN_GATE_REPORT = (
    "your plan does not include this",
    [
        ("dagnam usage", "see your plan and what it includes"),
        ("", "Upgrade your plan, or choose an option your plan includes."),
    ],
)
_PERMISSION_REPORTS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "insufficient_scope": (
        "your API key does not have the scope this action needs",
        [("dagnam whoami", "see which key is configured")],
    ),
    "api_key_not_allowed": (
        "this action needs a web session; API keys are not accepted",
        [("", "Do this in the web app instead.")],
    ),
}


def permission_report(exc: BaseException) -> ErrorReport:
    # The registry dispatches by isinstance, so exc IS a PermissionDeniedError.
    denied = cast("PermissionDeniedError", exc)
    title, tries = _PERMISSION_REPORTS.get(denied.reason, _PLAN_GATE_REPORT)
    if denied.reason == "insufficient_scope":
        tries = [*tries, ("", scope_hint(denied.required_scopes))]
    return ErrorReport(title=title, fields=_detail(exc), tries=tries)


def scope_hint(scopes: list[str] | None) -> str:
    """Say which scope the new key needs; the server accepts any one of them."""
    if not scopes:
        needed = "that scope"
    elif len(scopes) == 1:
        needed = f"the `{scopes[0]}` scope"
    else:
        needed = "one of the scopes " + ", ".join(f"`{scope}`" for scope in scopes)
    return f"Create a key with {needed} in the web app, then run `dagnam login` with it."
