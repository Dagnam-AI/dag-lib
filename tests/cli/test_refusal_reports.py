"""CLI reports for server refusals: quota, credits, state, account and permission."""

from __future__ import annotations

import pytest

from dagnam._core.exceptions import (
    AccountPendingDeletionError,
    AccountSuspendedError,
    DeploymentStateError,
    EmailNotVerifiedError,
    PermissionDeniedError,
    QuotaExceededError,
)
from dagnam.cli import errors as errors_mod, refusal_reports


def test_quota_error_suggests_usage() -> None:
    out = errors_mod.render_error(QuotaExceededError("Storage quota exceeded"))
    assert "Error: plan limit reached" in out
    assert "dagnam usage" in out


def test_deployment_state_error() -> None:
    out = errors_mod.render_error(DeploymentStateError("cannot stop a stopped deployment"))
    assert "current state" in out
    assert "dagnam deployments get <id>" in out


def test_deployment_limit_suggests_listing_and_deleting() -> None:
    exc = DeploymentStateError(
        "Active deployment limit reached.", reason="deployment_limit_reached"
    )
    out = errors_mod.render_error(exc)
    assert "Error: active deployment limit reached" in out
    assert "dagnam deployments list" in out
    assert "dagnam deployments delete <id>" in out
    assert "paused" in out  # paused deployments still count toward the limit
    assert "Active deployment limit reached." in out
    assert "current state" not in out
    assert "Stop one" not in out  # there is no stop command


def test_quota_error_for_credits_suggests_top_up() -> None:
    exc = QuotaExceededError(
        "Not enough credits. Required: 120 credits; available: 40.",
        required_credits=120,
        available_credits=40,
    )
    out = errors_mod.render_error(exc)
    assert "Error: not enough credits" in out
    assert "Top up" in out
    assert "Required: 120 credits; available: 40." in out
    assert "plan limit reached" not in out
    assert "upgrade your plan" not in out


# ------------------------------------------------- 403 refusals get their own hint


def test_email_not_verified_says_verify_not_plan() -> None:
    out = errors_mod.render_error(EmailNotVerifiedError("Email address is not verified."))
    assert "Error: your email address is not verified" in out
    assert "Email address is not verified." in out
    assert "verif" in out.split("Try:")[1]
    assert "plan or role" not in out


def test_account_suspended_says_contact_support() -> None:
    out = errors_mod.render_error(AccountSuspendedError("This account has been suspended."))
    assert "Error: this account is suspended" in out
    assert "support" in out
    assert "plan or role" not in out


def test_account_pending_deletion_says_cancel_in_web_app() -> None:
    out = errors_mod.render_error(AccountPendingDeletionError("Scheduled for deletion."))
    assert "Error: this account is scheduled for deletion" in out
    assert "cancel" in out
    assert "plan or role" not in out


@pytest.mark.parametrize(
    ("scopes", "phrase"),
    [
        (["write"], "Create a key with the `write` scope"),
        (["read", "read:datasets"], "Create a key with one of the scopes `read`, `read:datasets`"),
    ],
)
def test_insufficient_scope_names_the_required_scopes(scopes: list[str], phrase: str) -> None:
    exc = PermissionDeniedError("No.", reason="insufficient_scope", required_scopes=scopes)
    out = errors_mod.render_error(exc)
    assert phrase in out
    assert "that scope" not in out


def test_insufficient_scope_says_create_a_key_with_the_scope() -> None:
    exc = PermissionDeniedError("Needs the write scope.", reason="insufficient_scope")
    out = errors_mod.render_error(exc)
    assert "Error: your API key does not have the scope this action needs" in out
    assert "Needs the write scope." in out
    assert "dagnam login" in out
    assert "plan or role" not in out


def test_api_key_not_allowed_says_use_the_web_app() -> None:
    exc = PermissionDeniedError("Needs a web session.", reason="api_key_not_allowed")
    out = errors_mod.render_error(exc)
    assert "Error: this action needs a web session; API keys are not accepted" in out
    assert "web app" in out
    assert "dagnam login" not in out  # a new key will not help


@pytest.mark.parametrize(
    "reason", ["feature_gated", "remote_gpu_not_entitled", "compute_strategy_not_in_tier"]
)
def test_plan_gates_say_upgrade(reason: str) -> None:
    out = errors_mod.render_error(PermissionDeniedError("Not on your plan.", reason=reason))
    assert "Error: your plan does not include this" in out
    assert "dagnam usage" in out
    assert "Upgrade" in out


def test_refusal_without_a_server_message_renders_no_detail_field() -> None:
    out = errors_mod.render_error(AccountSuspendedError(""))
    assert "Error: this account is suspended" in out
    assert "Detail" not in out


@pytest.mark.parametrize(
    ("scopes", "needed"),
    [(None, "that scope"), ([], "that scope"), (["write"], "the `write` scope")],
)
def test_scope_hint_names_what_the_key_needs(scopes: list[str] | None, needed: str) -> None:
    assert refusal_reports.scope_hint(scopes) == (
        f"Create a key with {needed} in the web app, then run `dagnam login` with it."
    )
