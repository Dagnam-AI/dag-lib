"""CLI usage subcommand."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, StrCapture


# ---------------------------------------------------------------- usage


def test_usage_table(run_cli: CliRunner, capsys: StrCapture) -> None:
    snapshot = {
        "plan": {"code": "pro", "display_name": "Pro"},
        "read_only_grace": False,
        "limits": [
            {"key": "concurrent_training_jobs", "current": 1, "limit": 3},
            {"key": "training_minutes", "current": 10, "limit": None},
        ],
    }
    fake = SimpleNamespace(entitlements=mock.Mock(return_value=snapshot))
    with mock.patch("dagnam.account", fake):
        run_cli(["usage"])
    out = capsys.readouterr().out
    assert "Plan: Pro" in out
    assert "concurrent_training_jobs" in out
    assert "training_minutes" in out


def test_usage_json(run_cli: CliRunner, capsys: StrCapture) -> None:
    snapshot = {"plan": {"code": "free"}, "limits": []}
    fake = SimpleNamespace(entitlements=mock.Mock(return_value=snapshot))
    with mock.patch("dagnam.account", fake):
        run_cli(["usage", "--json"])
    assert json.loads(capsys.readouterr().out) == snapshot


def test_usage_apierror_exits(run_cli: CliRunner) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(entitlements=mock.Mock(side_effect=APIError(500, "boom")))
    with mock.patch("dagnam.account", fake):
        with pytest.raises(SystemExit):
            run_cli(["usage"])


def test_usage_read_only_grace_and_pending_plan(run_cli: CliRunner, capsys: StrCapture) -> None:
    snapshot = {
        "plan": {"display_name": "Pro"},
        "read_only_grace": True,
        "pending_plan": "enterprise",
        "limits": [{"key": "jobs", "current": 2, "limit": 5}],
    }
    fake = SimpleNamespace(entitlements=mock.Mock(return_value=snapshot))
    with mock.patch("dagnam.account", fake):
        run_cli(["usage"])
    out = capsys.readouterr().out
    assert "READ-ONLY GRACE" in out
    assert "Pending plan: enterprise" in out
    assert "Remaining" in out


def test_usage_no_limits_returns_message(run_cli: CliRunner, capsys: StrCapture) -> None:
    snapshot = {"plan": {"code": "free"}, "limits": []}
    fake = SimpleNamespace(entitlements=mock.Mock(return_value=snapshot))
    with mock.patch("dagnam.account", fake):
        run_cli(["usage"])
    assert "No limit information returned." in capsys.readouterr().out
