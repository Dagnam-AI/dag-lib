"""CLI usage subcommand."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

from dagnam.cli import account as account_mod

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, StrCapture


# ---------------------------------------------------------------- usage


def test_usage_format_helpers_cover_units_and_edge_cases() -> None:
    assert account_mod._format_usage_value("storage.bytes", 1024**2) == "1.0 MB"
    assert account_mod._format_usage_value("storage.bytes", 2 * 1024**4) == "2.0 TB"
    assert account_mod._format_usage_value("jobs", 1_200) == "1.2K"
    assert account_mod._format_usage_value("records", 2_500_000) == "2.5M"
    assert account_mod._format_usage_value(123, 5) == "5"
    assert account_mod._format_usage_value("bad", "unknown") == "-"
    # Known non-byte metrics carry a short unit suffix; unknown keys stay bare.
    assert account_mod._format_usage_value("projects.count", 37) == "37 projects"
    assert account_mod._format_usage_value("training.concurrent_jobs", 3) == "3 jobs"
    assert account_mod._format_usage_value("models.max_parameters", 1_000_000_000) == "1B params"
    assert account_mod._limit_unit("deployments.count") == "deploys"
    assert account_mod._limit_unit("unknown.key") == ""
    assert account_mod._limit_unit(123) == ""
    assert account_mod._limit_label("training.concurrent_jobs") == "concurrent training jobs"
    assert account_mod._limit_label("custom.quota_name") == "custom quota name"
    assert account_mod._limit_label(None) == "-"
    assert account_mod._remaining_bar(5, 0) == "----------"
    assert account_mod._remaining_percent(5, 0) == "0%"
    assert account_mod._remaining_bar("n/a", 10) == "-"
    assert account_mod._remaining_percent("n/a", 10) == "-"


def test_usage_table(run_cli: CliRunner, capsys: StrCapture) -> None:
    snapshot = {
        "plan": {"code": "pro", "display_name": "Pro"},
        "read_only_grace": False,
        "limits": [
            {"key": "storage.bytes", "current": 385, "limit": 107374182400},
            {"key": "storage.max_upload_bytes", "current": 0, "limit": 10737418240},
            {"key": "models.max_parameters", "current": 0, "limit": 1_000_000_000},
            {"key": "projects.version_retention", "current": 13, "limit": 50},
            {"key": "training_minutes", "current": 10, "limit": None},
        ],
    }
    fake = SimpleNamespace(entitlements=mock.Mock(return_value=snapshot))
    with mock.patch("dagnam.account", fake):
        run_cli(["usage"])
    out = capsys.readouterr().out
    assert "Plan: Pro" in out
    assert "Limit type" in out
    assert "Meter" in out
    assert "Available %" in out
    assert "Available  Available %" not in out
    assert "storage" in out
    assert "385 B" in out
    assert "100.0 GB" in out
    assert "max upload size" in out
    assert "10.0 GB" in out
    assert "max model parameters" in out
    assert "1B" in out
    assert "1B params" in out
    assert "project versions retained" in out
    assert "37" in out
    assert "37 versions" in out
    assert "training minutes" in out
    assert "Remaining" in out
    assert "##########" in out
    assert "#######---" in out
    assert "74%" in out
    assert "storage.bytes" not in out
    assert "storage.max_upload_bytes" not in out
    assert "models.max_parameters" not in out
    assert "projects.version_retention" not in out
    assert "107374182400" not in out
    assert "1000000000" not in out


def test_usage_json(run_cli: CliRunner, capsys: StrCapture) -> None:
    snapshot = {"plan": {"code": "free"}, "limits": []}
    fake = SimpleNamespace(entitlements=mock.Mock(return_value=snapshot))
    with mock.patch("dagnam.account", fake):
        run_cli(["usage", "--json"])
    assert json.loads(capsys.readouterr().out) == snapshot


def test_usage_apierror_exits(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(entitlements=mock.Mock(side_effect=APIError(500, "boom")))
    with mock.patch("dagnam.account", fake):
        assert run_cli(["usage"]) == 1
    err = capsys.readouterr().err
    assert "Error: the Dagnam API had an internal error (HTTP 500)" in err
    assert "boom" in err


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
