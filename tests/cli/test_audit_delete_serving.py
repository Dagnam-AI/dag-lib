"""CLI ``audit delete`` while an endpoint is serving: the refusal, its output, and the override."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from tests.audit import _receipts as r
from tests.audit._cleanup import FakeCleanup
from tests.cli._audit_dirs import platform_with_everything, published_dir

from dagnam._core.exceptions import EndpointsServingError
from dagnam._types import JsonObject
from dagnam.audit.cleanup import DELETED_FILE
from dagnam.audit.state import load_state

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture

SENTENCE = (
    "Nothing was deleted: this audit still has 2 serving endpoints: tickets-ft-small "
    "(3f2a0c1e-0000-4000-8000-000000000001), tickets-ft-base (9b1c0c1e-0000-4000-8000-000000000002). "
    "Deleting the audit would delete them, and any app calling them would start getting errors. "
    "Stop them first with `dagnam audit cancel <audit-dir>`, or Pause on each one's page under "
    "Deployments in the Studio, then delete again. "
    "To delete them anyway, send include_endpoints=true "
    "(`dagnam audit delete <audit-dir> --include-endpoints`, dagnam 0.18.0 or later). "
    "An endpoint that is resuming cannot be paused until it is running; "
    "wait for it to finish, then stop it. "
    "If it is stuck deploying, delete that endpoint from its own page under Deployments, "
    "then delete the audit again."
)
ENDPOINTS: list[JsonObject] = [
    {
        "id": "3f2a0c1e-0000-4000-8000-000000000001",
        "name": "tickets-ft-small",
        "status": "running",
        "last_request_at": "2026-10-07T14:03:11Z",
    },
    {
        "id": "9b1c0c1e-0000-4000-8000-000000000002",
        "name": "tickets-ft-base",
        "status": "deploying",
        "last_request_at": None,
    },
]


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)


@pytest.fixture
def published(tmp_path: Path) -> Path:
    return published_dir(tmp_path)


@pytest.fixture
def platform(monkeypatch: PytestMonkeyPatch) -> FakeCleanup:
    fake = platform_with_everything(
        receipt=r.designed(
            r.row("deployment", "dep-1", "deleted"), r.row("project", "proj-1", "deleted")
        )
    )
    fake.serving_refusal = EndpointsServingError(SENTENCE, ENDPOINTS)
    monkeypatch.setattr("dagnam.cli.audit_cleanup.client_from_env", lambda: fake)
    return fake


def _files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _refused(run_cli: CliRunner, *argv: str) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "delete", *argv])
    assert exc.value.code == 1


def test_the_refusal_says_it_once_lists_the_endpoints_and_names_the_next_commands(
    run_cli: CliRunner, published: Path, platform: FakeCleanup, capsys: StrCapture
) -> None:
    _refused(run_cli, str(published), "--yes")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("Nothing was deleted") == 1  # the server's sentence, once
    assert "Error: Nothing was deleted: this audit still has 2 serving endpoints" in captured.err
    err = " ".join(captured.err.split())
    assert (
        "tickets-ft-small 3f2a0c1e-0000-4000-8000-000000000001 serving "
        "last request 2026-10-07 14:03 UTC"
    ) in err
    assert (
        "tickets-ft-base 9b1c0c1e-0000-4000-8000-000000000002 resuming no requests recorded"
    ) in err
    assert f"Nothing in {published} was removed." in err
    assert f"dagnam audit cancel {published} stop every endpoint" in err
    assert f"dagnam audit delete {published} --include-endpoints delete the endpoints too" in err


def test_a_refusal_leaves_the_directory_the_state_and_the_keys_exactly_as_they_were(
    run_cli: CliRunner, published: Path, platform: FakeCleanup
) -> None:
    before = _files(published)
    state = load_state(published)

    _refused(run_cli, str(published), "--yes")

    assert _files(published) == before  # state.json byte for byte, rows, keys: nothing removed
    assert not (published / DELETED_FILE).exists()
    assert load_state(published) == state
    assert platform.call_log == [("delete_audit", "audit-1")]  # asked once; nothing destructive
    assert platform.include_endpoints == [False]
    assert platform.present["deployment"] == {"dep-1", "dep-2"}


def test_under_json_stdout_carries_the_refusal_object(
    run_cli: CliRunner, published: Path, platform: FakeCleanup, capsys: StrCapture
) -> None:
    _refused(run_cli, str(published), "--yes", "--json")

    out = json.loads(capsys.readouterr().out)
    assert out["error"] == SENTENCE
    assert out["code"] == "endpoints_serving"
    assert out["endpoints"] == ENDPOINTS
    assert f"dagnam audit cancel {published}" in out["hint"]
    assert "--include-endpoints" in out["hint"]


def test_one_endpoint_is_listed_alone_and_an_odd_row_is_shown_as_far_as_it_reads(
    run_cli: CliRunner, published: Path, platform: FakeCleanup, capsys: StrCapture
) -> None:
    platform.serving_refusal = EndpointsServingError(
        "Nothing was deleted: x.",
        [
            {"id": "d-1", "status": "running", "last_request_at": "yesterday"},
            {"id": "d-2", "name": "esc\x1b[31mape", "status": "running", "last_request_at": 7},
        ],
    )

    _refused(run_cli, str(published), "--yes")

    raw = capsys.readouterr().err
    err = " ".join(raw.split())
    assert "d-1 d-1 serving last request yesterday" in err  # no name: the id; no date: as sent
    assert "\x1b" not in raw  # what the platform sends cannot drive the terminal
    assert "no requests recorded" not in err


def test_the_prompt_says_what_the_flag_adds_only_when_it_is_given(
    run_cli: CliRunner, published: Path, platform: FakeCleanup, capsys: StrCapture
) -> None:
    with mock.patch("builtins.input", return_value="no"), pytest.raises(SystemExit):
        run_cli(["audit", "delete", str(published)])
    assert "--include-endpoints" not in capsys.readouterr().out

    with mock.patch("builtins.input", return_value="no"), pytest.raises(SystemExit):
        run_cli(["audit", "delete", str(published), "--include-endpoints"])
    out = capsys.readouterr().out
    assert "--include-endpoints: endpoints that are still serving are deleted too" in out
    assert "apps calling them will start getting errors" in out
    assert platform.call_log == []


def test_include_endpoints_sends_the_override_and_shows_what_was_serving(
    run_cli: CliRunner, published: Path, platform: FakeCleanup, capsys: StrCapture
) -> None:
    served: JsonObject = {
        **r.row("deployment", "dep-1", "deleted"),
        "was_serving": True,
        "added_later": [1],
    }
    platform.server_receipt = r.designed(served, r.row("project", "proj-1", "deleted"))

    assert run_cli(["audit", "delete", str(published), "--yes", "--include-endpoints"]) == 0

    assert platform.include_endpoints == [True]
    out = capsys.readouterr().out
    assert "deployment dep-1: deleted (was serving)" in out
    assert "project proj-1: deleted\n" in out
    assert load_state(published).halted == {"reason": "deleted"}


def test_an_unpublished_audit_with_a_serving_endpoint_is_refused_the_same_way(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    from tests.cli._audit_dirs import cli_state

    from dagnam.audit.state import save_state

    root = tmp_path / "local"
    save_state(root, cli_state())
    fake = platform_with_everything()
    fake.statuses = {"dep-1": "deploying", "dep-2": "running"}
    monkeypatch.setattr("dagnam.cli.audit_cleanup.client_from_env", lambda: fake)
    before = _files(root)

    _refused(run_cli, str(root), "--yes")

    err = " ".join(capsys.readouterr().err.split())
    assert "Nothing was deleted: 2 endpoints of this audit may still be serving" in err
    assert "dep-1 dep-1 deploying" in err  # not "resuming": this client cannot know it ever served
    assert "dep-2 dep-2 serving" in err
    assert "cannot be paused from here: wait for it to settle" in err
    assert "last request" not in err
    assert "no requests recorded" not in err
    assert {name for name, _ in fake.call_log} == {"get_deployment"}
    assert _files(root) == before


def test_the_flag_defaults_off_and_is_documented(run_cli: CliRunner, capsys: StrCapture) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "delete", "--help"])
    assert exc.value.code == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "--include-endpoints" in out
    assert "Also delete endpoints that are still serving" in out


def test_a_half_finished_delete_run_with_the_override_suggests_the_override(
    run_cli: CliRunner, published: Path, platform: FakeCleanup, capsys: StrCapture
) -> None:
    platform.serving_refusal = None
    platform.server_receipt = r.designed(r.row("deployment", "dep-1", "deleted"), status="halted")

    with pytest.raises(SystemExit):
        run_cli(["audit", "delete", str(published), "--yes", "--include-endpoints"])

    err = " ".join(capsys.readouterr().err.split())
    assert f"dagnam audit delete {published} --yes --include-endpoints" in err


def test_a_half_finished_delete_without_the_override_does_not_suggest_it(
    run_cli: CliRunner, published: Path, platform: FakeCleanup, capsys: StrCapture
) -> None:
    platform.serving_refusal = None
    platform.server_receipt = r.designed(r.row("deployment", "dep-1", "deleted"), status="halted")

    with pytest.raises(SystemExit):
        run_cli(["audit", "delete", str(published), "--yes"])

    err = " ".join(capsys.readouterr().err.split())
    assert f"dagnam audit delete {published} --yes" in err
    assert "--include-endpoints" not in err


def test_names_from_the_platform_stay_on_one_line_and_the_table_is_bounded(
    run_cli: CliRunner, published: Path, platform: FakeCleanup, capsys: StrCapture
) -> None:
    many: list[JsonObject] = [
        {"id": f"d-{i}", "name": f"n{i}\nfake row\tx", "status": "running"} for i in range(13)
    ]
    many[0]["name"] = "L" * 255
    platform.serving_refusal = EndpointsServingError("Nothing was deleted: many.", many)

    _refused(run_cli, str(published), "--yes")

    lines = capsys.readouterr().err.splitlines()
    rows = [line for line in lines if line.startswith("    ") and "d-" in line]
    assert len(rows) == 10  # capped
    assert any("(+3 more)" in line for line in lines)
    assert all("fake row" not in line or line.lstrip().startswith("n") for line in rows)
    assert not any(line.strip().startswith("fake row") for line in lines)  # no split row
    assert max(len(line) for line in rows) < 140  # a 255-character name is clipped
    assert any("L" * 37 + "..." in line for line in rows)
