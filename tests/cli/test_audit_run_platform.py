"""CLI ``audit run`` asks the platform which contract it runs before anything is uploaded."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from tests.audit._platform import FakePlatform
from tests.cli._audit_run import Build

from dagnam._core.exceptions import APIError, ResponseError
from dagnam._types import JsonObject

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)


@pytest.mark.parametrize(
    ("answer", "says"),
    [
        # A platform one contract minor behind: both versions are named.
        (
            {"revision": "abc", "version": "1", "contracts": "0.3.1"},
            "runs dagnam-contracts 0.3.1 and this SDK runs 0.4.2",
        ),
        # A platform from before the key: exactly what it answers today.
        (
            {"revision": "abc", "version": "1"},
            "does not report its dagnam-contracts version and this SDK runs 0.4.2",
        ),
        # A platform from before the route.
        (
            APIError(404, "Not Found"),
            "does not report its dagnam-contracts version and this SDK runs 0.4.2",
        ),
        # A value that is not a version proves nothing.
        (
            {"contracts": "unknown"},
            "does not report its dagnam-contracts version and this SDK runs 0.4.2",
        ),
    ],
)
def test_a_platform_behind_this_sdks_contract_stops_the_run_before_anything_is_uploaded(
    run_cli: CliRunner,
    prepared_dir: Path,
    platform: FakePlatform,
    build: Build,
    capsys: StrCapture,
    answer: JsonObject | APIError,
    says: str,
) -> None:
    """An SDK ahead of its platform used to upload the rows and then stop every workload at the PII check."""
    build.answer = answer
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(prepared_dir), "--yes"])
    assert exc.value.code == 1
    err = " ".join(capsys.readouterr().err.split())
    assert f"platform_too_old: the platform at https://x {says}." in err
    assert "The platform has not been upgraded for this SDK yet" in err
    assert "nothing was uploaded or spent by this run" in err
    assert build.reads == 1
    assert platform.call_log == []
    assert not (prepared_dir / "state.json").exists()


def test_local_only_is_held_to_the_same_platform_check(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform, build: Build, capsys: StrCapture
) -> None:
    """``--local-only`` skips the mirror into the account, not the platform.

    The rows are still uploaded and the candidates still trained there, so an
    older platform stops a local-only run exactly as it stops any other: before
    the upload, not at the PII check after it.
    """
    build.answer = {"revision": "abc", "version": "1"}
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(prepared_dir), "--yes", "--local-only"])
    assert exc.value.code == 1
    assert "platform_too_old: " in capsys.readouterr().err
    assert build.reads == 1
    assert platform.call_log == []


def test_a_platform_behind_is_a_json_error_under_json(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform, build: Build, capsys: StrCapture
) -> None:
    build.answer = {"contracts": "0.3.9"}
    with pytest.raises(SystemExit):
        run_cli(["audit", "run", str(prepared_dir), "--yes", "--json"])
    out = capsys.readouterr().out
    assert json.loads(out[out.index("\n{") + 1 :])["error"].startswith("platform_too_old: ")


@pytest.mark.parametrize(
    "answer",
    [
        {"contracts": "0.4.2"},
        {"contracts": "0.4.0"},  # a patch behind is the same contract, and says nothing
        {"contracts": "0.4.1"},
        {"contracts": "0.5.0rc1"},
        {"contracts": "1.0.0"},
        # The request itself failed: the run's own calls are what report an outage.
        APIError(0, "Request failed: connection refused"),
        APIError(503, "Service Unavailable"),
    ],
)
def test_a_platform_at_or_ahead_of_this_sdk_or_one_that_cannot_be_asked_runs_as_before(
    run_cli: CliRunner,
    prepared_dir: Path,
    platform: FakePlatform,
    build: Build,
    capsys: StrCapture,
    answer: JsonObject | APIError,
) -> None:
    build.answer = answer
    assert run_cli(["audit", "run", str(prepared_dir), "--yes", "--floor", "0.5"]) == 0
    captured = capsys.readouterr()
    assert "platform" not in captured.err  # silent: the check says nothing when it passes
    assert "platform_too_old" not in captured.out
    assert build.reads == 1  # read once, and before the first platform call of the run
    assert platform.call_log[0] == "create_project"
    assert platform.submits == 2


def test_the_platform_is_not_asked_before_the_upload_is_confirmed(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform, build: Build, tty: None
) -> None:
    with mock.patch("builtins.input", return_value="no"), pytest.raises(SystemExit):
        run_cli(["audit", "run", str(prepared_dir)])
    assert build.reads == 0


def test_a_newer_platform_patch_is_said_before_the_run_and_the_run_goes_on(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform, build: Build, capsys: StrCapture
) -> None:
    """Contract patches change what is found inside the same classes: say so, never stop."""
    build.answer = {"contracts": "0.4.3"}
    assert run_cli(["audit", "run", str(prepared_dir), "--yes", "--floor", "0.5"]) == 0
    err = " ".join(capsys.readouterr().err.split())
    assert "the platform runs dagnam-contracts 0.4.3 and this install 0.4.2" in err
    assert "Run `pip install -U dagnam-contracts` and scan again" in err
    assert platform.submits == 2


def test_an_answer_that_is_not_a_build_document_stops_the_run_like_a_missing_key(
    run_cli: CliRunner, prepared_dir: Path, platform: FakePlatform, build: Build, capsys: StrCapture
) -> None:
    """A 200 with HTML or a list proves nothing either; the route is as unreadable as a missing one."""
    build.answer = ResponseError(0, "Expected JSON object, got list")
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(prepared_dir), "--yes"])
    assert exc.value.code == 1
    err = " ".join(capsys.readouterr().err.split())
    assert "does not report its dagnam-contracts version" in err
    assert platform.call_log == []
