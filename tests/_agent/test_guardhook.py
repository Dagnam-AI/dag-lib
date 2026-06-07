"""Tests for the cross-platform PreToolUse guard hook."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

from dagnam._agent import guardhook

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, StrCapture


def _run(
    event: object, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> tuple[int, dict[str, object]]:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    code = guardhook.main()
    out = capsys.readouterr().out
    return code, (json.loads(out) if out.strip() else {})


def test_denies_unconfirmed_deploy(monkeypatch: PytestMonkeyPatch, capsys: StrCapture) -> None:
    code, payload = _run(
        {"tool_name": "Bash", "tool_input": {"command": "dagnam deployments create --name x"}},
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert payload["permissionDecision"] == "deny"


def test_denies_unconfirmed_training_delete(
    monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    code, payload = _run(
        {"tool_name": "Bash", "tool_input": {"command": "dagnam training delete j1 j2"}},
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert payload["permissionDecision"] == "deny"


def test_allows_confirmed_deploy(monkeypatch: PytestMonkeyPatch, capsys: StrCapture) -> None:
    code, payload = _run(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "DAGNAM_CONFIRM=1 dagnam deployments create --name x"},
        },
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert payload == {}  # silent allow


def test_allows_non_costly_command(monkeypatch: PytestMonkeyPatch, capsys: StrCapture) -> None:
    code, payload = _run(
        {"tool_name": "Bash", "tool_input": {"command": "dagnam dataset list --json"}},
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert payload == {}


def test_allows_on_malformed_event(monkeypatch: PytestMonkeyPatch, capsys: StrCapture) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    assert guardhook.main() == 0
    assert capsys.readouterr().out.strip() == ""
