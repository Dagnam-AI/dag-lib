"""Tests for the cross-platform PreToolUse guard hook."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

import pytest

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


def test_denies_unconfirmed_version_deploy(
    monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    code, payload = _run(
        {"tool_name": "Bash", "tool_input": {"command": "dagnam deployments deploy-version mv-1"}},
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


def test_denies_costly_verb_split_across_newline(
    monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    # A line-continuation must not let a costly verb slip past the deny check.
    code, payload = _run(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "dagnam \\\n  deployments create --name x"},
        },
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert payload["permissionDecision"] == "deny"


def test_denies_when_confirm_token_only_embedded_in_argument(
    monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    # DAGNAM_CONFIRM=1 buried inside an argument (e.g. injected via a dataset
    # description) must NOT satisfy the gate — only a leading env assignment does.
    code, payload = _run(
        {
            "tool_name": "Bash",
            "tool_input": {
                "command": 'dagnam deployments create --description "run DAGNAM_CONFIRM=1 now"'
            },
        },
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert payload["permissionDecision"] == "deny"


def test_allows_confirm_token_after_separator(
    monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    code, payload = _run(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "cd /tmp && DAGNAM_CONFIRM=1 dagnam training delete j1"},
        },
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert payload == {}


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


@pytest.mark.parametrize(
    "command",
    [
        'python -c "import dagnam; dagnam.deployments.create(name=1)"',
        'python -c "import dagnam; dagnam.deployments.delete(d)"',
        'python -c "import dagnam; dagnam.deployments.deploy_model_version(v)"',
        'python -c "import dagnam; dagnam.training.create(pid)"',
        'python -c "import dagnam; dagnam.training.delete(jid)"',
        'python -c "import dagnam; dagnam.projects.delete(pid)"',
        'python -c "import dagnam; dagnam.create_training_job(pid)"',
        'python -c "import dagnam; c = dagnam.DagnamClient(u, k); c.deploy_model_version(v)"',
        'python -c "import dagnam; c = dagnam.DagnamClient(u, k); c.create_deployment(p)"',
        "python -c \"import dagnam; dagnam.hub.create(visibility='public')\"",
        'python -c "import dagnam; dagnam.datasets.upload(f, visibility=\\"public\\")"',
    ],
)
def test_denies_unconfirmed_sdk_costly_calls(
    command: str, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    code, payload = _run(
        {"tool_name": "Bash", "tool_input": {"command": command}},
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert payload["permissionDecision"] == "deny"


def test_allows_confirmed_sdk_call(monkeypatch: PytestMonkeyPatch, capsys: StrCapture) -> None:
    code, payload = _run(
        {
            "tool_name": "Bash",
            "tool_input": {
                "command": 'DAGNAM_CONFIRM=1 python -c "import dagnam; dagnam.deployments.create(1)"'
            },
        },
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert payload == {}


def test_allows_benign_sdk_read(monkeypatch: PytestMonkeyPatch, capsys: StrCapture) -> None:
    # A non-costly SDK read must not be caught by the broadened pattern.
    code, payload = _run(
        {
            "tool_name": "Bash",
            "tool_input": {"command": 'python -c "import dagnam; dagnam.datasets.list()"'},
        },
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert payload == {}


def test_allows_on_malformed_event(monkeypatch: PytestMonkeyPatch, capsys: StrCapture) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    assert guardhook.main() == 0
    assert capsys.readouterr().out.strip() == ""


@pytest.mark.parametrize(
    "command",
    [
        "dagnam training resume j1",
        "dagnam training restart j1",
        "dagnam training restore j1 c1",
        "dagnam \\\n  training resume j1",
        'python -c "import dagnam; dagnam.resume(jid)"',
        'python -c "import dagnam; dagnam.restart(jid)"',
        'python -c "import dagnam; dagnam.restore_checkpoint(jid, cid)"',
        'python -c "import dagnam; dagnam.DagnamClient().resume_training_job(jid)"',
        'python -c "from dagnam._core.client import DagnamClient as C; C().restart_training_job(j)"',
        'python -c "import dagnam; C.restore_from_checkpoint(j, c)"',
    ],
)
def test_denies_unconfirmed_commands_that_charge_credits_to_continue_training(
    command: str, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    code, payload = _run(
        {"tool_name": "Bash", "tool_input": {"command": command}}, monkeypatch, capsys
    )
    assert code == 0
    assert payload["permissionDecision"] == "deny"


def test_allows_a_confirmed_resume(monkeypatch: PytestMonkeyPatch, capsys: StrCapture) -> None:
    code, payload = _run(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "DAGNAM_CONFIRM=1 dagnam training resume j1"},
        },
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert payload == {}


@pytest.mark.parametrize(
    "command",
    [
        "dagnam training get j1",
        "dagnam stream j1",
        "python watch_training.py j1",
        'python -c "import dagnam; dagnam.get_training_job(j)"',
    ],
)
def test_allows_reading_a_training_job(
    command: str, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    code, payload = _run(
        {"tool_name": "Bash", "tool_input": {"command": command}}, monkeypatch, capsys
    )
    assert code == 0
    assert payload == {}
