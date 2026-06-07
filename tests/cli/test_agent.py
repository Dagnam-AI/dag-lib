"""Tests for the `dagnam agent` CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from dagnam._agent import install as install_mod
from dagnam.cli import agent as agent_cli

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, StrCapture


def _ns(**kw: object) -> argparse.Namespace:
    return argparse.Namespace(**kw)


def _record_installs(monkeypatch: PytestMonkeyPatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def fake(h: str, *, method: str = "copy") -> install_mod.InstallResult:
        calls.append((h, method))
        return install_mod.InstallResult(harness=h, skill_dest=Path(h), method=method)

    monkeypatch.setattr(install_mod, "install_harness", fake)
    return calls


# --- install ----------------------------------------------------------------


def test_install_all_flag_skips_prompt_and_installs_detected(
    monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    monkeypatch.setattr(install_mod, "detect_harnesses", lambda: ["claude", "codex"])
    calls = _record_installs(monkeypatch)
    agent_cli.cmd_agent_install(_ns(all=True, yes=True, claude=False, codex=False, symlink=False))
    assert [h for h, _ in calls] == ["claude", "codex"]
    assert "Installed" in capsys.readouterr().out


def test_install_explicit_claude_flag_uses_chosen(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr(install_mod, "detect_harnesses", lambda: ["claude", "codex"])
    calls = _record_installs(monkeypatch)
    agent_cli.cmd_agent_install(_ns(all=False, yes=True, claude=True, codex=False, symlink=False))
    assert [h for h, _ in calls] == ["claude"]  # honored the flag, did not auto-detect


def test_install_prompt_accepted_with_symlink(
    monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    monkeypatch.setattr(install_mod, "detect_harnesses", lambda: ["claude"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    calls = _record_installs(monkeypatch)
    agent_cli.cmd_agent_install(_ns(all=False, yes=False, claude=False, codex=False, symlink=True))
    assert calls == [("claude", "symlink")]
    assert "Installed" in capsys.readouterr().out


def test_install_declined_at_prompt_installs_nothing(
    monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    monkeypatch.setattr(install_mod, "detect_harnesses", lambda: ["claude"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    calls = _record_installs(monkeypatch)
    agent_cli.cmd_agent_install(_ns(all=False, yes=False, claude=False, codex=False, symlink=False))
    assert calls == []
    assert "Aborted" in capsys.readouterr().out


def test_install_no_harness_detected_errors(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr(install_mod, "detect_harnesses", lambda: [])
    with pytest.raises(SystemExit):
        agent_cli.cmd_agent_install(
            _ns(all=False, yes=True, claude=False, codex=False, symlink=False)
        )


# --- uninstall --------------------------------------------------------------


def _record_uninstalls(monkeypatch: PytestMonkeyPatch) -> list[str]:
    removed: list[str] = []
    monkeypatch.setattr(install_mod, "uninstall_harness", lambda h: removed.append(h) or [])
    return removed


def test_uninstall_all_flag(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr(install_mod, "detect_harnesses", lambda: ["codex"])
    removed = _record_uninstalls(monkeypatch)
    agent_cli.cmd_agent_uninstall(_ns(all=True, yes=True, claude=False, codex=False))
    assert removed == ["codex"]


def test_uninstall_prompt_accepted(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr(install_mod, "detect_harnesses", lambda: ["codex"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    removed = _record_uninstalls(monkeypatch)
    agent_cli.cmd_agent_uninstall(_ns(all=False, yes=False, claude=False, codex=False))
    assert removed == ["codex"]


def test_uninstall_declined(monkeypatch: PytestMonkeyPatch, capsys: StrCapture) -> None:
    monkeypatch.setattr(install_mod, "detect_harnesses", lambda: ["codex"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    removed = _record_uninstalls(monkeypatch)
    agent_cli.cmd_agent_uninstall(_ns(all=False, yes=False, claude=False, codex=False))
    assert removed == []
    assert "Aborted" in capsys.readouterr().out


def test_uninstall_no_harness_detected_errors(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr(install_mod, "detect_harnesses", lambda: [])
    with pytest.raises(SystemExit):
        agent_cli.cmd_agent_uninstall(_ns(all=False, yes=True, claude=False, codex=False))


# --- registration -----------------------------------------------------------


def test_register_agent_wires_subcommand() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    agent_cli.register_agent(sub)
    args = parser.parse_args(["agent", "install", "--all", "--yes"])
    assert args.func is agent_cli.cmd_agent_install
    args2 = parser.parse_args(["agent", "uninstall", "--all", "--yes"])
    assert args2.func is agent_cli.cmd_agent_uninstall
