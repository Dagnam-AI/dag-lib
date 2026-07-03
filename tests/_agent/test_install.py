"""Tests for dagnam._agent.install."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING

import pytest

from dagnam._agent import install

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


# --- detection --------------------------------------------------------------


def test_detect_finds_nothing_on_clean_home(fake_home: Path) -> None:
    assert install.detect_harnesses() == []


def test_detect_finds_claude(fake_home: Path) -> None:
    (fake_home / ".claude").mkdir()
    assert install.detect_harnesses() == ["claude"]


def test_detect_finds_codex_via_dot_codex(fake_home: Path) -> None:
    (fake_home / ".codex").mkdir()
    assert install.detect_harnesses() == ["codex"]


def test_detect_finds_both_via_dot_agents(fake_home: Path) -> None:
    (fake_home / ".claude").mkdir()
    (fake_home / ".agents").mkdir()  # codex skills root
    assert install.detect_harnesses() == ["claude", "codex"]


# --- version ----------------------------------------------------------------


def test_package_version_returns_installed() -> None:
    assert re.match(r"^\d+\.\d+", install.package_version())


def test_package_version_falls_back_for_source_checkout(monkeypatch: PytestMonkeyPatch) -> None:
    from importlib.metadata import PackageNotFoundError

    def boom(_name: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr(install, "version", boom)
    import dagnam

    assert install.package_version() == dagnam.__version__


# --- plan_for ---------------------------------------------------------------


def test_plan_for_claude_targets_skills_dir(fake_home: Path) -> None:
    plan = install.plan_for("claude")
    assert plan.skill_dest == fake_home / ".claude" / "skills" / "dagnam"
    assert plan.extra_dests == (fake_home / ".claude" / "plugins" / "dagnam",)


def test_plan_for_codex_targets_agents_dir(fake_home: Path) -> None:
    plan = install.plan_for("codex")
    assert plan.skill_dest == fake_home / ".agents" / "skills" / "dagnam"
    assert plan.extra_dests == (fake_home / ".codex" / "hooks.json",)


# --- claude install ---------------------------------------------------------


def test_install_claude_copies_skill_and_stamps_version(fake_home: Path) -> None:
    result = install.install_harness("claude", method="copy")
    dest = fake_home / ".claude" / "skills" / "dagnam"
    assert (dest / "SKILL.md").exists()
    assert (dest / "scripts" / "plan.py").exists()
    stamped = _read_json(install._claude_plugin_dir() / "plugin.json")
    assert stamped["version"] == install.package_version()
    assert result.method == "copy"


def test_install_is_idempotent(fake_home: Path) -> None:
    install.install_harness("claude", method="copy")
    install.install_harness("claude", method="copy")  # must not raise; overwrites in place
    assert (fake_home / ".claude" / "skills" / "dagnam" / "SKILL.md").exists()


def test_install_symlink_method(fake_home: Path, monkeypatch: PytestMonkeyPatch) -> None:
    def fake_symlink(self: Path, target: Path, *, target_is_directory: bool = False) -> None:
        return None

    monkeypatch.setattr(Path, "symlink_to", fake_symlink)
    result = install.install_harness("claude", method="symlink")
    assert result.method == "symlink"


def test_install_symlink_falls_back_to_copy(
    fake_home: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    def boom(self: Path, target: Path, *, target_is_directory: bool = False) -> None:
        raise OSError("symlinks need privilege here")

    monkeypatch.setattr(Path, "symlink_to", boom)
    result = install.install_harness("claude", method="symlink")
    assert result.method == "copy"
    assert (fake_home / ".claude" / "skills" / "dagnam" / "SKILL.md").exists()


def test_remove_dest_unlinks_symlink(tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    target = tmp_path / "link"
    unlinked: dict[str, Path] = {}
    monkeypatch.setattr(Path, "is_symlink", lambda self: True)
    monkeypatch.setattr(Path, "unlink", lambda self, **_: unlinked.setdefault("p", self))
    assert install._remove_dest(target) is True
    assert unlinked["p"] == target


# --- codex install ----------------------------------------------------------


def test_install_codex_places_yaml_and_merges_hook(fake_home: Path) -> None:
    result = install.install_harness("codex", method="copy")
    skill = fake_home / ".agents" / "skills" / "dagnam"
    assert (skill / "SKILL.md").exists()
    yaml_text = (skill / "agents" / "openai.yaml").read_text(encoding="utf-8")
    assert f"version: {install.package_version()}" in yaml_text
    merged = _read_json(fake_home / ".codex" / "hooks.json")
    assert "PreToolUse" in merged["hooks"]  # type: ignore[index]
    assert result.harness == "codex"


def _seed_codex_hooks(fake_home: Path, text: str) -> Path:
    codex = fake_home / ".codex"
    codex.mkdir()
    path = codex / "hooks.json"
    path.write_text(text, encoding="utf-8")
    return path


def test_codex_hook_merge_preserves_existing(fake_home: Path) -> None:
    _seed_codex_hooks(fake_home, json.dumps({"hooks": {"SessionStart": [{"command": ["echo"]}]}}))
    install.install_harness("codex", method="copy")
    merged = _read_json(fake_home / ".codex" / "hooks.json")
    hooks = merged["hooks"]
    assert isinstance(hooks, dict)
    assert "SessionStart" in hooks  # preserved
    assert "PreToolUse" in hooks  # added


def test_codex_hook_merge_tolerates_corrupt_json(fake_home: Path) -> None:
    _seed_codex_hooks(fake_home, "{ this is not json")
    install.install_harness("codex", method="copy")
    merged = _read_json(fake_home / ".codex" / "hooks.json")
    assert "PreToolUse" in merged["hooks"]  # type: ignore[index]


def test_codex_hook_merge_tolerates_non_dict_root(fake_home: Path) -> None:
    _seed_codex_hooks(fake_home, "[]")
    install.install_harness("codex", method="copy")
    merged = _read_json(fake_home / ".codex" / "hooks.json")
    assert "PreToolUse" in merged["hooks"]  # type: ignore[index]


def test_codex_hook_merge_tolerates_non_dict_hooks(fake_home: Path) -> None:
    _seed_codex_hooks(fake_home, json.dumps({"hooks": "oops"}))
    install.install_harness("codex", method="copy")
    merged = _read_json(fake_home / ".codex" / "hooks.json")
    hooks = merged["hooks"]
    assert isinstance(hooks, dict)
    assert "PreToolUse" in hooks


def test_codex_hook_merge_tolerates_non_list_pretooluse(fake_home: Path) -> None:
    _seed_codex_hooks(fake_home, json.dumps({"hooks": {"PreToolUse": "x"}}))
    install.install_harness("codex", method="copy")
    merged = _read_json(fake_home / ".codex" / "hooks.json")
    pre = merged["hooks"]["PreToolUse"]  # type: ignore[index]
    assert isinstance(pre, list)
    assert any(install._is_guard_hook_entry(entry) for entry in pre)


def test_codex_hook_merge_is_idempotent(fake_home: Path) -> None:
    install.install_harness("codex", method="copy")
    install.install_harness("codex", method="copy")
    merged = _read_json(fake_home / ".codex" / "hooks.json")
    assert len(merged["hooks"]["PreToolUse"]) == 1  # type: ignore[index]


# --- uninstall --------------------------------------------------------------


def test_uninstall_claude_removes_skill_and_plugin(fake_home: Path) -> None:
    install.install_harness("claude", method="copy")
    removed = install.uninstall_harness("claude")
    assert not (fake_home / ".claude" / "skills" / "dagnam").exists()
    assert not install._claude_plugin_dir().exists()
    assert len(removed) == 2


def test_uninstall_codex_removes_skill(fake_home: Path) -> None:
    install.install_harness("codex", method="copy")
    install.uninstall_harness("codex")
    assert not (fake_home / ".agents" / "skills" / "dagnam").exists()


def test_codex_guard_hook_uses_current_interpreter(fake_home: Path) -> None:
    install.install_harness("codex", method="copy")
    pre = _read_json(fake_home / ".codex" / "hooks.json")["hooks"]["PreToolUse"]  # type: ignore[index]
    guard = next(entry for entry in pre if install._is_guard_hook_entry(entry))
    assert guard["command"][0] == sys.executable  # not a bare "python"


def test_claude_guard_json_uses_current_interpreter(fake_home: Path) -> None:
    install.install_harness("claude", method="copy")
    guard = _read_json(install._claude_plugin_dir() / "hooks" / "guard.json")
    command = guard["hooks"]["PreToolUse"][0]["hooks"][0]["command"]  # type: ignore[index]
    assert sys.executable in command
    assert "-m dagnam._agent.guardhook" in command


def test_uninstall_codex_removes_guard_hook_but_keeps_others(fake_home: Path) -> None:
    _seed_codex_hooks(fake_home, json.dumps({"hooks": {"SessionStart": [{"command": ["echo"]}]}}))
    install.install_harness("codex", method="copy")
    removed = install.uninstall_harness("codex")

    hooks = _read_json(fake_home / ".codex" / "hooks.json")["hooks"]
    assert not any(install._is_guard_hook_entry(e) for e in hooks["PreToolUse"])  # type: ignore[index]
    assert hooks["SessionStart"] == [{"command": ["echo"]}]  # type: ignore[index]
    assert (fake_home / ".codex" / "hooks.json") in removed


def test_uninstall_clean_home_is_noop(fake_home: Path) -> None:
    assert install.uninstall_harness("claude") == []


def test_stamp_claude_guard_interpreter_only_rewrites_matching_hook(tmp_path: Path) -> None:
    guard = tmp_path / "guard.json"
    guard.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "hooks": [
                                {"type": "command", "command": "echo unrelated"},
                                {"type": "command", "command": "python -m dagnam._agent.guardhook"},
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    install._stamp_claude_guard_interpreter(guard)
    hooks = _read_json(guard)["hooks"]["PreToolUse"][0]["hooks"]  # type: ignore[index]
    assert hooks[0]["command"] == "echo unrelated"  # untouched
    assert sys.executable in hooks[1]["command"]  # rewritten to this interpreter


def test_is_guard_hook_entry_rejects_non_dict() -> None:
    assert install._is_guard_hook_entry("not-a-dict") is False
    assert install._is_guard_hook_entry({"command": ["echo", "hi"]}) is False


def test_unmerge_codex_hook_no_file_is_noop(fake_home: Path) -> None:
    assert install._unmerge_codex_hook() == []


def test_unmerge_codex_hook_tolerates_corrupt_json(fake_home: Path) -> None:
    _seed_codex_hooks(fake_home, "{ not json")
    assert install._unmerge_codex_hook() == []


def test_unmerge_codex_hook_tolerates_non_dict_root(fake_home: Path) -> None:
    _seed_codex_hooks(fake_home, "[]")
    assert install._unmerge_codex_hook() == []


def test_unmerge_codex_hook_tolerates_non_dict_hooks(fake_home: Path) -> None:
    _seed_codex_hooks(fake_home, json.dumps({"hooks": "oops"}))
    assert install._unmerge_codex_hook() == []


def test_unmerge_codex_hook_tolerates_non_list_pretooluse(fake_home: Path) -> None:
    _seed_codex_hooks(fake_home, json.dumps({"hooks": {"PreToolUse": "x"}}))
    assert install._unmerge_codex_hook() == []


def test_unmerge_codex_hook_noop_when_guard_absent(fake_home: Path) -> None:
    _seed_codex_hooks(fake_home, json.dumps({"hooks": {"PreToolUse": [{"command": ["echo"]}]}}))
    assert install._unmerge_codex_hook() == []
