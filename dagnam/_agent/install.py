"""Detect installed agent harnesses and wire the Dagnam skill into each.

Pure logic: returns dataclasses describing planned/performed actions; it never prints
or prompts (``dagnam.cli.agent`` owns presentation). It must not import ``dagnam.cli``;
the installed version is read via ``importlib.metadata``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shutil
import sys

Harness = str  # "claude" | "codex"

# The guard hook must run under the SAME interpreter that has ``dagnam``
# installed. A bare ``python`` on PATH may resolve to a different interpreter
# (pipx / uv-tool / venv installs), where ``-m dagnam._agent.guardhook`` raises
# ModuleNotFoundError and breaks every matched tool call. Pin ``sys.executable``.
_GUARD_HOOK_MODULE_ARGS = ["-m", "dagnam._agent.guardhook"]


def _guard_hook_entry() -> dict[str, object]:
    """The Codex ``hooks.json`` guard entry, pinned to the current interpreter."""
    return {"command": [sys.executable, *_GUARD_HOOK_MODULE_ARGS]}


def _is_guard_hook_entry(entry: object) -> bool:
    """True if ``entry`` is our guard hook.

    Matches regardless of which interpreter path it was written with, so
    re-install stays idempotent and uninstall can find it even if the
    interpreter moved.
    """
    if not isinstance(entry, dict):
        return False
    command = entry.get("command")
    return isinstance(command, list) and command[-2:] == _GUARD_HOOK_MODULE_ARGS


@dataclass(frozen=True)
class InstallPlan:
    """What ``install_harness`` will write for one harness (no side effects)."""

    harness: Harness
    skill_dest: Path
    extra_dests: tuple[Path, ...] = ()


@dataclass
class InstallResult:
    """What ``install_harness`` actually wrote for one harness."""

    harness: Harness
    skill_dest: Path
    method: str  # "symlink" | "copy"
    wrote: list[Path] = field(default_factory=list)


def _home() -> Path:
    return Path.home()


def _assets_root() -> Path:
    """Filesystem path to the bundled ``dagnam/_agent`` assets (works post-pip-install)."""
    return Path(str(resources.files("dagnam._agent")))


def package_version() -> str:
    """Installed dagnam version, falling back to the in-tree value for source checkouts."""
    try:
        return version("dagnam")
    except PackageNotFoundError:
        from dagnam import __version__

        return __version__


def detect_harnesses() -> list[Harness]:
    """Return installed harnesses in stable order. Claude -> ~/.claude; Codex -> ~/.codex or ~/.agents."""
    home = _home()
    found: list[Harness] = []
    if (home / ".claude").exists():
        found.append("claude")
    if (home / ".codex").exists() or (home / ".agents").exists():
        found.append("codex")
    return found


def _skill_dest(harness: Harness) -> Path:
    home = _home()
    if harness == "claude":
        return home / ".claude" / "skills" / "dagnam"
    return home / ".agents" / "skills" / "dagnam"  # codex


def _claude_plugin_dir() -> Path:
    return _home() / ".claude" / "plugins" / "dagnam"


def plan_for(harness: Harness) -> InstallPlan:
    """Describe the install destinations for a harness without writing anything."""
    extra: tuple[Path, ...]
    if harness == "claude":
        extra = (_claude_plugin_dir(),)
    else:
        extra = (_home() / ".codex" / "hooks.json",)
    return InstallPlan(harness=harness, skill_dest=_skill_dest(harness), extra_dests=extra)


def _remove_dest(dest: Path) -> bool:
    """Remove ``dest`` whether it is a symlink or a real directory. Returns True if removed."""
    if dest.is_symlink():
        dest.unlink()
        return True
    if dest.exists():
        shutil.rmtree(dest)
        return True
    return False


def _try_symlink(dest: Path, src: Path) -> bool:
    """Attempt a directory symlink; return False on OSError (e.g. unprivileged Windows)."""
    try:
        dest.symlink_to(src, target_is_directory=True)
    except OSError:
        return False
    return True


def _copy_tree(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def _place_skill(method: str, dest: Path) -> str:
    """Copy or symlink the bundled ``skill/`` into ``dest``. Returns the method used."""
    src = _assets_root() / "skill"
    _remove_dest(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if method == "symlink" and _try_symlink(dest, src):
        return "symlink"
    _copy_tree(src, dest)
    return "copy"


def _stamp_version(path: Path, key: str = "version") -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data[key] = package_version()
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _merge_codex_hook(wrote: list[Path]) -> None:
    """Idempotently merge the guard hook into ``~/.codex/hooks.json``, preserving other entries."""
    codex = _home() / ".codex"
    codex.mkdir(parents=True, exist_ok=True)
    hooks_path = codex / "hooks.json"
    data: dict[str, object] = {}
    if hooks_path.exists():
        try:
            loaded = json.loads(hooks_path.read_text(encoding="utf-8"))
            data = loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            data = {}
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        hooks = {}
        data["hooks"] = hooks
    pre = hooks.setdefault("PreToolUse", [])
    if not isinstance(pre, list):
        pre = []
        hooks["PreToolUse"] = pre
    if not any(_is_guard_hook_entry(entry) for entry in pre):
        pre.append(_guard_hook_entry())
    hooks_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    wrote.append(hooks_path)


def _install_codex_extras(wrote: list[Path]) -> None:
    """Place + version-stamp the Codex skill metadata, then merge the guard hook."""
    dest_agents = _skill_dest("codex") / "agents"
    dest_agents.mkdir(parents=True, exist_ok=True)
    src_yaml = _assets_root() / "codex" / "agents" / "openai.yaml"
    yaml_text = src_yaml.read_text(encoding="utf-8").replace(
        "version: 0.0.0", f"version: {package_version()}"
    )
    yaml_path = dest_agents / "openai.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    wrote.append(yaml_path)
    _merge_codex_hook(wrote)


def install_harness(harness: Harness, *, method: str = "copy") -> InstallResult:
    """Install the skill (and adapter files) for one harness. Idempotent."""
    dest = _skill_dest(harness)
    used = _place_skill(method, dest)
    wrote = [dest]
    if harness == "claude":
        plugin_dir = _claude_plugin_dir()
        _copy_tree(_assets_root() / "claude", plugin_dir)
        _stamp_version(plugin_dir / "plugin.json")
        _stamp_claude_guard_interpreter(plugin_dir / "hooks" / "guard.json")
        wrote.append(plugin_dir)
    else:  # codex
        _install_codex_extras(wrote)
    return InstallResult(harness=harness, skill_dest=dest, method=used, wrote=wrote)


def _stamp_claude_guard_interpreter(guard_path: Path) -> None:
    """Rewrite the copied Claude guard hook to invoke the current interpreter.

    The bundled asset ships a bare ``python`` command; pin it to
    ``sys.executable`` (double-quoted so a path with spaces survives the shell)
    for the same reason as the Codex entry — a different PATH ``python`` cannot
    import ``dagnam``.
    """
    data = json.loads(guard_path.read_text(encoding="utf-8"))
    command = f'"{sys.executable}" ' + " ".join(_GUARD_HOOK_MODULE_ARGS)
    for group in data.get("hooks", {}).get("PreToolUse", []):
        for hook in group.get("hooks", []):
            if isinstance(hook, dict) and "-m dagnam._agent.guardhook" in str(hook.get("command")):
                hook["command"] = command
    guard_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _unmerge_codex_hook() -> list[Path]:
    """Remove our guard entry from ``~/.codex/hooks.json``, leaving others intact.

    Mirrors ``_merge_codex_hook`` so a codex uninstall — or a plain
    ``pip uninstall dagnam`` follow-up — doesn't leave the hook firing against a
    now-missing module.
    """
    hooks_path = _home() / ".codex" / "hooks.json"
    if not hooks_path.exists():
        return []
    try:
        data = json.loads(hooks_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        return []
    pre = hooks.get("PreToolUse")
    if not isinstance(pre, list):
        return []
    kept = [entry for entry in pre if not _is_guard_hook_entry(entry)]
    if len(kept) == len(pre):
        return []
    hooks["PreToolUse"] = kept
    hooks_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return [hooks_path]


def uninstall_harness(harness: Harness) -> list[Path]:
    """Remove what ``install_harness`` wrote for a harness. Returns removed paths. Idempotent."""
    removed: list[Path] = []
    dest = _skill_dest(harness)
    if _remove_dest(dest):
        removed.append(dest)
    if harness == "claude":
        plugin_dir = _claude_plugin_dir()
        if plugin_dir.exists():
            shutil.rmtree(plugin_dir)
            removed.append(plugin_dir)
    else:  # codex
        removed.extend(_unmerge_codex_hook())
    return removed
