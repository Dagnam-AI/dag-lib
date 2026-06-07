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

Harness = str  # "claude" | "codex"

_GUARD_HOOK_ENTRY: dict[str, object] = {"command": ["python", "-m", "dagnam._agent.guardhook"]}


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
    if _GUARD_HOOK_ENTRY not in pre:
        pre.append(_GUARD_HOOK_ENTRY)
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
        wrote.append(plugin_dir)
    else:  # codex
        _install_codex_extras(wrote)
    return InstallResult(harness=harness, skill_dest=dest, method=used, wrote=wrote)


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
    return removed
