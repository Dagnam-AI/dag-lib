"""`dagnam agent` -- install the Dagnam Agent Skill into Claude Code and/or Codex."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from dagnam._agent import install as _install
from dagnam.cli.common import error

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction


def _selected(args: argparse.Namespace) -> list[str]:
    """Resolve which harnesses to act on: explicit flags, ``--all``, else auto-detect."""
    if args.all:
        return _install.detect_harnesses()
    chosen = [h for h, on in (("claude", args.claude), ("codex", args.codex)) if on]
    return chosen or _install.detect_harnesses()


def cmd_agent_install(args: argparse.Namespace) -> None:
    """Install the skill into the selected/detected harnesses (auto-detect & prompt by default)."""
    targets = _selected(args)
    if not targets:
        error("No supported agent harness detected (looked for ~/.claude and ~/.codex/~/.agents).")
    print(f"Will install the Dagnam skill for: {', '.join(targets)}")
    for h in targets:
        print(f"  {h}: {_install.plan_for(h).skill_dest}")
    if not args.yes and input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Aborted. Nothing was installed.")
        return
    method = "symlink" if args.symlink else "copy"
    for h in targets:
        _install.install_harness(h, method=method)
    print(f"Installed the Dagnam skill for: {', '.join(targets)}")


def cmd_agent_uninstall(args: argparse.Namespace) -> None:
    """Remove the skill from the selected/detected harnesses."""
    targets = _selected(args)
    if not targets:
        error("No supported agent harness detected.")
    if not args.yes and input(
        f"Remove the Dagnam skill from {', '.join(targets)}? [y/N] "
    ).strip().lower() not in ("y", "yes"):
        print("Aborted. Nothing was removed.")
        return
    for h in targets:
        _install.uninstall_harness(h)
    print(f"Removed the Dagnam skill from: {', '.join(targets)}")


def register_agent(subparsers: SubParsersAction) -> None:
    """Register the ``agent`` command group."""
    agent = subparsers.add_parser(
        "agent",
        help="Install the Dagnam Agent Skill into Claude Code / Codex.",
        description="Install or remove the cross-platform Dagnam Agent Skill.",
    )
    agent_sub = agent.add_subparsers(dest="agent_command", required=True)

    install_cmd = agent_sub.add_parser("install", help="Install the skill (auto-detect & prompt).")
    install_cmd.add_argument("--claude", action="store_true", help="Target Claude Code.")
    install_cmd.add_argument("--codex", action="store_true", help="Target Codex.")
    install_cmd.add_argument(
        "--all", action="store_true", help="Install to all detected harnesses."
    )
    install_cmd.add_argument(
        "--yes", action="store_true", help="Skip the confirmation prompt (for CI)."
    )
    install_cmd.add_argument(
        "--symlink", action="store_true", help="Symlink instead of copying the skill."
    )
    install_cmd.set_defaults(func=cmd_agent_install)

    uninstall_cmd = agent_sub.add_parser("uninstall", help="Remove the installed skill.")
    uninstall_cmd.add_argument("--claude", action="store_true", help="Target Claude Code.")
    uninstall_cmd.add_argument("--codex", action="store_true", help="Target Codex.")
    uninstall_cmd.add_argument(
        "--all", action="store_true", help="Remove from all detected harnesses."
    )
    uninstall_cmd.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    uninstall_cmd.set_defaults(func=cmd_agent_uninstall)
