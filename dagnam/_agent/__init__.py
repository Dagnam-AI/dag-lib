"""Agent-integration assets and installer.

This private package bundles the cross-platform Agent Skill (``skill/``) plus the
Claude Code and Codex adapter files (``claude/``, ``codex/``), and the importable
logic that powers them (``install.py``, ``runner.py``, ``guardhook.py``). It ships
as package data so ``dagnam agent install`` can locate it via ``importlib.resources``
after a pip install.

It must never import ``dagnam.cli`` (the CLI depends on this package, not the
reverse); the installed version is read via ``importlib.metadata`` instead.
"""
