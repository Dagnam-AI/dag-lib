"""Branded argparse layer: grouped help and nearest-match suggestions.

Pure stdlib (argparse/difflib/re). Imports only from ``dagnam.cli.common`` and
``dagnam.cli.errors`` (both stdlib-light) to respect the SDK layer contract,
and never imports ``dagnam._core`` at module scope so the CLI stays
import-light (tests/cli/test_lightweight_import.py).
"""

from __future__ import annotations

import argparse
import difflib
import re
from typing import TYPE_CHECKING, NoReturn, override

from dagnam.cli.common import DOCS_URL, format_ascii_art, resolve_version
from dagnam.cli.errors import styled_error_label

if TYPE_CHECKING:
    from collections.abc import Iterable

# Single source of truth for grouping top-level commands in the help. Every
# registered top-level command must appear in exactly one group; the drift test
# in tests/cli/test_parser.py asserts membership both ways.
COMMAND_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Auth", ("login", "register", "logout", "whoami")),
    ("Data", ("dataset", "cache")),
    ("Models", ("projects", "codegen", "hub", "checkpoint", "inference", "deployments")),
    ("Training", ("training", "stream")),
    ("Account", ("usage", "account", "profile", "keys", "config", "version", "agent")),
)

ALL_GROUPED_COMMANDS: tuple[str, ...] = tuple(
    command for _title, commands in COMMAND_GROUPS for command in commands
)

COMMAND_DESCRIPTIONS: dict[str, str] = {
    "login": "Authenticate and store an API key.",
    "register": "Create an account and store an API key.",
    "logout": "Remove stored credentials.",
    "whoami": "Show the current authenticated identity.",
    "dataset": "Browse and download datasets.",
    "cache": "Inspect and clear the local dataset cache.",
    "projects": "Manage projects.",
    "codegen": "Generate model code from a project.",
    "hub": "Browse the model hub.",
    "checkpoint": "List and download training checkpoints.",
    "inference": "Run inference against a deployment.",
    "deployments": "Manage model deployments.",
    "training": "Create, inspect, and manage training jobs.",
    "stream": "Stream live training events.",
    "usage": "Show plan, usage, and remaining limits.",
    "account": "Manage settings and notification preferences.",
    "profile": "View a user's public profile.",
    "keys": "Create, list, and revoke API keys.",
    "config": "Inspect and update saved configuration.",
    "version": "Show version and environment info.",
    "agent": "Install the Dagnam Agent Skill into Claude Code / Codex.",
}

EXAMPLES: tuple[str, ...] = (
    "  dagnam login                         Authenticate with an API key",
    "  dagnam register                      Create an account and store an API key",
    "  dagnam dataset list --search mnist   Search available datasets",
    "  dagnam projects create --title X     Create a new project",
    "  dagnam training create <pid> ...     Start a training job",
    "  dagnam training attach <jid> -- ...  Attach local metrics to a child process",
    "  dagnam config set training_metrics_path <path>",
    "  dagnam usage                         Show plan usage and limits",
    "  dagnam deployments logs <id>         Tail a deployment's logs",
)


def suggest(token: str, candidates: Iterable[str]) -> str | None:
    """Return the closest candidate to ``token``, or ``None`` if none is close."""
    matches = difflib.get_close_matches(token, list(candidates), n=1, cutoff=0.6)
    return matches[0] if matches else None


def format_root_help(*, banner: bool) -> str:
    """Render the top-level help.

    ``banner=True`` (bare ``dagnam`` invocation) shows the full ASCII art;
    ``banner=False`` (``-h``/``--help``) shows a compact one-line wordmark.
    """
    head = (
        f"{format_ascii_art()}\n\ndagnam {resolve_version()}"
        if banner
        else f"dagnam {resolve_version()} - Official CLI for Dagnam.AI"
    )
    lines = [head, "", "Usage: dagnam [-h] [-v] [--debug] <command> ...", "", "Commands:"]
    command_width = max(len(command) for command in ALL_GROUPED_COMMANDS)
    for title, commands in COMMAND_GROUPS:
        lines.append(f"  {title}:")
        lines.extend(
            f"    {command.ljust(command_width)}  {COMMAND_DESCRIPTIONS[command]}"
            for command in commands
        )
    lines.extend(
        [
            "",
            "Options:",
            "  -h, --help     Show this help and exit.",
            "  -v, --version  Show the dagnam version and exit.",
            "  --debug        Show full tracebacks on error.",
            "",
            "Examples:",
            *EXAMPLES,
            "",
            f"Docs: {DOCS_URL}",
        ]
    )
    return "\n".join(lines) + "\n"


_INVALID_CHOICE = re.compile(r"invalid choice: '(?P<bad>[^']*)' \(choose from (?P<choices>.+)\)")
_UNRECOGNIZED = re.compile(r"unrecognized arguments: (?P<args>.+)")


def _subparsers_action(parser: argparse.ArgumentParser) -> argparse.Action | None:
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict) and any(
            isinstance(choice, argparse.ArgumentParser) for choice in choices.values()
        ):
            return action
    return None


def _choice_help(action: argparse.Action) -> dict[str, str]:
    choices: dict[str, str] = {}
    for choice in getattr(action, "_choices_actions", ()):
        name = getattr(choice, "dest", None)
        help_text = getattr(choice, "help", None)
        if isinstance(name, str) and isinstance(help_text, str):
            choices[name] = help_text
    return choices


def _choice_names(action: argparse.Action) -> tuple[str, ...]:
    choices = getattr(action, "choices", {})
    return tuple(choices) if isinstance(choices, dict) else ()


def _format_group_help(parser: argparse.ArgumentParser, action: argparse.Action) -> str:
    names = _choice_names(action)
    command_width = max((len(name) for name in names), default=0)
    descriptions = _choice_help(action)
    lines = [
        f"Usage: {parser.prog} [-h] <command> ...",
        "",
    ]
    if parser.description:
        lines.extend([parser.description, ""])
    lines.append("Commands:")
    lines.extend(f"  {name.ljust(command_width)}  {descriptions.get(name, '')}" for name in names)
    lines.extend(
        [
            "",
            "Options:",
            "  -h, --help    Show this help and exit.",
            "",
            f"Docs: {DOCS_URL}",
        ]
    )
    return "\n".join(lines) + "\n"


class DagnamArgumentParser(argparse.ArgumentParser):
    """Argparse parser with grouped root help and nearest-match suggestions.

    Children created by ``add_subparsers().add_parser(...)`` inherit this class
    automatically, so the suggestion behavior applies at every command level.
    """

    @override
    def format_help(self) -> str:
        if self.prog == "dagnam":
            return format_root_help(banner=False)
        subparsers = _subparsers_action(self)
        if subparsers is not None:
            return _format_group_help(self, subparsers)
        return super().format_help() + f"\nDocs: {DOCS_URL}\n"

    def error(self, message: str) -> NoReturn:  # type: ignore[override]  # typeshed types error() -> None; we return NoReturn
        self.exit(2, self._format_error(message))

    def _format_error(self, message: str) -> str:
        is_root = self.prog == "dagnam"
        choice = _INVALID_CHOICE.search(message)
        if choice is not None:
            kind = "command" if is_root else "subcommand"
            candidates = re.findall(r"'([^']*)'", choice.group("choices"))
            return self._suggestion_block(choice.group("bad"), kind, candidates)
        unrecognized = _UNRECOGNIZED.search(message)
        if unrecognized is not None:
            bad = unrecognized.group("args").split()[0]
            candidates = list(self._option_string_actions)
            return self._suggestion_block(bad, "option", candidates)
        label = styled_error_label()
        return f"{label} {message}\n\nRun '{self.prog} --help' for usage.\nDocs: {DOCS_URL}\n"

    def _suggestion_block(self, bad: str, kind: str, candidates: list[str]) -> str:
        match = suggest(bad, candidates)
        lines = [f"{styled_error_label()} unknown {kind} '{bad}'", ""]
        if match is not None:
            lines += [f"  Did you mean '{match}'?", ""]
        lines += [f"Run '{self.prog} --help' to see all {kind}s.", f"Docs: {DOCS_URL}"]
        return "\n".join(lines) + "\n"
