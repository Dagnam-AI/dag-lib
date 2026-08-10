"""Unit tests for the branded argparse parser layer."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import cast

import pytest

from dagnam.cli import _parser
from dagnam.cli.common import DOCS_URL


class TestSuggest:
    def test_returns_close_match(self) -> None:
        assert _parser.suggest("trainig", ["training", "dataset"]) == "training"

    def test_returns_none_when_no_close_match(self) -> None:
        assert _parser.suggest("zzzzzz", ["training", "dataset"]) is None

    def test_returns_none_for_empty_candidates(self) -> None:
        assert _parser.suggest("training", []) is None


class TestCommandGroups:
    def test_flat_list_matches_groups(self) -> None:
        flat = tuple(c for _title, cmds in _parser.COMMAND_GROUPS for c in cmds)
        assert flat == _parser.ALL_GROUPED_COMMANDS

    def test_no_duplicate_commands(self) -> None:
        flat = _parser.ALL_GROUPED_COMMANDS
        assert len(flat) == len(set(flat))


class TestFormatRootHelp:
    def test_compact_help_has_no_full_banner(self) -> None:
        out = _parser.format_root_help(banner=False)
        assert "Official CLI for Dagnam.AI" in out
        assert "Commands:" in out
        assert "  Auth:" in out
        assert "    login        Authenticate and store an API key." in out
        assert "    logout       Remove stored credentials." in out
        assert "    whoami       Show the current authenticated identity." in out
        assert "Examples:" in out
        assert DOCS_URL in out
        # Compact help must NOT contain the multi-line ASCII art block.
        assert "█" not in out  # full-block glyph used only in the banner

    def test_compact_help_lists_command_descriptions_by_category(self) -> None:
        out = _parser.format_root_help(banner=False)
        expected_lines = (
            "  Data:",
            "    dataset      Browse and download datasets.",
            "    cache        Inspect and clear the local dataset cache.",
            "  Models:",
            "    projects     Manage projects.",
            "    codegen      Generate model code from a project.",
            "    hub          Browse the model hub.",
            "    checkpoint   List and download training checkpoints.",
            "    inference    Run inference against a deployment.",
            "    deployments  Manage model deployments.",
            "  Training:",
            "    training     Create, inspect, and manage training jobs.",
            "    stream       Stream live training events.",
            "  Account:",
            "    usage        Show plan, usage, and remaining limits.",
            "    config       Inspect and update saved configuration.",
            "    version      Show version and environment info.",
            "    agent        Install the Dagnam Agent Skill into Claude Code / Codex.",
        )
        for line in expected_lines:
            assert line in out

    def test_banner_help_includes_ascii_art(self) -> None:
        from dagnam.cli.common import format_ascii_art

        out = _parser.format_root_help(banner=True)
        assert format_ascii_art() in out
        assert "Commands:" in out
        assert DOCS_URL in out


def _root() -> _parser.DagnamArgumentParser:
    parser = _parser.DagnamArgumentParser(prog="dagnam")
    sub = parser.add_subparsers(dest="command", metavar="<command>", required=False)
    sub.add_parser("training")
    sub.add_parser("dataset")
    return parser


class TestParserErrors:
    def test_unknown_command_suggests(self, capsys: object) -> None:
        parser = _root()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["trainig"])
        assert exc.value.code == 2  # type: ignore[union-attr]
        err = capsys.readouterr().err  # type: ignore[attr-defined]
        assert "unknown command 'trainig'" in err
        assert "Did you mean 'training'?" in err
        assert DOCS_URL in err

    def test_unknown_subcommand_suggests(self, capsys: object) -> None:
        parser = _parser.DagnamArgumentParser(prog="dagnam dataset")
        sub = parser.add_subparsers(dest="action", metavar="<action>", required=False)
        sub.add_parser("list")
        sub.add_parser("get")
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["lst"])
        assert exc.value.code == 2  # type: ignore[union-attr]
        err = capsys.readouterr().err  # type: ignore[attr-defined]
        assert "unknown subcommand 'lst'" in err
        assert "Did you mean 'list'?" in err

    def test_unknown_command_no_close_match_omits_suggestion(self, capsys: object) -> None:
        parser = _root()
        with pytest.raises(SystemExit):
            parser.parse_args(["zzzzzz"])
        err = capsys.readouterr().err  # type: ignore[attr-defined]
        assert "unknown command 'zzzzzz'" in err
        assert "Did you mean" not in err

    def test_unknown_option_suggests(self, capsys: object) -> None:
        parser = _parser.DagnamArgumentParser(prog="dagnam")
        parser.add_argument("--verbose", action="store_true")
        with pytest.raises(SystemExit):
            parser.parse_args(["--verboze"])
        err = capsys.readouterr().err  # type: ignore[attr-defined]
        assert "unknown option '--verboze'" in err
        assert "Did you mean '--verbose'?" in err

    def test_other_error_falls_through(self, capsys: object) -> None:
        parser = _parser.DagnamArgumentParser(prog="dagnam")
        parser.add_argument("name")
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([])
        assert exc.value.code == 2  # type: ignore[union-attr]
        err = capsys.readouterr().err  # type: ignore[attr-defined]
        assert "required" in err
        assert DOCS_URL in err


class TestParserHelp:
    def test_root_format_help_is_compact(self) -> None:
        parser = _parser.DagnamArgumentParser(prog="dagnam")
        out = parser.format_help()
        assert "Official CLI for Dagnam.AI" in out
        assert "Commands:" in out

    def test_subparser_help_appends_docs_footer(self) -> None:
        parser = _parser.DagnamArgumentParser(prog="dagnam dataset")
        out = parser.format_help()
        assert out.rstrip().endswith(DOCS_URL)

    def test_group_help_handles_missing_description_and_child_help(self) -> None:
        parser = _parser.DagnamArgumentParser(prog="dagnam custom")
        sub = parser.add_subparsers(dest="custom_command", metavar="<command>")
        sub.add_parser("run")
        sub.add_parser("show", help="Show custom data.")

        out = parser.format_help()

        assert "Usage: dagnam custom [-h] <command> ..." in out
        assert "Commands:" in out
        assert "  run  " in out
        assert "  show  Show custom data." in out
        assert "\n\nCommands:" in out
        assert out.rstrip().endswith(DOCS_URL)

    def test_choice_help_ignores_malformed_choice_actions(self) -> None:
        action = cast(
            "argparse.Action",
            SimpleNamespace(
                _choices_actions=(
                    SimpleNamespace(dest="bad", help=None),
                    SimpleNamespace(dest="good", help="Good command."),
                )
            ),
        )

        assert _parser._choice_help(action) == {"good": "Good command."}


class TestGroupingDrift:
    def test_every_registered_command_is_grouped(self) -> None:
        from dagnam.cli.main import build_parser

        parser = build_parser()
        subactions = [
            action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
        ]
        assert len(subactions) == 1
        registered = set(subactions[0].choices)
        grouped = set(_parser.ALL_GROUPED_COMMANDS)
        assert registered == grouped, (
            f"missing from COMMAND_GROUPS: {registered - grouped}; "
            f"stale in COMMAND_GROUPS: {grouped - registered}"
        )


class TestChoiceCandidateRecovery:
    """argparse's error *message* is not a stable interface.

    It renders ``(choose from 'a', 'b')`` on some Python versions and
    ``(choose from a, b)`` on others. Scraping only the quoted form yielded no
    candidates on an unquoted build, which silently dropped the
    "Did you mean ...?" line from every unknown-command error while the rest
    of the message still looked correct — a suggestion feature that had
    stopped suggesting.
    """

    def test_prefers_the_real_subparser_choices_over_the_message(self) -> None:
        parser = _root()
        # Deliberately unparseable as a choices list: the real answer must
        # come from the subparsers action, not from this string.
        assert "training" in parser._choice_candidates("<<garbage>>")

    def test_unquoted_message_rendering_is_parsed(self) -> None:
        parser = _parser.DagnamArgumentParser(prog="dagnam thing")
        assert parser._choice_candidates("alpha, beta") == ["alpha", "beta"]

    def test_quoted_message_rendering_is_parsed(self) -> None:
        parser = _parser.DagnamArgumentParser(prog="dagnam thing")
        assert parser._choice_candidates("'alpha', 'beta'") == ["alpha", "beta"]

    def test_empty_choices_yield_no_candidates(self) -> None:
        parser = _parser.DagnamArgumentParser(prog="dagnam thing")
        assert parser._choice_candidates("") == []

    def test_a_plain_choices_argument_still_suggests(self, capsys: object) -> None:
        # No subparsers action at all, so the message is the only source.
        parser = _parser.DagnamArgumentParser(prog="dagnam thing")
        parser.add_argument("mode", choices=["alpha", "beta"])
        with pytest.raises(SystemExit):
            parser.parse_args(["alpah"])
        err = capsys.readouterr().err  # type: ignore[attr-defined]
        assert "Did you mean 'alpha'?" in err
