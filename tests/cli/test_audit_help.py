"""Every ``dagnam audit`` subcommand links its docs section and states each flag's default."""

from __future__ import annotations

import argparse
import re
from typing import TYPE_CHECKING

import pytest

from dagnam.cli._parser import DagnamArgumentParser
from dagnam.cli.main import build_parser

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, StrCapture

DOCS_PAGE = "https://dagnam.ai/docs/dag-lib/workload-audit"
DOCS_LINE = re.compile(rf"^Docs: {re.escape(DOCS_PAGE)}#([a-z0-9-]+)$")


def _audit_subparsers() -> dict[str, argparse.ArgumentParser]:
    root = build_parser()
    (command_action,) = (
        a
        for a in root._actions
        if isinstance(a, argparse._SubParsersAction)  # pyright: ignore[reportPrivateUsage]
    )
    audit = command_action.choices["audit"]
    (audit_action,) = (
        a
        for a in audit._actions
        if isinstance(a, argparse._SubParsersAction)  # pyright: ignore[reportPrivateUsage]
    )
    return dict(audit_action.choices)


SUBPARSERS = _audit_subparsers()


def _anchor(name: str) -> str:
    match = DOCS_LINE.match(SUBPARSERS[name].format_help().rstrip("\n").splitlines()[-1])
    assert match is not None
    return match.group(1)


def test_the_audit_group_has_the_commands_the_docs_describe() -> None:
    assert set(SUBPARSERS) == {"scan", "run", "status", "key", "cancel", "delete"}


@pytest.mark.parametrize("name", sorted(SUBPARSERS))
def test_every_subcommand_help_ends_with_its_docs_link(name: str) -> None:
    help_text = SUBPARSERS[name].format_help()

    assert _anchor(name)
    assert help_text.endswith("\n")


def test_the_anchors_are_distinct_and_name_headings_of_the_docs_page() -> None:
    anchors = [_anchor(n) for n in sorted(SUBPARSERS)]

    assert len(set(anchors)) == len(anchors)
    assert {
        f"dagnam-audit-{n}" for n in ("scan", "run", "status", "key", "cancel", "delete")
    } <= set(anchors)


def _flags_with_a_default() -> list[tuple[str, argparse.Action]]:
    """Every optional, non-boolean, non-required flag of every audit subcommand."""
    return [
        (name, action)
        for name, parser in sorted(SUBPARSERS.items())
        for action in parser._actions  # pyright: ignore[reportPrivateUsage]
        if action.option_strings
        and not isinstance(action, argparse._HelpAction)  # pyright: ignore[reportPrivateUsage]
        and action.nargs != 0
        and not action.required
    ]


def test_the_introspection_finds_the_flags_the_issue_names() -> None:
    found = {action.option_strings[0] for _, action in _flags_with_a_default()}

    assert {"--window", "--max-credits", "--workloads", "--floor"} <= found


@pytest.mark.parametrize(
    ("name", "action"),
    _flags_with_a_default(),
    ids=lambda v: v if isinstance(v, str) else v.option_strings[0],
)
def test_every_flag_help_states_its_default(name: str, action: argparse.Action) -> None:
    help_text = action.help or ""

    assert "default" in help_text.lower(), (name, action.option_strings, help_text)
    if action.default is not None:
        assert str(action.default) in help_text, (name, action.option_strings, help_text)


def test_the_flags_the_plan_names_state_the_real_defaults() -> None:
    from dagnam_contracts.audit.verdict import FLOOR_JSON, FLOOR_LABEL

    scan, run = SUBPARSERS["scan"].format_help(), SUBPARSERS["run"].format_help()
    flat = " ".join((scan + run).split())

    assert "default 30d" in flat
    assert f"{FLOOR_LABEL}" in flat
    assert f"{FLOOR_JSON}" in flat
    assert "default: all audited" in flat
    assert "rounded up to 100" in flat


def test_the_help_flag_prints_the_link_through_the_real_entry_point(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run_cli(["audit", "key", "--help"])

    assert excinfo.value.code == 0
    assert capsys.readouterr().out.rstrip().endswith(f"Docs: {DOCS_PAGE}#dagnam-audit-key")


def test_a_parser_without_a_docs_url_keeps_the_generic_link() -> None:
    assert (
        DagnamArgumentParser(prog="x")
        .format_help()
        .rstrip()
        .endswith("Docs: https://dagnam.ai/docs")
    )
