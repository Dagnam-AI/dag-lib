"""Command-line parser and entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys
from typing import TYPE_CHECKING

from dagnam.cli._parser import DagnamArgumentParser

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction


def build_parser() -> DagnamArgumentParser:
    """Build the top-level ``dagnam`` CLI parser.

    Each command group registers its arguments via a ``register_*`` function in
    the matching ``dagnam.cli.<domain>`` module; this wires them onto one shared
    subparsers action in display order. The root uses ``DagnamArgumentParser`` so
    every subcommand inherits typo suggestions and grouped help.
    """
    from dagnam.cli.account import register_account
    from dagnam.cli.agent import register_agent
    from dagnam.cli.cache import register_cache
    from dagnam.cli.checkpoint import register_checkpoint
    from dagnam.cli.codegen import register_codegen
    from dagnam.cli.common import format_version_banner
    from dagnam.cli.dataset import register_dataset
    from dagnam.cli.deployment import register_deployments
    from dagnam.cli.hub import register_hub
    from dagnam.cli.inference import register_inference
    from dagnam.cli.login import register_login
    from dagnam.cli.project import register_projects
    from dagnam.cli.training import register_training

    parser = DagnamArgumentParser(
        prog="dagnam",
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=format_version_banner(),
        help="Show the dagnam version and exit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show full tracebacks on error.",
    )
    # _SubParsersAction is invariant in typeshed but DagnamArgumentParser IS an
    # ArgumentParser; cast so register_* functions (typed SubParsersAction) accept it.
    subparsers: SubParsersAction = parser.add_subparsers(  # pyright: ignore[reportAssignmentType]
        dest="command", metavar="<command>", required=False
    )

    register_login(subparsers)
    register_dataset(subparsers)
    register_cache(subparsers)
    register_projects(subparsers)
    register_deployments(subparsers)
    register_inference(subparsers)
    register_codegen(subparsers)
    register_hub(subparsers)
    register_checkpoint(subparsers)
    register_training(subparsers)
    register_account(subparsers)
    register_agent(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    from dagnam.cli._parser import format_root_help
    from dagnam.cli.common import configure_console_encoding, run_command

    # Upgrade the console to UTF-8 before the banner is built so the branded art
    # renders on capable terminals; it degrades to ASCII on cp1252 (G019).
    configure_console_encoding()
    parser = build_parser()
    tokens = list(sys.argv[1:] if argv is None else argv)
    # argparse does not strip the ``--`` separator for *nested* subparsers
    # (CPython gh-72795), so ``dagnam training attach <job> -- python train.py``
    # would fail with "unrecognized arguments: --". Split it off ourselves and
    # hand the trailing tokens to the subcommand's ``command`` passthrough.
    passthrough: list[str] | None = None
    if "--" in tokens:
        split = tokens.index("--")
        tokens, passthrough = tokens[:split], tokens[split + 1 :]
    args = parser.parse_args(tokens)
    if passthrough is not None:
        # Only ``attach`` declares a ``command`` *positional* (nargs="*"), which
        # materializes as a list; the top-level subparsers ``dest="command"`` is
        # the subcommand-name string. So a list is our signal that a trailing
        # command is accepted here -- anything else is a usage error.
        existing = getattr(args, "command", None)
        if not isinstance(existing, list):
            parser.error("the '--' command separator is not valid here")
        args.command = [*existing, *passthrough]
    if not hasattr(args, "func"):
        # Bare ``dagnam``: friendly welcome with the full banner.
        print(format_root_help(banner=True), end="")
        return 0
    return run_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
