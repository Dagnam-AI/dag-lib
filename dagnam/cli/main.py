"""Command-line parser and entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``dagnam`` CLI parser.

    Each command group's argument registration lives beside its handlers in the
    matching ``dagnam.cli.<domain>`` module as a ``register_*`` function; this
    function wires them onto a single shared subparsers action in display order.
    """
    from dagnam.cli.account import register_account
    from dagnam.cli.agent import register_agent
    from dagnam.cli.cache import register_cache
    from dagnam.cli.checkpoint import register_checkpoint
    from dagnam.cli.codegen import register_codegen
    from dagnam.cli.common import format_ascii_art, format_version_banner
    from dagnam.cli.dataset import register_dataset
    from dagnam.cli.deployment import register_deployments
    from dagnam.cli.hub import register_hub
    from dagnam.cli.inference import register_inference
    from dagnam.cli.login import register_login
    from dagnam.cli.project import register_projects
    from dagnam.cli.training import register_training

    parser = argparse.ArgumentParser(
        prog="dagnam",
        description=(
            f"{format_ascii_art()}\n\n"
            "Official CLI for Dagnam.AI datasets, projects, deployments, and training."
        ),
        epilog=(
            "Examples:\n"
            "  dagnam login                         Authenticate with an API key\n"
            "  dagnam dataset list --search mnist   Search available datasets\n"
            "  dagnam projects create --title X     Create a new project\n"
            "  dagnam training create <pid> ...     Start a training job\n"
            "  dagnam training attach <jid> -- ... Attach local metrics to a child process\n"
            "  dagnam config set training_metrics_path <path>\n"
            "                                       Set the default local metrics JSONL path\n"
            "  dagnam usage                         Show plan usage and limits\n"
            "  dagnam deployments logs <id>         Tail a deployment's logs\n\n"
            "Docs: https://dagnam.ai/docs"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=format_version_banner(),
        help="Show the dagnam version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
