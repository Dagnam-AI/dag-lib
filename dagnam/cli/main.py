"""Command-line parser and entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import getpass


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``dagnam`` CLI parser."""
    from dagnam.cli.account import (
        cmd_config_get,
        cmd_config_list,
        cmd_config_set,
        cmd_config_unset,
        cmd_logout,
        cmd_version,
        cmd_whoami,
    )
    from dagnam.cli.cache import cmd_cache_clear, cmd_cache_list
    from dagnam.cli.codegen import (
        cmd_codegen_download,
        cmd_codegen_generate,
        cmd_codegen_preview,
        cmd_codegen_validate,
    )
    from dagnam.cli.common import resolve_version
    from dagnam.cli.dataset import cmd_dataset_download, cmd_dataset_info, cmd_dataset_list
    from dagnam.cli.deployment import (
        cmd_deployments_create,
        cmd_deployments_delete,
        cmd_deployments_get,
        cmd_deployments_list,
        cmd_deployments_logs,
        cmd_deployments_metrics,
        cmd_deployments_pause,
        cmd_deployments_resume,
    )
    from dagnam.cli.hub import (
        cmd_hub_featured,
        cmd_hub_fork,
        cmd_hub_get,
        cmd_hub_search,
        cmd_hub_star,
        cmd_hub_trending,
        cmd_hub_unstar,
    )
    from dagnam.cli.inference import (
        cmd_inference_batch,
        cmd_inference_health,
        cmd_inference_run,
    )
    from dagnam.cli.login import cmd_login
    from dagnam.cli.project import (
        cmd_projects_create,
        cmd_projects_delete,
        cmd_projects_duplicate,
        cmd_projects_get,
        cmd_projects_list,
    )
    from dagnam.cli.training import (
        cmd_checkpoint_download,
        cmd_checkpoint_list,
        cmd_stream,
        cmd_training_attach,
    )

    parser = argparse.ArgumentParser(
        prog="dagnam",
        description="Official CLI for Dagnam.AI datasets, projects, deployments, and training.",
        epilog=(
            "Examples:\n"
            "  dagnam login                         Authenticate with an API key\n"
            "  dagnam dataset list --search mnist   Search available datasets\n"
            "  dagnam projects create --title X     Create a new project\n"
            "  dagnam deployments logs <id>         Tail a deployment's logs\n\n"
            "Docs: https://dagnam.ai/docs"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"dagnam {resolve_version()}",
        help="Show the dagnam version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_collection_output_args(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--verbose",
            action="store_true",
            help="Print the full JSON response instead of the concise table.",
        )
        command.add_argument("--output", help="Save the full JSON response to this path.")

    def _login(args: argparse.Namespace) -> None:
        cmd_login(args, getpass.getpass)

    login = subparsers.add_parser(
        "login",
        help="Authenticate and store an API key.",
        description="Log in to Dagnam.AI and save credentials to ~/.dagnam/config.json.",
    )
    login.add_argument("--api-url", help="API base URL (default: https://api.dagnam.ai).")
    login.add_argument(
        "--training-metrics-path",
        help="Persist the local generated-training metrics JSONL path.",
    )
    login.set_defaults(func=_login)

    dataset = subparsers.add_parser(
        "dataset",
        help="Browse and download datasets.",
        description="List, inspect, and download datasets from Dagnam.AI.",
    )
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_list = dataset_sub.add_parser(
        "list", help="List datasets.", description="List available datasets."
    )
    dataset_list.add_argument("--api-url", help="Override the API base URL.")
    dataset_list.add_argument("--api-key", help="Override the API key.")
    dataset_list.add_argument(
        "--type", default="all", help="Filter by dataset type (default: all)."
    )
    dataset_list.add_argument("--search", help="Filter by name/description substring.")
    dataset_list.set_defaults(func=cmd_dataset_list)
    dataset_info = dataset_sub.add_parser(
        "info", help="Show dataset metadata.", description="Show metadata for one dataset."
    )
    dataset_info.add_argument("dataset_id", help="ID of the dataset to inspect.")
    dataset_info.add_argument("--api-url", help="Override the API base URL.")
    dataset_info.add_argument("--api-key", help="Override the API key.")
    dataset_info.set_defaults(func=cmd_dataset_info)
    dataset_download = dataset_sub.add_parser(
        "download",
        help="Download a dataset.",
        description="Download a dataset to a local directory.",
    )
    dataset_download.add_argument("dataset_id", help="ID of the dataset to download.")
    dataset_download.add_argument(
        "--output-dir", default=".", help="Destination directory (default: current dir)."
    )
    dataset_download.set_defaults(func=cmd_dataset_download)

    cache = subparsers.add_parser(
        "cache",
        help="Inspect and clear the local dataset cache.",
        description="Manage the on-disk dataset cache.",
    )
    cache_sub = cache.add_subparsers(dest="cache_command", required=True)
    cache_sub.add_parser(
        "list", help="List cached datasets.", description="List datasets in the local cache."
    ).set_defaults(func=cmd_cache_list)
    cache_sub.add_parser(
        "clear", help="Delete the local cache.", description="Remove all cached datasets."
    ).set_defaults(func=cmd_cache_clear)

    projects = subparsers.add_parser(
        "projects",
        help="Manage projects.",
        description="Create, list, inspect, and delete projects.",
    )
    project_sub = projects.add_subparsers(dest="project_command", required=True)
    project_list = project_sub.add_parser(
        "list", help="List projects.", description="List your projects."
    )
    project_list.add_argument("--framework", help="Filter by framework (e.g. pytorch).")
    project_list.add_argument("--search", help="Filter by title/description substring.")
    project_list.add_argument("--page", type=int, default=1, help="Page number (default: 1).")
    project_list.add_argument(
        "--limit", type=int, default=20, help="Results per page (default: 20)."
    )
    _add_collection_output_args(project_list)
    project_list.set_defaults(func=cmd_projects_list)
    project_get = project_sub.add_parser(
        "get", help="Show a project.", description="Show details for one project."
    )
    project_get.add_argument("project_id", help="ID of the project.")
    project_get.set_defaults(func=cmd_projects_get)
    project_create = project_sub.add_parser(
        "create", help="Create a project.", description="Create a new project."
    )
    project_create.add_argument("--title", required=True, help="Project title (required).")
    project_create.add_argument(
        "--framework", default="pytorch", help="Framework (default: pytorch)."
    )
    project_create.add_argument("--description", help="Optional project description.")
    project_create.add_argument(
        "--visibility", default="private", help="private or public (default: private)."
    )
    project_create.set_defaults(func=cmd_projects_create)
    project_delete = project_sub.add_parser(
        "delete", help="Delete a project.", description="Delete a project permanently."
    )
    project_delete.add_argument("project_id", help="ID of the project to delete.")
    project_delete.set_defaults(func=cmd_projects_delete)
    project_dup = project_sub.add_parser(
        "duplicate", help="Duplicate a project.", description="Clone an existing project."
    )
    project_dup.add_argument("project_id", help="ID of the project to duplicate.")
    project_dup.add_argument("--title", help="Title for the copy (defaults to a derived name).")
    project_dup.set_defaults(func=cmd_projects_duplicate)

    deployments = subparsers.add_parser(
        "deployments",
        help="Manage model deployments.",
        description="Create, control, and observe deployments.",
    )
    deployment_sub = deployments.add_subparsers(dest="deployment_command", required=True)
    deployment_list = deployment_sub.add_parser(
        "list", help="List deployments.", description="List your deployments."
    )
    deployment_list.add_argument("--status", help="Filter by status.")
    deployment_list.add_argument("--platform", help="Filter by platform.")
    deployment_list.add_argument("--project-id", help="Filter by project ID.")
    deployment_list.add_argument("--search", help="Filter by name substring.")
    deployment_list.add_argument("--page", type=int, default=1, help="Page number (default: 1).")
    deployment_list.add_argument(
        "--limit", type=int, default=20, help="Results per page (default: 20)."
    )
    _add_collection_output_args(deployment_list)
    deployment_list.set_defaults(func=cmd_deployments_list)
    deployment_get = deployment_sub.add_parser(
        "get", help="Show a deployment.", description="Show details for one deployment."
    )
    deployment_get.add_argument("deployment_id", help="ID of the deployment.")
    deployment_get.set_defaults(func=cmd_deployments_get)
    deployment_create = deployment_sub.add_parser(
        "create", help="Create a deployment.", description="Create a new deployment."
    )
    deployment_create.add_argument("--project-id", required=True, help="Project ID (required).")
    deployment_create.add_argument("--name", required=True, help="Deployment name (required).")
    deployment_create.add_argument(
        "--checkpoint-path", required=True, help="Checkpoint path to deploy (required)."
    )
    deployment_create.add_argument("--platform", required=True, help="Target platform (required).")
    deployment_create.add_argument(
        "--deployment-type", required=True, help="Deployment type (required)."
    )
    deployment_create.add_argument(
        "--instance-type", required=True, help="Compute instance type (required)."
    )
    deployment_create.add_argument(
        "--num-instances", type=int, default=1, help="Instance count (default: 1)."
    )
    deployment_create.set_defaults(func=cmd_deployments_create)
    deployment_help = {
        "pause": ("Pause a deployment.", "Pause a running deployment."),
        "resume": ("Resume a deployment.", "Resume a paused deployment."),
        "delete": ("Delete a deployment.", "Delete a deployment permanently."),
        "logs": ("Show deployment logs.", "Fetch logs for a deployment."),
        "metrics": ("Show deployment metrics.", "Fetch metrics for a deployment."),
    }
    for command_name, handler in {
        "pause": cmd_deployments_pause,
        "resume": cmd_deployments_resume,
        "delete": cmd_deployments_delete,
        "logs": cmd_deployments_logs,
        "metrics": cmd_deployments_metrics,
    }.items():
        short_help, long_help = deployment_help[command_name]
        command = deployment_sub.add_parser(
            command_name, help=short_help, description=long_help
        )
        command.add_argument("deployment_id", help="ID of the deployment.")
        if command_name == "logs":
            command.add_argument("--level", help="Filter by log level.")
            command.add_argument("--search", help="Filter by message substring.")
            command.add_argument(
                "--limit", type=int, default=100, help="Max lines (default: 100)."
            )
        if command_name == "metrics":
            command.add_argument(
                "--time-range", default="24h", help="Window, e.g. 24h (default: 24h)."
            )
        command.set_defaults(func=handler)

    inference = subparsers.add_parser(
        "inference",
        help="Run inference against a deployment.",
        description="Run single or batch inference and check deployment health.",
    )
    inference_sub = inference.add_subparsers(dest="inference_command", required=True)
    run = inference_sub.add_parser(
        "run", help="Run one inference.", description="Send one input to a deployment."
    )
    run.add_argument("deployment_id", help="ID of the deployment.")
    run.add_argument(
        "--input", required=True, help="JSON literal, or @path to a JSON file (required)."
    )
    run.set_defaults(func=cmd_inference_run)
    batch = inference_sub.add_parser(
        "batch", help="Run batch inference.", description="Send multiple inputs to a deployment."
    )
    batch.add_argument("deployment_id", help="ID of the deployment.")
    batch.add_argument(
        "--inputs", required=True, help="JSON array literal, or @path to a JSON file (required)."
    )
    batch.set_defaults(func=cmd_inference_batch)
    health = inference_sub.add_parser(
        "health", help="Check deployment health.", description="Report deployment health status."
    )
    health.add_argument("deployment_id", help="ID of the deployment.")
    health.set_defaults(func=cmd_inference_health)

    codegen = subparsers.add_parser(
        "codegen",
        help="Generate model code from a project.",
        description="Generate, preview, validate, or download model code.",
    )
    codegen_sub = codegen.add_subparsers(dest="codegen_command", required=True)
    codegen_help = {
        "generate": ("Generate code.", "Generate model code for a project."),
        "preview": ("Preview code.", "Preview generated code without saving."),
        "validate": ("Validate code.", "Validate that a project can generate code."),
        "download": ("Download code.", "Download generated code to a file."),
    }
    for command_name, handler in {
        "generate": cmd_codegen_generate,
        "preview": cmd_codegen_preview,
        "validate": cmd_codegen_validate,
        "download": cmd_codegen_download,
    }.items():
        short_help, long_help = codegen_help[command_name]
        command = codegen_sub.add_parser(
            command_name, help=short_help, description=long_help
        )
        command.add_argument("project_id", help="ID of the project.")
        command.add_argument(
            "--framework", default="pytorch", help="Framework (default: pytorch)."
        )
        command.add_argument("--version-id", help="Specific project version ID.")
        command.add_argument(
            "--async", action="store_true", help="Run asynchronously and return a job."
        )
        command.add_argument("--output", help="Write output to this path.")
        command.set_defaults(func=handler)

    hub = subparsers.add_parser(
        "hub",
        help="Browse the model hub.",
        description="Search, inspect, star, and fork models on the hub.",
    )
    hub_sub = hub.add_subparsers(dest="hub_command", required=True)
    hub_search = hub_sub.add_parser(
        "search", help="Search the hub.", description="Search models on the hub."
    )
    hub_search.add_argument("--search", help="Query string.")
    hub_search.add_argument("--task-type", help="Filter by task type.")
    hub_search.add_argument("--framework", help="Filter by framework.")
    hub_search.add_argument(
        "--sort-by", default="popular", help="Sort order (default: popular)."
    )
    hub_search.add_argument("--page", type=int, default=1, help="Page number (default: 1).")
    hub_search.add_argument(
        "--limit", type=int, default=20, help="Results per page (default: 20)."
    )
    _add_collection_output_args(hub_search)
    hub_search.set_defaults(func=cmd_hub_search)
    hub_help = {
        "get": ("Show a hub model.", "Show details for one hub model."),
        "star": ("Star a model.", "Star a hub model."),
        "unstar": ("Unstar a model.", "Remove a star from a hub model."),
        "fork": ("Fork a model.", "Fork a hub model into your account."),
    }
    for command_name, handler in {
        "get": cmd_hub_get,
        "star": cmd_hub_star,
        "unstar": cmd_hub_unstar,
        "fork": cmd_hub_fork,
    }.items():
        short_help, long_help = hub_help[command_name]
        command = hub_sub.add_parser(command_name, help=short_help, description=long_help)
        command.add_argument("model_id", help="ID of the hub model.")
        command.set_defaults(func=handler)
    hub_featured = hub_sub.add_parser(
        "featured", help="List featured models.", description="List featured hub models."
    )
    _add_collection_output_args(hub_featured)
    hub_featured.set_defaults(func=cmd_hub_featured)
    hub_trending = hub_sub.add_parser(
        "trending", help="List trending models.", description="List trending hub models."
    )
    hub_trending.add_argument(
        "--days", type=int, default=7, help="Trailing window in days (default: 7)."
    )
    _add_collection_output_args(hub_trending)
    hub_trending.set_defaults(func=cmd_hub_trending)

    checkpoint = subparsers.add_parser(
        "checkpoint",
        help="List and download training checkpoints.",
        description="List or download checkpoints for a training job.",
    )
    checkpoint_sub = checkpoint.add_subparsers(dest="checkpoint_command", required=True)
    checkpoint_list = checkpoint_sub.add_parser(
        "list", help="List checkpoints.", description="List checkpoints for a job."
    )
    checkpoint_list.add_argument("job_id", help="ID of the training job.")
    checkpoint_list.set_defaults(func=cmd_checkpoint_list)
    checkpoint_download = checkpoint_sub.add_parser(
        "download", help="Download a checkpoint.", description="Download a job checkpoint."
    )
    checkpoint_download.add_argument("job_id", help="ID of the training job.")
    checkpoint_download.add_argument(
        "checkpoint_id",
        nargs="?",
        default="latest",
        help="Checkpoint ID (default: latest).",
    )
    checkpoint_download.set_defaults(func=cmd_checkpoint_download)

    stream = subparsers.add_parser(
        "stream",
        help="Stream live training events.",
        description="Stream training events for a job over SSE.",
    )
    stream.add_argument("job_id", help="ID of the training job.")
    stream.add_argument(
        "--heartbeats", action="store_true", help="Include heartbeat events."
    )
    stream.add_argument("--json", action="store_true", help="Emit raw JSON events.")
    stream.set_defaults(func=cmd_stream)

    training_cmd = subparsers.add_parser(
        "training",
        help="Manage training job helpers.",
        description="Attach local training metrics to a Dagnam job.",
    )
    training_sub = training_cmd.add_subparsers(dest="training_command", required=True)
    attach = training_sub.add_parser(
        "attach",
        help="Upload local JSONL training metrics for a job.",
        description=(
            "Attach a local training run to a Dagnam job. Use '--' before a "
            "training command, for example: dagnam training attach <job-id> -- python train.py"
        ),
    )
    attach.add_argument("job_id", help="ID of the training job.")
    attach.add_argument("--metrics-path", help="Metrics JSONL path to watch.")
    attach.add_argument("--replay", action="store_true", help="Upload existing file contents first.")
    attach.add_argument("command", nargs="*", help="Command to run after '--'.")
    attach.set_defaults(func=cmd_training_attach)

    version_cmd = subparsers.add_parser(
        "version",
        help="Show version and environment info.",
        description="Print the dagnam version, Python version, and platform.",
    )
    version_cmd.set_defaults(func=cmd_version)

    whoami = subparsers.add_parser(
        "whoami",
        help="Show the current authenticated identity.",
        description="Show the resolved API URL, masked API key, and its source.",
    )
    whoami.set_defaults(func=cmd_whoami)

    logout = subparsers.add_parser(
        "logout",
        help="Remove stored credentials.",
        description="Remove the stored API key from ~/.dagnam/config.json.",
    )
    logout.set_defaults(func=cmd_logout)

    config_cmd = subparsers.add_parser(
        "config",
        help="Inspect and update saved configuration.",
        description="Read and update supported values in ~/.dagnam/config.json.",
    )
    config_sub = config_cmd.add_subparsers(dest="config_command", required=True)
    config_list = config_sub.add_parser(
        "list", help="Print all config values.", description="Print config (api_key masked)."
    )
    config_list.set_defaults(func=cmd_config_list)
    config_get = config_sub.add_parser(
        "get", help="Print one config value.", description="Print a single config value."
    )
    config_get.add_argument("key", help="Config key to read, e.g. api_url.")
    config_get.set_defaults(func=cmd_config_get)
    config_set = config_sub.add_parser(
        "set",
        help="Set a config value.",
        description="Set a supported config value such as training_metrics_path.",
    )
    config_set.add_argument("key", help="Config key to set, e.g. training_metrics_path.")
    config_set.add_argument("value", help="Value to save.")
    config_set.set_defaults(func=cmd_config_set)
    config_unset = config_sub.add_parser(
        "unset",
        help="Unset a config value.",
        description="Unset a supported config value such as training_metrics_path.",
    )
    config_unset.add_argument("key", help="Config key to unset, e.g. training_metrics_path.")
    config_unset.set_defaults(func=cmd_config_unset)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0
