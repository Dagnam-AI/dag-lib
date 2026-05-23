"""Command-line parser and entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import getpass


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``dagnam`` CLI parser."""
    from dagnam.cli.cache import cmd_cache_clear, cmd_cache_list
    from dagnam.cli.codegen import (
        cmd_codegen_download,
        cmd_codegen_generate,
        cmd_codegen_preview,
        cmd_codegen_validate,
    )
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
    from dagnam.cli.training import cmd_checkpoint_download, cmd_checkpoint_list, cmd_stream

    parser = argparse.ArgumentParser(prog="dagnam")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _login(args: argparse.Namespace) -> None:
        cmd_login(args, getpass.getpass)

    login = subparsers.add_parser("login")
    login.add_argument("--api-url")
    login.set_defaults(func=_login)

    dataset = subparsers.add_parser("dataset")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_list = dataset_sub.add_parser("list")
    dataset_list.add_argument("--api-url")
    dataset_list.add_argument("--api-key")
    dataset_list.add_argument("--type", default="all")
    dataset_list.add_argument("--search")
    dataset_list.set_defaults(func=cmd_dataset_list)
    dataset_info = dataset_sub.add_parser("info")
    dataset_info.add_argument("dataset_id")
    dataset_info.add_argument("--api-url")
    dataset_info.add_argument("--api-key")
    dataset_info.set_defaults(func=cmd_dataset_info)
    dataset_download = dataset_sub.add_parser("download")
    dataset_download.add_argument("dataset_id")
    dataset_download.add_argument("--output-dir", default=".")
    dataset_download.set_defaults(func=cmd_dataset_download)

    cache = subparsers.add_parser("cache")
    cache_sub = cache.add_subparsers(dest="cache_command", required=True)
    cache_sub.add_parser("list").set_defaults(func=cmd_cache_list)
    cache_sub.add_parser("clear").set_defaults(func=cmd_cache_clear)

    projects = subparsers.add_parser("projects")
    project_sub = projects.add_subparsers(dest="project_command", required=True)
    project_list = project_sub.add_parser("list")
    project_list.add_argument("--framework")
    project_list.add_argument("--search")
    project_list.add_argument("--page", type=int, default=1)
    project_list.add_argument("--limit", type=int, default=20)
    project_list.set_defaults(func=cmd_projects_list)
    project_get = project_sub.add_parser("get")
    project_get.add_argument("project_id")
    project_get.set_defaults(func=cmd_projects_get)
    project_create = project_sub.add_parser("create")
    project_create.add_argument("--title", required=True)
    project_create.add_argument("--framework", default="pytorch")
    project_create.add_argument("--description")
    project_create.add_argument("--visibility", default="private")
    project_create.set_defaults(func=cmd_projects_create)
    project_delete = project_sub.add_parser("delete")
    project_delete.add_argument("project_id")
    project_delete.set_defaults(func=cmd_projects_delete)
    project_dup = project_sub.add_parser("duplicate")
    project_dup.add_argument("project_id")
    project_dup.add_argument("--title")
    project_dup.set_defaults(func=cmd_projects_duplicate)

    deployments = subparsers.add_parser("deployments")
    deployment_sub = deployments.add_subparsers(dest="deployment_command", required=True)
    deployment_list = deployment_sub.add_parser("list")
    deployment_list.add_argument("--status")
    deployment_list.add_argument("--platform")
    deployment_list.add_argument("--project-id")
    deployment_list.add_argument("--search")
    deployment_list.add_argument("--page", type=int, default=1)
    deployment_list.add_argument("--limit", type=int, default=20)
    deployment_list.set_defaults(func=cmd_deployments_list)
    deployment_get = deployment_sub.add_parser("get")
    deployment_get.add_argument("deployment_id")
    deployment_get.set_defaults(func=cmd_deployments_get)
    deployment_create = deployment_sub.add_parser("create")
    deployment_create.add_argument("--project-id", required=True)
    deployment_create.add_argument("--name", required=True)
    deployment_create.add_argument("--checkpoint-path", required=True)
    deployment_create.add_argument("--platform", required=True)
    deployment_create.add_argument("--deployment-type", required=True)
    deployment_create.add_argument("--instance-type", required=True)
    deployment_create.add_argument("--num-instances", type=int, default=1)
    deployment_create.set_defaults(func=cmd_deployments_create)
    for command_name, handler in {
        "pause": cmd_deployments_pause,
        "resume": cmd_deployments_resume,
        "delete": cmd_deployments_delete,
        "logs": cmd_deployments_logs,
        "metrics": cmd_deployments_metrics,
    }.items():
        command = deployment_sub.add_parser(command_name)
        command.add_argument("deployment_id")
        if command_name == "logs":
            command.add_argument("--level")
            command.add_argument("--search")
            command.add_argument("--limit", type=int, default=100)
        if command_name == "metrics":
            command.add_argument("--time-range", default="24h")
        command.set_defaults(func=handler)

    inference = subparsers.add_parser("inference")
    inference_sub = inference.add_subparsers(dest="inference_command", required=True)
    run = inference_sub.add_parser("run")
    run.add_argument("deployment_id")
    run.add_argument("--input", required=True)
    run.set_defaults(func=cmd_inference_run)
    batch = inference_sub.add_parser("batch")
    batch.add_argument("deployment_id")
    batch.add_argument("--inputs", required=True)
    batch.set_defaults(func=cmd_inference_batch)
    health = inference_sub.add_parser("health")
    health.add_argument("deployment_id")
    health.set_defaults(func=cmd_inference_health)

    codegen = subparsers.add_parser("codegen")
    codegen_sub = codegen.add_subparsers(dest="codegen_command", required=True)
    for command_name, handler in {
        "generate": cmd_codegen_generate,
        "preview": cmd_codegen_preview,
        "validate": cmd_codegen_validate,
        "download": cmd_codegen_download,
    }.items():
        command = codegen_sub.add_parser(command_name)
        command.add_argument("project_id")
        command.add_argument("--framework", default="pytorch")
        command.add_argument("--version-id")
        command.add_argument("--async", action="store_true")
        command.add_argument("--output")
        command.set_defaults(func=handler)

    hub = subparsers.add_parser("hub")
    hub_sub = hub.add_subparsers(dest="hub_command", required=True)
    hub_search = hub_sub.add_parser("search")
    hub_search.add_argument("--search")
    hub_search.add_argument("--task-type")
    hub_search.add_argument("--framework")
    hub_search.add_argument("--sort-by", default="popular")
    hub_search.add_argument("--page", type=int, default=1)
    hub_search.add_argument("--limit", type=int, default=20)
    hub_search.set_defaults(func=cmd_hub_search)
    for command_name, handler in {
        "get": cmd_hub_get,
        "star": cmd_hub_star,
        "unstar": cmd_hub_unstar,
        "fork": cmd_hub_fork,
    }.items():
        command = hub_sub.add_parser(command_name)
        command.add_argument("model_id")
        command.set_defaults(func=handler)
    hub_sub.add_parser("featured").set_defaults(func=cmd_hub_featured)
    hub_trending = hub_sub.add_parser("trending")
    hub_trending.add_argument("--days", type=int, default=7)
    hub_trending.set_defaults(func=cmd_hub_trending)

    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint_sub = checkpoint.add_subparsers(dest="checkpoint_command", required=True)
    checkpoint_list = checkpoint_sub.add_parser("list")
    checkpoint_list.add_argument("job_id")
    checkpoint_list.set_defaults(func=cmd_checkpoint_list)
    checkpoint_download = checkpoint_sub.add_parser("download")
    checkpoint_download.add_argument("job_id")
    checkpoint_download.add_argument("checkpoint_id", nargs="?", default="latest")
    checkpoint_download.set_defaults(func=cmd_checkpoint_download)

    stream = subparsers.add_parser("stream")
    stream.add_argument("job_id")
    stream.add_argument("--heartbeats", action="store_true")
    stream.add_argument("--json", action="store_true")
    stream.set_defaults(func=cmd_stream)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "func")
    handler(args)
    return 0
