"""Command-line interface for the dagnam client library."""

from __future__ import annotations

import argparse
import getpass
import sys

from dagnam.cli.cache import _cmd_cache_clear, _cmd_cache_list
from dagnam.cli.codegen import (
    _cmd_codegen_download,
    _cmd_codegen_generate,
    _cmd_codegen_preview,
    _cmd_codegen_validate,
)
from dagnam.cli.dataset import _cmd_dataset_download, _cmd_dataset_info, _cmd_dataset_list
from dagnam.cli.deployment import (
    _cmd_deployments_create,
    _cmd_deployments_delete,
    _cmd_deployments_get,
    _cmd_deployments_list,
    _cmd_deployments_logs,
    _cmd_deployments_metrics,
    _cmd_deployments_pause,
    _cmd_deployments_resume,
)
from dagnam.cli.hub import (
    _cmd_hub_featured,
    _cmd_hub_fork,
    _cmd_hub_get,
    _cmd_hub_search,
    _cmd_hub_star,
    _cmd_hub_trending,
    _cmd_hub_unstar,
)
from dagnam.cli.inference import (
    _cmd_inference_batch,
    _cmd_inference_health,
    _cmd_inference_run,
)
from dagnam.cli.login import _cmd_login
from dagnam.cli.project import (
    _cmd_projects_create,
    _cmd_projects_delete,
    _cmd_projects_duplicate,
    _cmd_projects_get,
    _cmd_projects_list,
)
from dagnam.cli.training import _cmd_checkpoint_download, _cmd_checkpoint_list, _cmd_stream


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dagnam",
        description="Dagnam.AI CLI — manage datasets and local cache.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- login ---
    login_parser = subparsers.add_parser("login", help="Save API key")
    login_parser.add_argument(
        "--api-url",
        default=None,
        help="Custom API URL (default: https://api.dagnam.ai)",
    )

    # --- dataset ---
    dataset_parser = subparsers.add_parser("dataset", help="Dataset operations")
    ds_sub = dataset_parser.add_subparsers(dest="dataset_command")

    ds_sub.add_parser("list", help="List available datasets")

    dl_parser = ds_sub.add_parser("download", help="Download a dataset")
    dl_parser.add_argument("dataset_id", help="Dataset ID to download")

    info_parser = ds_sub.add_parser("info", help="Show dataset metadata")
    info_parser.add_argument("dataset_id", help="Dataset ID to inspect")

    # --- cache ---
    cache_parser = subparsers.add_parser("cache", help="Local cache operations")
    cache_sub = cache_parser.add_subparsers(dest="cache_command")

    cache_sub.add_parser("list", help="List cached datasets")
    cache_sub.add_parser("clear", help="Delete all cached datasets")

    # --- inference ---
    inf_parser = subparsers.add_parser("inference", help="Call a deployed model")
    inf_sub = inf_parser.add_subparsers(dest="inference_command")

    inf_run = inf_sub.add_parser("run", help="Single prediction")
    inf_run.add_argument("deployment_id", help="Deployment ID")
    inf_run.add_argument(
        "--input",
        required=True,
        help='JSON literal (e.g. \'{"text":"hi"}\') or @path/to/input.json',
    )

    inf_batch = inf_sub.add_parser("batch", help="Batch prediction")
    inf_batch.add_argument("deployment_id", help="Deployment ID")
    inf_batch.add_argument(
        "--inputs",
        required=True,
        help="JSON array literal or @path/to/inputs.json",
    )

    inf_health = inf_sub.add_parser("health", help="Check deployment health")
    inf_health.add_argument("deployment_id", help="Deployment ID")

    # --- checkpoint ---
    ck_parser = subparsers.add_parser("checkpoint", help="Training checkpoint operations")
    ck_sub = ck_parser.add_subparsers(dest="checkpoint_command")

    ck_list = ck_sub.add_parser("list", help="List checkpoints for a training job")
    ck_list.add_argument("job_id", help="Training job ID")

    ck_dl = ck_sub.add_parser("download", help="Download a checkpoint")
    ck_dl.add_argument("job_id", help="Training job ID")
    ck_dl.add_argument(
        "checkpoint_id",
        nargs="?",
        default=None,
        help="Checkpoint ID (default: latest/best)",
    )

    # --- stream ---
    stream_parser = subparsers.add_parser("stream", help="Tail live training events via SSE")
    stream_parser.add_argument("job_id", help="Training job ID")
    stream_parser.add_argument("--json", action="store_true", help="Emit one JSON object per line")
    stream_parser.add_argument("--heartbeats", action="store_true", help="Include heartbeat events")

    # --- deployments ---
    dep_parser = subparsers.add_parser("deployments", help="Deployment operations")
    dep_sub = dep_parser.add_subparsers(dest="deployments_command")

    dep_list = dep_sub.add_parser("list", help="List deployments")
    dep_list.add_argument("--status")
    dep_list.add_argument("--platform")
    dep_list.add_argument("--project-id")
    dep_list.add_argument("--search")
    dep_list.add_argument("--page", type=int)
    dep_list.add_argument("--limit", type=int)

    dep_get = dep_sub.add_parser("get", help="Get deployment details")
    dep_get.add_argument("deployment_id")

    dep_create = dep_sub.add_parser("create", help="Create a deployment")
    dep_create.add_argument("--name", required=True)
    dep_create.add_argument("--project-id", required=True)
    dep_create.add_argument("--checkpoint-path", required=True)
    dep_create.add_argument("--platform", required=True)
    dep_create.add_argument("--deployment-type", required=True)
    dep_create.add_argument("--instance-type", required=True)
    dep_create.add_argument("--num-instances", type=int)

    dep_pause = dep_sub.add_parser("pause", help="Pause a deployment")
    dep_pause.add_argument("deployment_id")

    dep_resume = dep_sub.add_parser("resume", help="Resume a deployment")
    dep_resume.add_argument("deployment_id")

    dep_del = dep_sub.add_parser("delete", help="Delete a deployment")
    dep_del.add_argument("deployment_id")

    dep_logs = dep_sub.add_parser("logs", help="Get deployment logs")
    dep_logs.add_argument("deployment_id")
    dep_logs.add_argument("--level")
    dep_logs.add_argument("--search")
    dep_logs.add_argument("--limit", type=int)

    dep_metrics = dep_sub.add_parser("metrics", help="Get deployment metrics")
    dep_metrics.add_argument("deployment_id")
    dep_metrics.add_argument("--time-range")

    # --- hub ---
    hub_parser = subparsers.add_parser("hub", help="Model hub operations")
    hub_sub = hub_parser.add_subparsers(dest="hub_command")

    hub_search = hub_sub.add_parser("search", help="Search models")
    hub_search.add_argument("--search")
    hub_search.add_argument("--framework")
    hub_search.add_argument("--task-type")
    hub_search.add_argument("--sort-by")
    hub_search.add_argument("--page", type=int)
    hub_search.add_argument("--limit", type=int)

    hub_get = hub_sub.add_parser("get", help="Get model details")
    hub_get.add_argument("model_id")

    hub_star = hub_sub.add_parser("star", help="Star a model")
    hub_star.add_argument("model_id")

    hub_unstar = hub_sub.add_parser("unstar", help="Unstar a model")
    hub_unstar.add_argument("model_id")

    hub_fork = hub_sub.add_parser("fork", help="Fork a model")
    hub_fork.add_argument("model_id")

    hub_sub.add_parser("featured", help="List featured models")

    hub_trending = hub_sub.add_parser("trending", help="List trending models")
    hub_trending.add_argument("--days", type=int)

    # --- projects ---
    proj_parser = subparsers.add_parser("projects", help="Project operations")
    proj_sub = proj_parser.add_subparsers(dest="projects_command")

    proj_list = proj_sub.add_parser("list", help="List projects")
    proj_list.add_argument("--framework")
    proj_list.add_argument("--search")
    proj_list.add_argument("--page", type=int)
    proj_list.add_argument("--limit", type=int)

    proj_get = proj_sub.add_parser("get", help="Get project details")
    proj_get.add_argument("project_id")

    proj_create = proj_sub.add_parser("create", help="Create a project")
    proj_create.add_argument("--title", required=True)
    proj_create.add_argument("--framework")
    proj_create.add_argument("--description")
    proj_create.add_argument("--visibility")

    proj_del = proj_sub.add_parser("delete", help="Delete a project")
    proj_del.add_argument("project_id")

    proj_dup = proj_sub.add_parser("duplicate", help="Duplicate a project")
    proj_dup.add_argument("project_id")
    proj_dup.add_argument("--title")

    # --- codegen ---
    cg_parser = subparsers.add_parser("codegen", help="Code generation operations")
    cg_sub = cg_parser.add_subparsers(dest="codegen_command")

    cg_gen = cg_sub.add_parser("generate", help="Generate code")
    cg_gen.add_argument("project_id")
    cg_gen.add_argument("--framework")
    cg_gen.add_argument("--version-id")
    cg_gen.add_argument("--async", action="store_true", dest="async")

    cg_preview = cg_sub.add_parser("preview", help="Preview generated code")
    cg_preview.add_argument("project_id")
    cg_preview.add_argument("--framework")
    cg_preview.add_argument("--version-id")

    cg_validate = cg_sub.add_parser("validate", help="Validate project for codegen")
    cg_validate.add_argument("project_id")
    cg_validate.add_argument("--version-id")

    cg_download = cg_sub.add_parser("download", help="Download generated code")
    cg_download.add_argument("project_id")
    cg_download.add_argument("--framework")
    cg_download.add_argument("--version-id")
    cg_download.add_argument("--output")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(2)

    if args.command == "login":
        _cmd_login(args, getpass.getpass)
    elif args.command == "dataset":
        if getattr(args, "dataset_command", None) is None:
            parser.parse_args(["dataset", "--help"])
        elif args.dataset_command == "list":
            _cmd_dataset_list(args)
        elif args.dataset_command == "download":
            _cmd_dataset_download(args)
        elif args.dataset_command == "info":
            _cmd_dataset_info(args)
    elif args.command == "cache":
        if getattr(args, "cache_command", None) is None:
            parser.parse_args(["cache", "--help"])
        elif args.cache_command == "list":
            _cmd_cache_list(args)
        elif args.cache_command == "clear":
            _cmd_cache_clear(args)
    elif args.command == "inference":
        if getattr(args, "inference_command", None) is None:
            parser.parse_args(["inference", "--help"])
        elif args.inference_command == "run":
            _cmd_inference_run(args)
        elif args.inference_command == "batch":
            _cmd_inference_batch(args)
        elif args.inference_command == "health":
            _cmd_inference_health(args)
    elif args.command == "checkpoint":
        if getattr(args, "checkpoint_command", None) is None:
            parser.parse_args(["checkpoint", "--help"])
        elif args.checkpoint_command == "list":
            _cmd_checkpoint_list(args)
        elif args.checkpoint_command == "download":
            _cmd_checkpoint_download(args)
    elif args.command == "stream":
        _cmd_stream(args)
    elif args.command == "deployments":
        if getattr(args, "deployments_command", None) is None:
            parser.parse_args(["deployments", "--help"])
        elif args.deployments_command == "list":
            _cmd_deployments_list(args)
        elif args.deployments_command == "get":
            _cmd_deployments_get(args)
        elif args.deployments_command == "create":
            _cmd_deployments_create(args)
        elif args.deployments_command == "pause":
            _cmd_deployments_pause(args)
        elif args.deployments_command == "resume":
            _cmd_deployments_resume(args)
        elif args.deployments_command == "delete":
            _cmd_deployments_delete(args)
        elif args.deployments_command == "logs":
            _cmd_deployments_logs(args)
        elif args.deployments_command == "metrics":
            _cmd_deployments_metrics(args)
    elif args.command == "hub":
        if getattr(args, "hub_command", None) is None:
            parser.parse_args(["hub", "--help"])
        elif args.hub_command == "search":
            _cmd_hub_search(args)
        elif args.hub_command == "get":
            _cmd_hub_get(args)
        elif args.hub_command == "star":
            _cmd_hub_star(args)
        elif args.hub_command == "unstar":
            _cmd_hub_unstar(args)
        elif args.hub_command == "fork":
            _cmd_hub_fork(args)
        elif args.hub_command == "featured":
            _cmd_hub_featured(args)
        elif args.hub_command == "trending":
            _cmd_hub_trending(args)
    elif args.command == "projects":
        if getattr(args, "projects_command", None) is None:
            parser.parse_args(["projects", "--help"])
        elif args.projects_command == "list":
            _cmd_projects_list(args)
        elif args.projects_command == "get":
            _cmd_projects_get(args)
        elif args.projects_command == "create":
            _cmd_projects_create(args)
        elif args.projects_command == "delete":
            _cmd_projects_delete(args)
        elif args.projects_command == "duplicate":
            _cmd_projects_duplicate(args)
    elif args.command == "codegen":
        if getattr(args, "codegen_command", None) is None:
            parser.parse_args(["codegen", "--help"])
        elif args.codegen_command == "generate":
            _cmd_codegen_generate(args)
        elif args.codegen_command == "preview":
            _cmd_codegen_preview(args)
        elif args.codegen_command == "validate":
            _cmd_codegen_validate(args)
        elif args.codegen_command == "download":
            _cmd_codegen_download(args)
