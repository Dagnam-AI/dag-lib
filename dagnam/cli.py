"""Command-line interface for the dagnam client library.

Usage:
    dagnam login                     Save API key to ~/.dagnam/config.json
    dagnam dataset list              List available datasets
    dagnam dataset download <id>     Download a dataset to local cache
    dagnam dataset info <id>         Show dataset metadata
    dagnam cache list                List cached datasets with sizes
    dagnam cache clear               Delete all cached datasets
"""

from __future__ import annotations

import argparse
import getpass
import json
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _human_size(nbytes: int | float) -> str:
    """Format byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def _dir_size(path: Path) -> int:
    """Recursively compute total size of a directory in bytes."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _error(msg: str) -> None:
    """Print an error message to stderr and exit."""
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _cmd_login(args: argparse.Namespace) -> None:
    from dagnam._core import config as _cfg
    from dagnam._core.client import DagnamClient
    from dagnam._core.exceptions import DagnamError

    api_key = getpass.getpass("API key: ")
    if not api_key.strip():
        _error("API key cannot be empty.")

    api_url = getattr(args, "api_url", None) or "https://api.dagnam.ai"

    # Validate by hitting the meta endpoint for a quick test
    client = DagnamClient(api_url, api_key)
    try:
        client.list_datasets()
    except DagnamError as exc:
        _error(f"Authentication failed: {exc}")

    # Persist
    _cfg.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config: dict = {}
    if _cfg.CONFIG_FILE.exists():
        try:
            config = json.loads(_cfg.CONFIG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            config = {}

    config["api_key"] = api_key
    if api_url != "https://api.dagnam.ai":
        config["api_url"] = api_url

    _cfg.CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Credentials saved to {_cfg.CONFIG_FILE}")


def _cmd_dataset_list(args: argparse.Namespace) -> None:
    from dagnam._core.auth import get_api_key, get_api_url
    from dagnam._core.client import DagnamClient
    from dagnam._core.exceptions import DagnamError

    try:
        api_key = get_api_key()
        api_url = get_api_url()
    except DagnamError as exc:
        _error(str(exc))

    client = DagnamClient(api_url, api_key)
    try:
        datasets = client.list_datasets()
    except DagnamError as exc:
        _error(str(exc))

    if not datasets:
        print("No datasets found.")
        return

    # Print formatted table
    header = f"{'ID':<40} {'Name':<25} {'Format':<8} {'Samples':>8} {'Type':<12}"
    print(header)
    print("-" * len(header))
    for ds in datasets:
        print(
            f"{ds.get('id', 'N/A'):<40} "
            f"{ds.get('name', 'N/A'):<25} "
            f"{ds.get('format', 'N/A'):<8} "
            f"{ds.get('num_samples', 'N/A'):>8} "
            f"{ds.get('dataset_type', 'N/A'):<12}"
        )


def _cmd_dataset_download(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam.data.cache import DEFAULT_CACHE_DIR
    from dagnam._core.exceptions import DagnamError

    dataset_id: str = args.dataset_id
    try:
        dagnam.load_dataset(dataset_id)
    except DagnamError as exc:
        _error(str(exc))

    print(f"Dataset '{dataset_id}' downloaded to {DEFAULT_CACHE_DIR / dataset_id}")


def _cmd_dataset_info(args: argparse.Namespace) -> None:
    from dagnam._core.auth import get_api_key, get_api_url
    from dagnam._core.client import DagnamClient
    from dagnam._core.exceptions import DagnamError

    dataset_id: str = args.dataset_id

    try:
        api_key = get_api_key()
        api_url = get_api_url()
    except DagnamError as exc:
        _error(str(exc))

    client = DagnamClient(api_url, api_key)
    try:
        meta = client.get_dataset_meta(dataset_id)
    except DagnamError as exc:
        _error(str(exc))

    for key, value in meta.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, list):
            print(f"{key}: {', '.join(str(v) for v in value)}")
        else:
            print(f"{key}: {value}")


def _cmd_cache_list(_args: argparse.Namespace) -> None:
    from dagnam.data import cache as _cache

    if not _cache.DEFAULT_CACHE_DIR.exists():
        print("Cache is empty.")
        return

    entries: list[tuple[str, str, int]] = []
    for child in sorted(_cache.DEFAULT_CACHE_DIR.iterdir()):
        if not child.is_dir():
            continue
        dataset_id = child.name
        meta_file = child / "meta.json"
        name = "unknown"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                name = meta.get("name", "unknown")
            except (json.JSONDecodeError, OSError):
                pass
        size = _dir_size(child)
        entries.append((dataset_id, name, size))

    if not entries:
        print("Cache is empty.")
        return

    header = f"{'ID':<40} {'Name':<25} {'Size':>10}"
    print(header)
    print("-" * len(header))
    for dataset_id, name, size in entries:
        print(f"{dataset_id:<40} {name:<25} {_human_size(size):>10}")


def _load_json_arg(value: str) -> object:
    """Parse --input/--inputs as JSON literal or @path/to/file.json."""
    if value.startswith("@"):
        path = Path(value[1:])
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def _cmd_inference_run(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        payload = _load_json_arg(args.input)
    except (json.JSONDecodeError, OSError) as exc:
        _error(f"Failed to parse --input: {exc}")

    try:
        result = dagnam.inference(args.deployment_id, payload)
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2))


def _cmd_inference_batch(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        payload = _load_json_arg(args.inputs)
    except (json.JSONDecodeError, OSError) as exc:
        _error(f"Failed to parse --inputs: {exc}")

    if not isinstance(payload, list):
        _error("--inputs must be a JSON array (or @path to a JSON array file)")

    try:
        result = dagnam.inference_batch(args.deployment_id, payload)
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2))


def _cmd_inference_health(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.deployment_health(args.deployment_id)
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2))


def _cmd_checkpoint_list(args: argparse.Namespace) -> None:
    from dagnam._core.auth import get_api_key, get_api_url
    from dagnam._core.client import DagnamClient
    from dagnam._core.exceptions import DagnamError

    try:
        client = DagnamClient(get_api_url(), get_api_key())
        checkpoints = client.list_checkpoints(args.job_id)
    except DagnamError as exc:
        _error(str(exc))

    if not checkpoints:
        print("No checkpoints found.")
        return
    header = f"{'ID':<40} {'Epoch':>6} {'Step':>8} {'Best':<6} {'Final':<6} {'Size':>10}"
    print(header)
    print("-" * len(header))
    for cp in checkpoints:
        print(
            f"{cp.get('id', 'N/A'):<40} "
            f"{cp.get('epoch', 0):>6} "
            f"{cp.get('step', 0):>8} "
            f"{str(cp.get('is_best', False)):<6} "
            f"{str(cp.get('is_final', False)):<6} "
            f"{_human_size(cp.get('file_size') or 0):>10}"
        )


def _cmd_checkpoint_download(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        path = dagnam.download_checkpoint(args.job_id, args.checkpoint_id)
    except DagnamError as exc:
        _error(str(exc))
    print(str(path))


def _cmd_stream(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    from dataclasses import asdict

    try:
        for ev in dagnam.stream_training(
            args.job_id,
            include_heartbeats=args.heartbeats,
        ):
            if args.json:
                print(json.dumps(asdict(ev)))
            else:
                print(f"[{ev.event}] {ev.data}")
    except DagnamError as exc:
        _error(str(exc))
    except KeyboardInterrupt:
        sys.exit(130)


# -- deployments handlers --------------------------------------------------

def _cmd_deployments_list(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        result = dagnam.deployments.list(
            status=args.status, platform=args.platform,
            project_id=args.project_id, search=args.search,
            page=args.page, limit=args.limit,
        )
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def _cmd_deployments_get(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        result = dagnam.deployments.get(args.deployment_id)
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def _cmd_deployments_create(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        result = dagnam.deployments.create(
            name=args.name, project_id=args.project_id,
            checkpoint_path=args.checkpoint_path, platform=args.platform,
            deployment_type=args.deployment_type, instance_type=args.instance_type,
            num_instances=args.num_instances,
        ).wait().result()
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def _cmd_deployments_pause(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        dagnam.deployments.pause(args.deployment_id).wait()
    except DagnamError as exc:
        _error(str(exc))
    print(f"Deployment {args.deployment_id} paused.")


def _cmd_deployments_resume(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        dagnam.deployments.resume(args.deployment_id).wait()
    except DagnamError as exc:
        _error(str(exc))
    print(f"Deployment {args.deployment_id} resumed.")


def _cmd_deployments_delete(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        dagnam.deployments.delete(args.deployment_id)
    except DagnamError as exc:
        _error(str(exc))
    print(f"Deployment {args.deployment_id} deleted.")


def _cmd_deployments_logs(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        result = dagnam.deployments.logs(
            args.deployment_id, level=args.level,
            search=args.search, limit=args.limit,
        )
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def _cmd_deployments_metrics(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        result = dagnam.deployments.metrics(args.deployment_id, time_range=args.time_range)
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2, default=str))


# -- hub handlers ----------------------------------------------------------

def _cmd_hub_search(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        result = dagnam.hub.search(
            search=args.search, framework=args.framework,
            task_type=args.task_type, sort_by=args.sort_by,
            page=args.page, limit=args.limit,
        )
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def _cmd_hub_get(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        result = dagnam.hub.get(args.model_id)
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def _cmd_hub_star(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        result = dagnam.hub.star(args.model_id)
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def _cmd_hub_unstar(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        result = dagnam.hub.unstar(args.model_id)
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def _cmd_hub_fork(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        result = dagnam.hub.fork(args.model_id)
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def _cmd_hub_featured(_args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        result = dagnam.hub.featured()
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def _cmd_hub_trending(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        result = dagnam.hub.trending(days=args.days)
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2, default=str))


# -- projects handlers -----------------------------------------------------

def _cmd_projects_list(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        result = dagnam.projects.list(
            framework=args.framework, search=args.search,
            page=args.page, limit=args.limit,
        )
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def _cmd_projects_get(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        result = dagnam.projects.get(args.project_id)
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def _cmd_projects_create(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        result = dagnam.projects.create(
            title=args.title, framework=args.framework,
            description=args.description, visibility=args.visibility,
        )
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def _cmd_projects_delete(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        dagnam.projects.delete(args.project_id)
    except DagnamError as exc:
        _error(str(exc))
    print(f"Project {args.project_id} deleted.")


def _cmd_projects_duplicate(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        result = dagnam.projects.duplicate(args.project_id, title=args.title)
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2, default=str))


# -- codegen handlers ------------------------------------------------------

def _cmd_codegen_generate(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        result = dagnam.codegen.generate(
            args.project_id, framework=args.framework,
            version_id=args.version_id, **{"async": getattr(args, "async")},
        )
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def _cmd_codegen_preview(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        result = dagnam.codegen.preview(
            args.project_id, framework=args.framework,
            version_id=args.version_id,
        )
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def _cmd_codegen_validate(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        result = dagnam.codegen.validate(args.project_id, version_id=args.version_id)
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def _cmd_codegen_download(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    try:
        result = dagnam.codegen.download(
            args.project_id, framework=args.framework,
            version_id=args.version_id, output=args.output,
        )
    except DagnamError as exc:
        _error(str(exc))
    print(json.dumps(result, indent=2, default=str))


def _cmd_cache_clear(_args: argparse.Namespace) -> None:
    from dagnam.data import cache as _cache

    if not _cache.DEFAULT_CACHE_DIR.exists():
        print("Cache is already empty.")
        return

    total = _dir_size(_cache.DEFAULT_CACHE_DIR)
    shutil.rmtree(_cache.DEFAULT_CACHE_DIR)
    print(f"Cleared cache. Freed {_human_size(total)}.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

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
    stream_parser.add_argument(
        "--json", action="store_true", help="Emit one JSON object per line"
    )
    stream_parser.add_argument(
        "--heartbeats", action="store_true", help="Include heartbeat events"
    )

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(2)

    if args.command == "login":
        _cmd_login(args)
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
