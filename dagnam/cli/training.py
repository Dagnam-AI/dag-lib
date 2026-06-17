"""Training command handlers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from typing import TYPE_CHECKING

from dagnam.cli.common import (
    add_collection_output_args,
    error,
    load_json_arg,
    print_json,
    print_next_step,
    write_json_file,
)
from dagnam.cli.presentation import Column, emit_result, pagination_footer, render_table

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction


def cmd_stream(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

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
        error(str(exc))
    except KeyboardInterrupt:
        sys.exit(130)


def cmd_training_attach(args: argparse.Namespace) -> None:
    """Attach a local metrics JSONL file or child process to a Dagnam job."""
    from dagnam._core.exceptions import AuthError, DagnamError
    from dagnam.training_attach import run_training_attach

    try:
        code = run_training_attach(
            job_id=args.job_id,
            metrics_path=args.metrics_path,
            command=args.command,
            replay=args.replay,
        )
    except AuthError:
        error("Not logged in. Run 'dagnam login'.")
    except FileNotFoundError as exc:
        error(str(exc))
    except DagnamError as exc:
        error(str(exc))
    sys.exit(code)


def _job_overrides(args: argparse.Namespace) -> dict | None:
    if not args.config:
        return None
    try:
        overrides = load_json_arg(args.config)
    except (json.JSONDecodeError, OSError) as exc:
        error(f"Could not read --config JSON: {exc}")
    if not isinstance(overrides, dict):
        error("--config must be a JSON object of TrainingConfig overrides.")
    return overrides


def cmd_training_create(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.create_training_job(
            args.project_id,
            framework=args.framework,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            loss_function=args.loss_function,
            training_dataset_id=args.dataset_id,
            validation_dataset_id=args.val_dataset_id,
            test_dataset_id=args.test_dataset_id,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            config_overrides=_job_overrides(args),
            max_duration_seconds=args.max_duration_seconds,
            confirm_resource_warning=args.confirm_resource_warning,
        )
    except DagnamError as exc:
        error(str(exc))
    print_json(result)
    job_id = result.get("id")
    print_next_step(f"dagnam stream {job_id or '<job-id>'}")


def cmd_training_get(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.get_training_job(args.job_id)
    except DagnamError as exc:
        error(str(exc))
    if args.output:
        write_json_file(args.output, result)
    if args.json or args.verbose:
        print_json(result)
        return
    print(f"Training job {result.get('id') or args.job_id}")
    print(f"Status: {result.get('status') or '-'}")
    print(f"Framework: {result.get('framework') or '-'}")
    print(f"Epoch: {result.get('current_epoch', 0)}/{result.get('total_epochs', 0)}")
    print(f"Progress: {result.get('progress_percentage', 0)}%")


def _render_jobs(result: object) -> str:
    items = result.get("items") if isinstance(result, dict) else None
    items = items if isinstance(items, list) else []
    if not items:
        return "No training jobs found."
    rows: list[dict[str, object]] = []
    for item in items:
        job = item if isinstance(item, dict) else {}
        rows.append(
            {
                **job,
                "epoch": f"{job.get('current_epoch', 0)}/{job.get('total_epochs', 0)}",
                "progress": f"{job.get('progress_percentage', 0)}%",
                "created": str(job.get("created_at") or "-").split("T", maxsplit=1)[0],
            }
        )
    table = render_table(
        (
            Column("ID", "id", 36),
            Column("Status", "status", 11),
            Column("Framework", "framework", 11),
            Column("Epoch", "epoch", 11, "right"),
            Column("Progress", "progress", 9, "right"),
            Column("Created", "created", 10),
        ),
        rows,
    )
    return f"{table}\n{pagination_footer(result)}"


def cmd_training_list(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.list_training_jobs(
            page=args.page,
            limit=args.limit,
            status=args.status,
            project_id=args.project_id,
        )
    except DagnamError as exc:
        error(str(exc))
    emit_result(
        result,
        output=args.output,
        json_stdout=args.json or args.verbose,
        render_human=_render_jobs,
    )


def cmd_training_cancel(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.cancel_training_job(args.job_id)
    except DagnamError as exc:
        error(str(exc))
    message = result.get("message")
    print(message or f"Training job {args.job_id} cancelled.")


def cmd_training_delete(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.delete_training_jobs(args.job_ids)
    except DagnamError as exc:
        error(str(exc))
    print_json(result)


def cmd_training_logs(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.training_logs(
            args.job_id,
            log_level=args.log_level,
            source=args.source,
            page=args.page,
            limit=args.limit,
        )
    except DagnamError as exc:
        error(str(exc))
    if args.output:
        write_json_file(args.output, result)
    print_json(result)


def cmd_training_metrics(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.training_metrics(
            args.job_id,
            metric_type=args.metric_type,
            epoch_start=args.epoch_start,
            epoch_end=args.epoch_end,
            epoch_summary=args.epoch_summary,
            page=args.page,
            limit=args.limit,
        )
    except DagnamError as exc:
        error(str(exc))
    if args.output:
        write_json_file(args.output, result)
    print_json(result)


def cmd_training_metrics_summary(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        result = dagnam.training_metrics_summary(args.job_id)
    except DagnamError as exc:
        error(str(exc))
    if args.output:
        write_json_file(args.output, result)
    print_json(result)


def register_training(subparsers: SubParsersAction) -> None:
    """Register the ``stream`` and ``training`` commands on the top-level subparsers."""
    stream = subparsers.add_parser(
        "stream",
        help="Stream live training events.",
        description="Stream training events for a job over SSE.",
    )
    stream.add_argument("job_id", help="ID of the training job.")
    stream.add_argument("--heartbeats", action="store_true", help="Include heartbeat events.")
    stream.add_argument("--json", action="store_true", help="Emit raw JSON events.")
    stream.set_defaults(func=cmd_stream)

    training_cmd = subparsers.add_parser(
        "training",
        help="Create, inspect, and manage training jobs.",
        description="Create, list, inspect, cancel, delete, or attach metrics to training jobs.",
    )
    training_sub = training_cmd.add_subparsers(dest="training_command", required=True)

    training_create = training_sub.add_parser(
        "create",
        help="Create a training job.",
        description="Create a platform training job for a project.",
    )
    training_create.add_argument("project_id", help="ID of the project to train.")
    training_create.add_argument(
        "--framework", default="pytorch", help="pytorch, tensorflow, or flax (default: pytorch)."
    )
    training_create.add_argument(
        "--epochs", type=int, required=True, help="Number of training epochs (required)."
    )
    training_create.add_argument(
        "--batch-size", type=int, required=True, help="Training batch size (required)."
    )
    training_create.add_argument(
        "--learning-rate", type=float, required=True, help="Initial learning rate (required)."
    )
    training_create.add_argument(
        "--optimizer",
        required=True,
        help="adam, adamw, sgd, rmsprop, or adagrad (required).",
    )
    training_create.add_argument(
        "--loss-function", required=True, help="Loss function name (required)."
    )
    training_create.add_argument(
        "--dataset-id", required=True, help="Training dataset ID (required)."
    )
    training_create.add_argument("--val-dataset-id", help="Validation dataset ID (optional).")
    training_create.add_argument("--test-dataset-id", help="Test dataset ID (optional).")
    training_create.add_argument(
        "--train-split", type=float, default=0.8, help="Train split ratio (default: 0.8)."
    )
    training_create.add_argument(
        "--val-split", type=float, default=0.1, help="Validation split ratio (default: 0.1)."
    )
    training_create.add_argument(
        "--test-split", type=float, default=0.1, help="Test split ratio (default: 0.1)."
    )
    training_create.add_argument(
        "--max-duration-seconds", type=int, help="Hard cap on run time, in seconds."
    )
    training_create.add_argument(
        "--confirm-resource-warning",
        action="store_true",
        help="Acknowledge a soft resource warning and proceed.",
    )
    training_create.add_argument(
        "--config",
        help="Advanced TrainingConfig overrides as a JSON literal or @path/to/file.json.",
    )
    training_create.set_defaults(func=cmd_training_create)

    training_list = training_sub.add_parser(
        "list", help="List training jobs.", description="List your training jobs."
    )
    training_list.add_argument("--status", help="Filter by status (comma-separated).")
    training_list.add_argument("--project-id", help="Filter by project ID.")
    training_list.add_argument("--page", type=int, default=1, help="Page number (default: 1).")
    training_list.add_argument(
        "--limit", type=int, default=20, help="Results per page (default: 20)."
    )
    add_collection_output_args(training_list)
    training_list.set_defaults(func=cmd_training_list)

    training_get = training_sub.add_parser(
        "get", help="Show a training job.", description="Show details for one training job."
    )
    training_get.add_argument("job_id", help="ID of the training job.")
    add_collection_output_args(training_get)
    training_get.set_defaults(func=cmd_training_get)

    training_cancel = training_sub.add_parser(
        "cancel", help="Cancel a training job.", description="Cancel a non-terminal training job."
    )
    training_cancel.add_argument("job_id", help="ID of the training job.")
    training_cancel.set_defaults(func=cmd_training_cancel)

    training_delete = training_sub.add_parser(
        "delete",
        help="Delete training jobs.",
        description="Delete one or more training jobs (1-100).",
    )
    training_delete.add_argument(
        "job_ids", nargs="+", help="One or more training job IDs to delete."
    )
    training_delete.set_defaults(func=cmd_training_delete)

    training_logs = training_sub.add_parser(
        "logs",
        help="Show historical training logs.",
        description="Fetch paginated logs for one training job.",
    )
    training_logs.add_argument("job_id", help="ID of the training job.")
    training_logs.add_argument(
        "--log-level",
        choices=("debug", "info", "warning", "error", "critical"),
        help="Filter by log level.",
    )
    training_logs.add_argument("--source", help="Filter by source, such as stdout or system.")
    training_logs.add_argument("--page", type=int, default=1, help="Page number (default: 1).")
    training_logs.add_argument(
        "--limit", type=int, default=100, help="Results per page (default: 100)."
    )
    training_logs.add_argument("--output", help="Write the full JSON response to this path.")
    training_logs.set_defaults(func=cmd_training_logs)

    training_metrics = training_sub.add_parser(
        "metrics",
        help="Show historical training metrics.",
        description="Fetch paginated metrics for one training job.",
    )
    training_metrics.add_argument("job_id", help="ID of the training job.")
    training_metrics.add_argument("--metric-type", help="Filter by metric type.")
    training_metrics.add_argument("--epoch-start", type=int, help="Start epoch, inclusive.")
    training_metrics.add_argument("--epoch-end", type=int, help="End epoch, inclusive.")
    training_metrics.add_argument(
        "--epoch-summary",
        action="store_true",
        help="Return the final metric value for each epoch and metric type.",
    )
    training_metrics.add_argument("--page", type=int, default=1, help="Page number (default: 1).")
    training_metrics.add_argument(
        "--limit", type=int, default=100, help="Results per page (default: 100)."
    )
    training_metrics.add_argument("--output", help="Write the full JSON response to this path.")
    training_metrics.set_defaults(func=cmd_training_metrics)

    training_metrics_summary = training_sub.add_parser(
        "metrics-summary",
        help="Show aggregate training metrics.",
        description="Fetch aggregate metrics for one training job.",
    )
    training_metrics_summary.add_argument("job_id", help="ID of the training job.")
    training_metrics_summary.add_argument(
        "--output", help="Write the full JSON response to this path."
    )
    training_metrics_summary.set_defaults(func=cmd_training_metrics_summary)

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
    attach.add_argument(
        "--replay",
        action="store_true",
        help=(
            "Upload existing file contents first. If no command is provided, "
            "replay existing events and exit."
        ),
    )
    attach.add_argument("command", nargs="*", help="Command to run after '--'.")
    attach.set_defaults(func=cmd_training_attach)
