"""Argument parsing for the ``stream`` and ``training`` commands.

Split from :mod:`dagnam.cli.training` (the handlers) so neither grows past a readable size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dagnam.cli.common import add_collection_output_args
from dagnam.cli.training import (
    cmd_stream,
    cmd_training_allowed_strategies,
    cmd_training_attach,
    cmd_training_cancel,
    cmd_training_create,
    cmd_training_dag,
    cmd_training_delete,
    cmd_training_download_code,
    cmd_training_estimate,
    cmd_training_get,
    cmd_training_list,
    cmd_training_logs,
    cmd_training_metrics,
    cmd_training_metrics_summary,
    cmd_training_restart,
    cmd_training_restore,
    cmd_training_resume,
)

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction


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

    training_restart = training_sub.add_parser(
        "restart",
        help="Restart a training job.",
        description="Restart a terminal training job as a fresh run.",
    )
    training_restart.add_argument("job_id", help="ID of the training job.")
    training_restart.set_defaults(func=cmd_training_restart)

    training_resume = training_sub.add_parser(
        "resume",
        help="Resume a paused training job.",
        description=(
            "Continue a paused training job from its last saved checkpoint, under the same id. "
            "The next stretch is charged up front."
        ),
    )
    training_resume.add_argument("job_id", help="ID of the paused training job.")
    training_resume.set_defaults(func=cmd_training_resume)

    training_restore = training_sub.add_parser(
        "restore",
        help="Restart a job from a checkpoint.",
        description="Start a new job from a checkpoint, under the same checks as restart.",
    )
    training_restore.add_argument("job_id", help="ID of the training job.")
    training_restore.add_argument("checkpoint_id", help="ID of the checkpoint to restore from.")
    training_restore.set_defaults(func=cmd_training_restore)

    training_estimate = training_sub.add_parser(
        "estimate",
        help="Estimate training resources.",
        description="Estimate compute cost for a training config without creating a job.",
    )
    training_estimate.add_argument(
        "--epochs", type=int, required=True, help="Number of training epochs (required)."
    )
    training_estimate.add_argument(
        "--batch-size", type=int, required=True, help="Training batch size (required)."
    )
    training_estimate.add_argument(
        "--learning-rate", type=float, required=True, help="Initial learning rate (required)."
    )
    training_estimate.add_argument(
        "--optimizer", required=True, help="adam, adamw, sgd, rmsprop, or adagrad (required)."
    )
    training_estimate.add_argument(
        "--loss-function", required=True, help="Loss function name (required)."
    )
    training_estimate.add_argument(
        "--dataset-id", required=True, help="Training dataset ID (required)."
    )
    training_estimate.add_argument("--val-dataset-id", help="Validation dataset ID (optional).")
    training_estimate.add_argument("--test-dataset-id", help="Test dataset ID (optional).")
    training_estimate.add_argument(
        "--train-split", type=float, default=0.8, help="Train split ratio (default: 0.8)."
    )
    training_estimate.add_argument(
        "--val-split", type=float, default=0.1, help="Validation split ratio (default: 0.1)."
    )
    training_estimate.add_argument(
        "--test-split", type=float, default=0.1, help="Test split ratio (default: 0.1)."
    )
    training_estimate.add_argument(
        "--config",
        help="Advanced TrainingConfig overrides as a JSON literal or @path/to/file.json.",
    )
    add_collection_output_args(training_estimate)
    training_estimate.set_defaults(func=cmd_training_estimate)

    training_strategies = training_sub.add_parser(
        "allowed-strategies",
        help="List available distribution strategies.",
        description="List the distribution strategies available to your credential.",
    )
    add_collection_output_args(training_strategies)
    training_strategies.set_defaults(func=cmd_training_allowed_strategies)

    training_download_code = training_sub.add_parser(
        "download-code",
        help="Download generated training code.",
        description="Download a job's generated training-code ZIP.",
    )
    training_download_code.add_argument("job_id", help="ID of the training job.")
    training_download_code.add_argument(
        "--out", help="Directory to save the ZIP into (default: current directory)."
    )
    training_download_code.set_defaults(func=cmd_training_download_code)

    training_dag = training_sub.add_parser(
        "dag",
        help="Download a job's DAG JSON.",
        description="Download a job's DAG JSON.",
    )
    training_dag.add_argument("job_id", help="ID of the training job.")
    training_dag.add_argument(
        "--out", help="Directory to save the DAG JSON into (default: current directory)."
    )
    training_dag.set_defaults(func=cmd_training_dag)

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
