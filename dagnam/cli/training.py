"""Training and checkpoint command handlers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from dagnam.cli.common import error, human_size, load_json_arg, print_json, write_json_file
from dagnam.cli.presentation import Column, emit_result, pagination_footer, render_table


def _numeric_json_value(value: object, default: int = 0) -> int | float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return value
    return default


def cmd_checkpoint_list(args: argparse.Namespace) -> None:
    from dagnam._core.auth import get_api_key, get_api_url
    from dagnam._core.client import DagnamClient
    from dagnam._core.exceptions import DagnamError

    try:
        client = DagnamClient(get_api_url(), get_api_key())
        checkpoints = client.list_checkpoints(args.job_id)
    except DagnamError as exc:
        error(str(exc))

    emit_result(
        checkpoints,
        output=args.output,
        json_stdout=args.json or args.verbose,
        render_human=_render_checkpoints,
    )


def _render_checkpoints(result: object) -> str:
    checkpoints = result if isinstance(result, list) else []
    if not checkpoints:
        return "No checkpoints found."
    rows = [
        {
            **checkpoint,
            "size": human_size(_numeric_json_value(checkpoint.get("file_size"))),
        }
        for checkpoint in checkpoints
        if isinstance(checkpoint, dict)
    ]
    return render_table(
        (
            Column("ID", "id", 40),
            Column("Epoch", "epoch", 6, "right"),
            Column("Step", "step", 8, "right"),
            Column("Best", "is_best", 6),
            Column("Final", "is_final", 6),
            Column("Size", "size", 10, "right"),
        ),
        rows,
    )


def cmd_checkpoint_download(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError

    checkpoint_id = None if args.checkpoint_id in (None, "latest", "best") else args.checkpoint_id
    prefer_best = args.checkpoint_id == "best"
    try:
        if args.output_dir:
            kwargs = {"cache_dir": Path(args.output_dir)}
            if prefer_best:
                kwargs["prefer_best"] = True
            path = dagnam.download_checkpoint(args.job_id, checkpoint_id, **kwargs)
        elif prefer_best:
            path = dagnam.download_checkpoint(args.job_id, checkpoint_id, prefer_best=True)
        else:
            path = dagnam.download_checkpoint(args.job_id, checkpoint_id)
    except DagnamError as exc:
        error(str(exc))
    print(str(path))


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
    message = result.get("message") if isinstance(result, dict) else None
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
