"""Training command handlers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys

from dagnam.cli.common import (
    error,
    format_local,
    load_json_arg,
    print_json,
    print_next_step,
    write_json_file,
)
from dagnam.cli.presentation import (
    Column,
    emit_result,
    pagination_footer,
    render_table,
    sanitize_terminal_text,
)


def cmd_stream(args: argparse.Namespace) -> None:
    from dagnam._core.sse import is_pause
    from dagnam.resources.training import follow_training

    paused = False
    try:
        for ev in follow_training(args.job_id, include_heartbeats=args.heartbeats):
            paused = paused or is_pause(ev)
            if args.json:
                print(json.dumps(asdict(ev)))
            else:
                print(f"[{ev.event}] {ev.data}")
    except KeyboardInterrupt:
        sys.exit(130)
    if paused:
        print_next_step(f"dagnam training resume {args.job_id}")


def cmd_training_attach(args: argparse.Namespace) -> None:
    """Attach a local metrics JSONL file or child process to a Dagnam job."""
    from dagnam.training_attach import run_training_attach

    try:
        code = run_training_attach(
            job_id=args.job_id,
            metrics_path=args.metrics_path,
            command=args.command,
            replay=args.replay,
        )
    except FileNotFoundError as exc:
        error(str(exc), hint="Check the path and retry.")
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
    print_json(result)
    job_id = result.get("id")
    print_next_step(f"dagnam stream {job_id or '<job-id>'}")


def cmd_training_get(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.get_training_job(args.job_id)
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
    if result.get("model_version_id"):
        print(f"Model version: {result['model_version_id']}")
    if result.get("status") == "paused":
        # A pause is the platform's way of saying the account could not fund the next stretch;
        # its sentence is the job's error_message. The progress is saved, so say how to go on.
        if result.get("completed_at"):
            print(f"Paused since: {format_local(result['completed_at'])}")
        if result.get("error_message"):
            print(f"Paused: {sanitize_terminal_text(result['error_message'])}")
        print_next_step(f"dagnam training resume {result.get('id') or args.job_id}")


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
                "created": format_local(job.get("created_at")),
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

    result = dagnam.list_training_jobs(
        page=args.page,
        limit=args.limit,
        status=args.status,
        project_id=args.project_id,
    )
    emit_result(
        result,
        output=args.output,
        json_stdout=args.json or args.verbose,
        render_human=_render_jobs,
    )


def cmd_training_cancel(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.cancel_training_job(args.job_id)
    message = result.get("message")
    print(message or f"Training job {args.job_id} cancelled.")


def cmd_training_delete(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.delete_training_jobs(args.job_ids)
    print_json(result)


def cmd_training_logs(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.training_logs(
        args.job_id,
        log_level=args.log_level,
        source=args.source,
        page=args.page,
        limit=args.limit,
    )
    if args.output:
        write_json_file(args.output, result)
    print_json(result)


def cmd_training_metrics(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.training_metrics(
        args.job_id,
        metric_type=args.metric_type,
        epoch_start=args.epoch_start,
        epoch_end=args.epoch_end,
        epoch_summary=args.epoch_summary,
        page=args.page,
        limit=args.limit,
    )
    if args.output:
        write_json_file(args.output, result)
    print_json(result)


def cmd_training_metrics_summary(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.training_metrics_summary(args.job_id)
    if args.output:
        write_json_file(args.output, result)
    print_json(result)


def cmd_training_restart(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.restart(args.job_id)
    print_json(result)
    print_next_step(f"dagnam stream {result.get('id') or '<job-id>'}")


def cmd_training_resume(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.resume(args.job_id)
    print_json(result)
    print_next_step(f"dagnam stream {result.get('id') or '<job-id>'}")


def cmd_training_restore(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.restore_checkpoint(args.job_id, args.checkpoint_id)
    print_json(result)
    print_next_step(f"dagnam stream {result.get('id') or '<job-id>'}")


def _render_estimate(result: object) -> str:
    data = result if isinstance(result, dict) else {}
    lines = [
        f"Estimated memory:   {data.get('estimated_memory_mb', '-')} MB",
        f"Estimated time:     {data.get('estimated_training_time_seconds', '-')} s",
        f"Estimated disk:     {data.get('estimated_disk_space_mb', '-')} MB",
        f"Estimated cost:     {data.get('estimated_cost_usd', '-')}",
    ]
    warnings = data.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in warnings)
    recommendations = data.get("recommendations")
    if isinstance(recommendations, list) and recommendations:
        lines.append("Recommendations:")
        lines.extend(f"  - {r}" for r in recommendations)
    return "\n".join(lines)


def cmd_training_estimate(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.estimate_resources(
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
    )
    emit_result(
        result,
        output=args.output,
        json_stdout=args.json or args.verbose,
        render_human=_render_estimate,
    )


def _render_strategies(result: object) -> str:
    data = dict(result) if isinstance(result, dict) else {}
    # The endpoint ships a registry-driven `required_tiers` {label: tier} map
    # alongside the flat availability entries (free strategies omitted). Pop it
    # out so it never renders as a phantom strategy row, and use it to annotate
    # each locked strategy with the minimum tier that unlocks it.
    raw_tiers = data.pop("required_tiers", {})
    required_tiers = raw_tiers if isinstance(raw_tiers, dict) else {}
    rows: list[dict[str, object]] = []
    for label, available in sorted(data.items()):
        tier = required_tiers.get(label)
        rows.append(
            {
                "strategy": label,
                "available": "Yes" if available else "No",
                # Tier only matters for a strategy the credential can't use.
                "tier": str(tier).title() if (not available and tier) else "-",
            }
        )
    if not rows:
        return "No strategies available."
    return render_table(
        (
            Column("Strategy", "strategy", 24),
            Column("Available", "available", 9),
            Column("Required Tier", "tier", 13),
        ),
        rows,
    )


def cmd_training_allowed_strategies(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.allowed_strategies()
    emit_result(
        result,
        output=args.output,
        json_stdout=args.json or args.verbose,
        render_human=_render_strategies,
    )


def cmd_training_download_code(args: argparse.Namespace) -> None:
    import dagnam

    path = dagnam.download_code(args.job_id, out=args.out)
    print(f"Saved training code to {path}")


def cmd_training_dag(args: argparse.Namespace) -> None:
    import dagnam

    path = dagnam.download_dag(args.job_id, out=args.out)
    print(f"Saved DAG to {path}")
