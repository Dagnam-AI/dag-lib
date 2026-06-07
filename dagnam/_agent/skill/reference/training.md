# Training reference

## CLI (`dagnam training ...`, add `--json` where supported)
- `dagnam training create <project_id> --epochs N --batch-size N --learning-rate F --optimizer adam --loss-function cross_entropy --dataset-id <id>` — start a job. Required: `--epochs --batch-size --learning-rate --optimizer --loss-function --dataset-id`. Optional: `--framework pytorch` (default), `--val-dataset-id`, `--test-dataset-id`, `--train-split 0.8`, `--val-split 0.1`, `--test-split 0.1`, `--max-duration-seconds`, `--confirm-resource-warning`, `--config @overrides.json`. **[guardrail: costly]**
- `dagnam training list [--status s] [--project-id id] [--page 1] [--limit 20]` (+ `--json/--verbose/--output`) — list jobs.
- `dagnam training get <job_id>` (+ `--json/--verbose/--output`) — job detail.
- `dagnam stream <job_id> [--heartbeats] [--json]` — stream live SSE events (this is a top-level command, not `training stream`).
- `dagnam training cancel <job_id>` — cancel a non-terminal job.
- `dagnam training delete <job_ids...>` — delete 1–100 jobs. **[guardrail: irreversible]**
- `dagnam training logs <job_id> [--log-level debug|info|warning|error|critical] [--source stdout] [--page] [--limit 100] [--output]`.
- `dagnam training metrics <job_id> [--metric-type] [--epoch-start] [--epoch-end] [--epoch-summary] [--page] [--limit 100] [--output]`.
- `dagnam training metrics-summary <job_id> [--output]`.
- `dagnam training attach <job_id> [--metrics-path p] [--replay] -- python train.py` — attach a local run's JSONL metrics to a platform job.

## SDK (`import dagnam`)
- `dagnam.create_training_job(project_id, epochs, batch_size, learning_rate, optimizer, loss_function, training_dataset_id, framework="pytorch", validation_dataset_id=None, test_dataset_id=None, train_split=0.8, val_split=0.1, test_split=0.1, config_overrides=None, max_duration_seconds=None, confirm_resource_warning=False) -> dict` — **[guardrail]** leave `confirm_resource_warning=False` on the first call to surface the resource estimate; only re-call with `True` after the user confirms.
- `dagnam.stream_training(job_id, last_event_id=None, include_heartbeats=False) -> Iterator[TrainingEvent]` — each event has `.event` (name) / `.data` (dict or str) / `.id`; terminal names: `complete`, `failed`, `cancelled`, `stream_end`.
- `dagnam.get_training_job(job_id)`, `dagnam.list_training_jobs(page=1, limit=20, status=None, project_id=None)`, `dagnam.cancel_training_job(job_id)`, `dagnam.delete_training_jobs(job_ids)`, `dagnam.training_logs(job_id, ...)`, `dagnam.training_metrics(job_id, ...)`, `dagnam.training_metrics_summary(job_id)`.
- `dagnam.download_checkpoint(job_id, checkpoint_id=None, cache_dir=None, prefer_best=True) -> Path`.

## Instrumenting a training script (generated / authored code)
A run reports back to the platform with the top-level instrumentation API:
`dagnam.init(project_id, framework, name=..., mode=...)`, `dagnam.report_metric(epoch, step, metrics)`,
`dagnam.report_progress(epoch, total_epochs, step, total_steps)`, `dagnam.report_log(level, message)`,
`dagnam.report_system(...)`, `dagnam.report_error(category, technical_summary, ...)`,
`dagnam.write_training_state(epoch, step, latest_checkpoint_path, ...)`.
For local runs that should appear in the platform UI: `dagnam training attach <job_id> -- python train.py`.

## Recipe — train and watch (with guardrail)
```python
import dagnam
# 1) preview impact (confirm_resource_warning=False) + confirm with the user, THEN:
job = dagnam.create_training_job("proj_1", epochs=5, batch_size=32, learning_rate=1e-3,
                                 optimizer="adam", loss_function="cross_entropy",
                                 training_dataset_id="ds_1")
# 2) long run -> delegate streaming to the dagnam-runner subagent / scripts/watch_training.py
ckpt = dagnam.download_checkpoint(job["id"])  # after `complete`
```
