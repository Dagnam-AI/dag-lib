# Troubleshooting reference

## Exceptions (`from dagnam._core.exceptions import ...`)
All inherit from `DagnamError`:
- `AuthError` — no API key or a 401. Fix: tell the user to run `dagnam login`.
- `APIError` — a non-2xx API response (`.status_code`, `.message`).
- `*NotFoundError` — `DatasetNotFoundError`, `ProjectNotFoundError`, `TrainingJobNotFoundError`,
  `DeploymentNotFoundError`, `CheckpointNotFoundError`, `HubModelNotFoundError`,
  `ArchitectureVersionNotFoundError`, `TaskNotFoundError` (404 on that resource).
- `QuotaExceededError` — over a plan/storage limit (check `dagnam.account.entitlements()`).
- `DeploymentValidationError` / `DeploymentStateError`, `CodegenError` / `CodegenValidationError`,
  `HubError`, `UploadError`, `CheckpointError`, `ChecksumError`, `StreamError`.
- `LROTimeoutError` / `LROFailedError` — a long-running op timed out or reached a failure state
  (`.state`, `.detail`). Catch these around `op.wait()` / `op.result()`.

Catch the base when you just want "any Dagnam error":
```python
from dagnam._core.exceptions import DagnamError
try:
    ...
except DagnamError as exc:
    ...
```

## Config & files
- Credentials/config: `~/.dagnam/config.json` (key is masked in `dagnam whoami` / `config list`).
- Cache root: `~/.dagnam` by default, or `DAGNAM_CACHE_DIR`.

## Optional extras
Some features need extras: `pip install 'dagnam[streaming]'` (SSE training streams),
`'dagnam[pytorch]'`, `'dagnam[tensorflow]'`, `'dagnam[flax]'`, `'dagnam[audio]'`, `'dagnam[aio]'`
(async client), or `'dagnam[all]'`. If `dagnam.stream_training` raises an ImportError, install
`dagnam[streaming]`.

## General
- **Always pass `--json`** when you intend to parse CLI output — the default tables are for humans.
- For long training/deploy loops, use the `dagnam-runner` subagent / `scripts/watch_training.py`
  so the SSE metric stream stays off the main context.
