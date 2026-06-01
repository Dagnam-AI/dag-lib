# Changelog

All notable changes to the `dagnam` Python SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Release-process documentation for maintainers.
- CLI version and account inspection commands: `dagnam --version`, `-v`,
  `version`, `whoami`, `logout`, and read-only `config list` / `config get`.
- Expanded CLI help text with command descriptions, examples, and argument
  descriptions.
- Training-job lifecycle SDK + CLI: `dagnam training create/list/get/cancel/
  delete/logs/metrics/metrics-summary`, including after-the-fact retrieval of
  historical logs and metrics.
- `dagnam usage` command and `dagnam.account.*` helpers for plan, entitlement,
  storage-quota, and per-API-key usage inspection.
- Width-aware table rendering, pagination footers, and consistent
  `--json` / `--output` flags across table-printing commands.
- `--no-progress` flag for `dagnam codegen download` (progress is also hidden
  automatically when stderr is not a TTY), matching `dagnam dataset download`.

### Changed

- Set the supported Python runtime to Python 3.12 so `dagnam[all]` includes
  every optional integration.
- CLI/SDK error messages now unwrap FastAPI `{"detail": ...}` bodies (including
  422 validation arrays) into concise human-readable text instead of raw JSON.
- **Breaking (SDK):** `dagnam.download_checkpoint(job_id)` with no
  `checkpoint_id` now selects the **latest** checkpoint by epoch/step (was
  best-then-latest). Pass `prefer_best=True` to restore the previous behavior.
  The `dagnam checkpoint download <job> best` CLI keyword maps to
  `prefer_best=True`.

## [0.1.0] - 2026-05-10

First public PyPI release of the official Dagnam.AI Python SDK.

### Added

- Dataset loading with API-key authentication, metadata lookup, local caching,
  SHA-256 verification, LRU eviction, resumable downloads, presigned download
  URLs, and dataset version selection.
- `DagnamDataset` adapters for polars, PyTorch, TensorFlow, and Flax/JAX.
- Tabular CSV, TSV, JSON, and JSONL loaders with deterministic train/validation
  and test splits.
- Image-folder and audio-folder dataset loaders with archive extraction,
  deterministic fallback splits, and framework transform hooks.
- System dataset resolution by friendly name.
- Dataset upload helpers for local files and server-side URL ingestion.
- Inference helpers for single and batch prediction plus deployment health.
- Checkpoint download with checksum verification and a dedicated checkpoint
  cache.
- Synchronous SSE training stream iterator with reconnect support.
- Deployment, Model Hub, project, and code generation resource modules.
- `LongRunningOperation` for polling asynchronous platform operations.
- `dagnam.aio.AsyncDagnamClient` for low-level async API access.
- CLI commands for login, datasets, cache, inference, checkpoints, streams,
  deployments, hub, projects, and code generation.
- Typed package marker via `dagnam/py.typed`.

### Security

- `dagnam login` writes `~/.dagnam/config.json` with owner-only permissions on
  POSIX systems and creates `~/.dagnam` with owner-only directory permissions.
- Archive extraction rejects unsafe paths, symlinks, special files, and oversized
  archives before unpacking media datasets.
- Downloaded dataset and checkpoint filenames from `Content-Disposition` are
  sanitized before writing to disk.

### Changed

- Licensed the SDK under Apache License 2.0 with a root `NOTICE` file included
  in source and wheel distributions.

### Compatibility

- Python `>=3.12,<3.13`.
- Dagnam backend `>=0.5.0, <0.7.0`.
