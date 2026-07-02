# Changelog

All notable changes to the `dagnam` Python SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Cross-platform **Agent Skill** for AI coding agents (Claude Code and Codex),
  shipped as package data and activated with `dagnam agent install`
  (auto-detect & prompt; `--claude` / `--codex` / `--all` / `--yes` / `--symlink`
  for explicit and CI installs; `dagnam agent uninstall` to remove). It installs a
  single write-once skill (`SKILL.md` + on-demand domain `reference/*.md` + helper
  `scripts/`) into each harness's auto-discovered skills directory, plus per-harness
  adapters: a Claude plugin with the `dagnam-runner` subagent and a `PreToolUse`
  guard hook, and a Codex `openai.yaml` + idempotent guard-hook merge into
  `~/.codex/hooks.json`. Skill/adapter versions are stamped to the installed SDK.
- **Dry-run / preview-by-default guardrail** for costly, irreversible, or public
  actions (training-job and deployment creation, deletes, hub publish): a behavioral
  gate in the skill plus a fail-open cross-platform `PreToolUse` deny hook
  (`python -m dagnam._agent.guardhook`).
- Claude plugin marketplace listing (`.claude-plugin/marketplace.json`) as an
  alternate `/plugin install` door.
- `dagnam codegen download` now accepts a **directory** as the destination and
  auto-extracts the generated files into it (the SDK `download(dest=<dir>)` returns
  the directory `Path`); previously passing a directory raised a `PermissionError`.
- Uniform CLI output convention: `codegen validate`/`preview`/`generate` and
  `projects architecture` now accept `--json` (JSON to stdout) and `--output PATH`
  (JSON to file), with human-readable summaries by default — matching the other
  command groups.

### Changed

- **BREAKING (SDK/SSE):** `open_training_stream`, `open_deployment_stream`, and
  async `stream_training_events` now mint short-lived, resource-scoped
  stream-access tokens per (re)connect and send streams `?token=...`; the SDK no
  longer sends long-lived `?api_key=` on SSE URLs. Direct consumers of the SSE
  endpoints must switch to `POST .../stream-access-token` before opening a
  stream, and the backend + SDK must deploy together.
- **BREAKING (CLI):** `dagnam codegen download --output PATH` is renamed to
  `--dest PATH` (no alias). `--output` now uniformly means "write the JSON result
  to a file"; `--dest` is the downloaded-artifact destination (a file path streams
  the ZIP; a directory auto-extracts). Migrate `codegen download ... --output X`
  to `... --dest X`.

- Internal: the quality gates are now clean at zero — `ruff` reports no errors
  and `pyright` (strict, minus the unavoidable untyped-ML-library completeness
  diagnostics) reports no errors. Remaining suppressions are centralized and
  documented in `pyproject.toml`. Async checkpoint/codegen downloads now write
  to disk via `asyncio.to_thread` so they no longer block the event loop, and
  `dagnam.data.loaders` submodules are imported lazily via `__getattr__`
  (PEP 562). No public API or runtime behavior changed.

## [0.5.0] - 2026-06-03

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

### Changed

- Internal: the `load_dataset` implementation module moved from
  `dagnam._core.load` to `dagnam.data.load` so it sits in the `data` layer it
  composes (cache, dataset adapters, system loaders), enforcing the
  `cli -> resources -> data -> _core` import contract. The public entry point
  `dagnam.load_dataset` is unchanged; only the internal module path moved.

### Removed

- Internal compatibility-shim modules that only forwarded old import paths:
  the `dagnam.services` package, `dagnam._core._common` / `_resolver` / `_sse`,
  the `dagnam.data.loaders.*_loader` forwarders (`audio_loader`, `csv_loader`,
  `flax_loader`, `image_folder_loader`, `json_loader`, `system_loader`,
  `tf_loader`) and `data.loaders.media_utils`, and the redundant
  `dagnam/resources/datasets_upload.py` module. These were internal forwarders
  with no external consumers; the canonical modules (`dagnam.resources.*`,
  `dagnam._core.{client.common,resolver,sse}`, `dagnam.data.loaders.{audio,csv,
  flax,image_folder,json_array,system,tf,media}`) are unchanged, and the
  documented public API (`dagnam.*` exports, the `dagnam.resources.datasets_upload`
  alias) keeps resolving.

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
