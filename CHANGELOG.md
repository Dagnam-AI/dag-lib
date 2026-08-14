# Changelog

All notable changes to the `dagnam` Python SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Model registry SDK: `dagnam.models`.** `push` creates a model entry and
  draft version, uploads every artifact file (its registry `artifact_type`
  inferred from the filename), and finalizes the version in one call.
  `resolve`, `get_lineage`, and `get_task_contract` fetch version metadata,
  lineage, and task contracts. `download` caches artifacts locally under
  `~/.dagnam/models/`, named from the artifact's real `filename` (or
  `logical_key`) when the server provides one, with the same checksum
  verification, atomic promotion, and LRU eviction discipline as
  `download_checkpoint` — eviction only runs once a `max_model_cache_size`
  (or shared `max_cache_size`) budget is configured, and never evicts the
  entry a call just wrote. New `ModelError` / `ModelNotFoundError`
  exceptions join the top-level `dagnam` and `dagnam.exceptions` exports.
  A `dagnam models` CLI group (`push`, `get`, `list`, `download`, `lineage`,
  `task-contract`) is also available.
- **Deployment revision history: `dagnam.deployments.revisions`.** Lists a
  deployment's revisions newest first — revision number, model version,
  serving engine, capacity mode, status, failure reason, creation time, and
  which one is currently serving. Paged via `page` / `limit` (the server caps
  a page at 200). Available on the sync and async clients as
  `get_deployment_revisions`, and as a `dagnam deployments revisions <id>`
  CLI subcommand.

## [0.9.0] - 2026-08-10

### Removed

- **The 1.18 MB validation corpus no longer ships in the wheel.** Every
  `pip install dagnam` carried `dagnam/_contracts/validation-corpus.json`, which
  has no runtime consumer — it is test data for the validators. It moves to
  `tests/_contracts/`, where a fixture belongs. The built wheel drops from
  390,680 to 311,723 bytes compressed, and `dagnam/_contracts/` is now the
  deprecation shim alone.
  `dagnam._contracts` still re-exports every public name from
  `dagnam_contracts` unchanged, so no import breaks in this release. **That
  shim is removed in 0.10.0** — import `dagnam_contracts` directly.

### Added

- **Video / 3-D data path.** New `dagnam.data.loaders.video` provides a
  `resize_frames` loader, driven by the dataset's declared metadata rather than
  by its name, so a video dataset can feed a `Conv3d`-style model. `"video"`
  joins the existing `apply_transform` dispatch and `_KIND_TO_MODALITY` tables,
  and the PyTorch-native loader applies its channels-first transpose to 5-D
  batches exactly as it already did for 2-D images (TensorFlow and Flax already
  matched, being channels-last). Clips are a plain numeric `.npz` read by the
  existing `ArrayDecoder` with `allow_pickle=False` — no new decoder and no
  relaxation of the deserialization defense.
- **Async dataset methods gain `version=` parity with the sync client.**
  `AsyncDagnamClient.get_dataset_meta`, `get_system_dataset_meta`, and
  `download_dataset` previously had no `version` parameter at all, so pinning
  a dataset version was only possible through the sync client. `version` is
  now accepted on all three (keyword-only on `download_dataset`, matching the
  sync signature) and forwarded as a `?version=` query parameter.

### Changed

- **Generated training scripts now pull the exact rows a platform split
  contains.** `/meta` serves each split's `member_row_indices`, and the
  converters resolve `split="train"` against that membership instead of
  re-deriving it from a client-side seeded shuffle. A percentage was never a
  consumable split: the eval holdout a contamination guard had cleared was not
  the one a run evaluated against. Datasets with no server-declared splits are
  unaffected and keep the deterministic ratio partition. `train`/`val`/`test`
  alias onto a platform `eval_holdout` split; a split that resolves to nothing
  now raises rather than quietly ratio-slicing rows the declared splits
  already own. Folder-backed image/audio loaders keep the ratio partition by
  design — a server row index does not name a file in a directory walk.
- **`dagnam <typo>` suggests the intended command again.** argparse renders
  its "invalid choice" message with or without quotes depending on the Python
  version, and the parser only scraped the quoted form — so on an unquoted
  build every unknown-command error silently lost its "Did you mean ...?"
  line. Candidates now come from the subparser's real `choices`, with a
  message fallback that accepts either rendering.

- **`version=` now resolves against real dataset versions, not only the legacy
  metadata map.** The backend previously resolved `?version=` solely against a
  string-keyed map in the dataset's metadata, so a numeric key like `"1"` only
  worked if someone had explicitly registered it there. With server-side
  `DatasetVersion` records, `"1"` (and any `version_number`, and a version's
  UUID) now resolves to that version's stored file — and every pre-existing
  dataset was backfilled as version 1. A call that passed `version="1"` and
  previously 404'd will now succeed and return the backfilled v1 content;
  legacy metadata-map keys keep resolving exactly as before, and are still
  tried when a numeric/UUID lookup finds nothing.

### Security

- **`keras>=3.15.0` security floor added to the `tensorflow` and `all` extras.**
  keras arrives only as TensorFlow's own dependency, and TensorFlow asks for
  nothing newer than `keras>=3.12.0`, so the resolved 3.14.1 carried six open
  advisories: arbitrary code execution via `Lambda` deserialization
  (GHSA-5gwj-m78q-7pq3) and `TorchModuleWrapper.from_config`
  (GHSA-v2w2-w228-c444), local-file disclosure via HDF5 ExternalLinks
  (GHSA-m8wh-29wm-52mv) and virtual datasets (GHSA-26c4-7vv6-867j), and path
  traversal via `DiskIOStore` layer names (GHSA-gh82-f9x8-5frx) and tar symlink
  entries (GHSA-58hv-7753-xmfq). 3.15.0 is the first release clearing all six.
  The SDK itself only calls `tf.keras.utils.image_dataset_from_directory`, so
  none of the affected paths run inside `dagnam` — but every one of them is
  reached by a user who loads a `.keras`/`.h5` artifact with the keras this
  extra installs, so the floor is declared in the published metadata rather
  than pinned only in this repo's lockfile.

## [0.8.0] - 2026-07-28

### Security

- **Breaking: `deployments.rollback()` no longer accepts a filesystem path.**
  `dagnam.deployments.rollback(deployment_id, checkpoint_path)` is now
  `dagnam.deployments.rollback(deployment_id, checkpoint_id)`, and the CLI's
  `dagnam deployments rollback --checkpoint-path` is now `--checkpoint-id`.
  The previous `checkpoint_path` parameter let a caller point a deployment at
  an arbitrary server-side filesystem path with no ownership check — a path
  traversal vector, and (via the resulting error message) a way to probe
  whether an arbitrary path existed on the server. The checkpoint is now
  resolved and re-authorized server-side by id through the same
  checkpoint -> training job -> project -> owner chain used elsewhere, and a
  checkpoint you don't own returns a uniform 404. Callers must update
  `rollback(...)` calls and the `dagnam deployments rollback` CLI invocation
  to pass a checkpoint id instead of a path.

- **Typed exceptions for server rejections that previously arrived
  undifferentiated.** `EmailNotVerifiedError`, `AccountSuspendedError`,
  `AccountLockedError`, `PayloadTooLargeError` and `InvalidURLError` are
  exported at package top level and raised from the corresponding backend
  markers. Each subclasses `APIError`, so existing `except APIError:` handlers
  keep working unchanged — this widens what callers *can* catch without
  narrowing what they already do. Every response mapper was updated, not only
  the one that surfaced the problem: a partial fix would have left the same
  rejection raising different types depending on which call path reached it.
  A caller can now, for example, distinguish "verify your email" from "your
  account is locked" and prompt correctly instead of showing a generic failure.

### Fixed

- Dataset upload pointed at endpoints that no longer exist; it now targets the
  live routes.
- **The test suite no longer segfaults when PyTorch and TensorFlow share a
  process.** `import tensorflow` followed by `import torchvision` crashes the
  interpreter (SIGSEGV); the reverse order is fine, and importing plain `torch`
  first is not enough. The suite exercises all three framework loaders in one
  process, so whichever imported first decided whether the run survived. The
  root `conftest.py` now pins the order. This affects consumers too: an
  application that mixes both frameworks should import the PyTorch side first.

### Changed

- Security floors raised on dependencies with known advisories: `pillow>=12.3.0`
  (13 advisories), `torch>=2.13.0` (GHSA-rrmf-rvhw-rf47), and a resolution
  constraint of `setuptools>=83.0.0` (PYSEC-2026-3447, reached only transitively
  via chex/tensorboard/tensorflow/torch).

### Compatibility

- Python `>=3.12`.
- Requires a Dagnam backend that resolves deployment rollbacks by checkpoint id
  and emits the `email_not_verified` / `account_suspended` / `account_locked` /
  `blocked_ip` / `invalid_url` markers this release maps to typed exceptions.
  Against an older backend the new exception classes simply never surface, but
  `deployments.rollback()` will fail — the id it now sends is not a path.


## [0.7.0] - 2026-07-13

### Added

- **Automatic retries for transient failures.** Retry-safe API requests
  (idempotent methods, plus any request carrying an idempotency key) now retry
  on connection errors and `429`/`5xx` responses using equal-jitter exponential
  backoff, bounded by a per-client **retry budget** (token bucket) so a flapping
  backend cannot trigger an unbounded retry storm. A server `Retry-After` header
  is honored but capped, so a hostile value cannot wedge the client in a long
  sleep.
- **Idempotency keys** for retry-safe writes: a retriable `POST` mints one
  `uuid4` `Idempotency-Key` and reuses it across retries, so a retried create is
  not applied twice server-side.
- **Cross-process cache locking.** Dataset/checkpoint cache writes and LRU
  eviction are serialized with a file lock (new `filelock>=3.13` base
  dependency), so multiple processes sharing a cache root no longer race or
  corrupt entries.
- Async parity: `dagnam.aio.AsyncDagnamClient` now applies the same
  retry/backoff and transport-error handling as the sync client, fully
  non-blocking on the event loop.
- `dagnam.ResponseError` — a public `APIError` subclass for malformed,
  undecodable, or wrong-shape server response bodies.
- Library logging contract: a package-level `NullHandler`, namespaced child
  loggers (`dagnam.http`/`dagnam.cache`/`dagnam.lro`/`dagnam.sse`) with a
  credential-redacting filter, and `dagnam.enable_debug_logging()` convenience.

### Changed

- `response_json_value`/`response_json_object`/`response_json_array` and
  `BaseDagnamClient._expect_object` now raise `ResponseError` (an `APIError`
  subclass) instead of a raw `TypeError`/`ValueError`/`json.JSONDecodeError`
  when a server response body is malformed, undecodable, or the wrong JSON
  shape. This affects every sync client method that decodes a response body
  (datasets, training, account, hub, projects, deployments, codegen). Code
  that narrowly caught `except TypeError`/`except ValueError` around these
  calls should catch `except dagnam.ResponseError` (or the broader
  `dagnam.APIError`/`dagnam.DagnamError`) instead. Client mixins that
  optionally fall back to a plain-text response body preserve that behavior.

### Security

- **Path traversal / arbitrary-file-write (critical).** A server-supplied
  dataset filename (dataset metadata / `Content-Disposition`) is now reduced to
  a safe bare basename before it is joined under the download directory.
  Previously a malicious or compromised server could return an absolute path or
  a `..` sequence and make a dataset download write outside the cache — a
  remote-code-execution vector (for example, overwriting a shell rc file).
- **Presigned-URL credential redaction.** The log/error redaction filter now
  scrubs presigned-URL credential parameters (`X-Amz-Signature`,
  `X-Amz-Credential`, `Signature`, `sig`, `AWSAccessKeyId`, …) in addition to
  `token`/`api_key`/`signature`, and transport-error text is scrubbed before it
  reaches a log record or exception — so a presigned dataset/checkpoint URL can
  no longer leak its signature into logs.
- **Cache trust boundary.** The SDK warns once when the cache root is
  group/world-writable (a same-size cache-poisoning risk on a shared host) and
  documents `verify=True` to force a full checksum re-verification on load for
  shared caches.
- The `LongRunningOperation` poll interval is floored at 0.1 s, so a hostile
  server flooding `429`/`503` cannot drive a `sleep(0)` busy-loop.

## [0.6.0] - 2026-07-02

### Added

- Full public **exception hierarchy** re-exported from the top-level package and
  a new `dagnam.exceptions` module, so callers can write `except dagnam.APIError`
  (or `except dagnam.DagnamError`) without reaching into the private
  `dagnam._core.exceptions` path.
- `numpy` and `Pillow` are now declared **base dependencies** (they are imported
  eagerly by the dataset layer / image loaders), so a plain `pip install dagnam`
  can load datasets instead of failing with a bare `ModuleNotFoundError`.
- `torchvision` is now part of the `pytorch` and `all` extras — the PyTorch
  image-folder loaders require it, and the README already promised it.
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
- Widened `requires-python` to `>=3.12` (no upper cap) and declared Python 3.13
  support; corrected dependency floors so extras are actually installable
  (`tensorflow>=2.16`, the first Python-3.12-capable TF; `torch>=2.4` for the
  `pytorch` extra).
- Label encoding is unified across `to_arrays()` and every framework loader via
  one canonical string-identity mapping, so an integer label column no longer
  crashes `to_pytorch_loader()` / `to_tensorflow_dataset()` / `to_flax_dataset()`
  with a bare `KeyError`.

### Fixed

- The PyTorch native-numpy validation split no longer leaks the **entire** train
  set into "val" when the validation ratio rounds down to zero.
- Cache size/info scans skip a file removed mid-scan (concurrent eviction)
  instead of aborting with `FileNotFoundError`.
- A corrupt (non-integer) `max_cache_size` config value no longer crashes a
  freshly completed download; it falls back to the default eviction budget.
- `to_arrays()` builds an object array for ragged (variable-length) features
  instead of raising `ValueError` under numpy ≥ 1.24.

### Security

- System-dataset `.npz` decoding now uses `allow_pickle=False` and refuses
  pickled object arrays — closing an arbitrary-code-execution vector where a
  crafted dataset could run code inside `numpy.load`.
- Generated-code ZIP extraction is hardened against **zip-slip**: archive
  members with path-traversal, absolute, or symlink paths are rejected instead
  of being written outside the destination directory.
- Checkpoint downloads that arrive without a server checksum (e.g. an S3
  presigned redirect) are now flagged with a loud warning instead of being
  silently accepted unverified.

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
