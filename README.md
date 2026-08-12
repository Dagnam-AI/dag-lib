# dagnam

[![PyPI version](https://img.shields.io/pypi/v/dagnam.svg)](https://pypi.org/project/dagnam/)
[![Python versions](https://img.shields.io/pypi/pyversions/dagnam.svg)](https://pypi.org/project/dagnam/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/Dagnam-AI/dag-lib/actions/workflows/dag-lib-ci.yml/badge.svg)](https://github.com/Dagnam-AI/dag-lib/actions/workflows/dag-lib-ci.yml)

The official Python SDK for Dagnam.AI.

`dagnam` lets Python users work with Dagnam datasets, checkpoints, training
streams, deployments, projects, code generation, and the Model Hub from scripts,
notebooks, services, and generated training code.

The API is usable today and stays backwards-compatible within a minor (`0.7.x`)
release line where practical, but the SDK is still marked alpha while the
platform API continues to mature.

## Installation

```bash
pip install dagnam
```

Python 3.12 is supported. The SDK targets this runtime so `dagnam[all]`
installs every optional integration from the published dependency set.

Optional framework extras:

```bash
pip install "dagnam[pytorch]"      # torch + torchvision
pip install "dagnam[audio]"        # torch + torchaudio
pip install "dagnam[tensorflow]"   # tensorflow
pip install "dagnam[flax]"         # jax + flax
pip install "dagnam[streaming]"    # SSE training/deployment streams
pip install "dagnam[aio]"          # async client
pip install "dagnam[all]"          # all optional integrations
```

## Authentication

The SDK resolves credentials in this order:

1. Explicit arguments such as `api_key=...` or `dagnam.configure(api_key=...)`
2. `DAGNAM_API_KEY`
3. `~/.dagnam/config.json`

```python
import dagnam

dagnam.configure(api_key="dgn_...")
```

You can also save credentials with the CLI:

```bash
dagnam login
```

By default the SDK talks to `https://api.dagnam.ai`. Override it with
`DAGNAM_API_URL`, `dagnam.configure(api_url=...)`, or per-call `api_url=...`.

## Local Generated-Code Metrics

Generated training projects install `dagnam` through their `requirements.txt`.
When running generated training locally, install the project requirements,
authenticate the SDK, and choose a persistent metrics JSONL path:

```bash
pip install -r requirements.txt
dagnam login
dagnam config set training_metrics_path ./dagnam_metrics.jsonl
```

Generated `train.py` imports `dagnam.training` and writes progress, metrics,
logs, system events, and structured errors to the metrics path. The path
resolution order is `DAGNAM_METRICS_PATH`, then
`~/.dagnam/config.json.training_metrics_path`, then `./dagnam_metrics.jsonl`.
Platform-launched Dagnam jobs set `DAGNAM_METRICS_PATH` explicitly, so the job
page can stream live progress through the backend.

Standalone local runs write metrics locally and do not upload them by
themselves. To view a laptop-local run in the hosted Dagnam frontend, attach the
run to a job explicitly:

```bash
dagnam training attach <job-id> -- python train.py
```

If training is already running and writing JSONL metrics, watch the file:

```bash
dagnam training attach <job-id> --metrics-path ./dagnam_metrics.jsonl
```

The attach command uses the credentials from `dagnam login`, sets
`DAGNAM_METRICS_PATH` for child commands, uploads metrics to the job-scoped
backend ingest endpoint, and lets the existing frontend job stream show live
progress. If no path is configured, metrics still write to
`./dagnam_metrics.jsonl` with a one-time warning, but they will not appear in
the hosted frontend until you run `dagnam training attach`.

## Quick Start

Load a dataset, inspect metadata, and create a framework loader:

```python
import dagnam

dataset = dagnam.load_dataset("550e8400-e29b-41d4-a716-446655440000")

print(dataset.info)
df = dataset.to_polars()

train_loader = dataset.to_pytorch_loader(
    split="train",
    batch_size=32,
    num_workers=4,
)
```

Call a deployed model:

```python
result = dagnam.inference(
    deployment_id="dep_abc123",
    inputs={"text": "Classify this sentence."},
)
```

Download the best checkpoint for a training job:

```python
checkpoint_path = dagnam.download_checkpoint("job_xyz789")
```

Stream training events:

```python
for event in dagnam.stream_training("job_xyz789"):
    if event.event == "metric":
        print(event.data)
```

## Datasets

`load_dataset()` handles authentication, metadata lookup, download, resumable
partial downloads, SHA-256 verification, local caching, LRU eviction, and
framework adapter construction.

```python
# User dataset by UUID
ds = dagnam.load_dataset("550e8400-e29b-41d4-a716-446655440000")

# System dataset by friendly name
mnist = dagnam.load_dataset("mnist-digits")

# Specific dataset version
v2 = dagnam.load_dataset("550e8400-e29b-41d4-a716-446655440000", version="v2")

# Presigned download URL, useful in generated code
signed = dagnam.load_dataset(
    "550e8400-e29b-41d4-a716-446655440000",
    presigned_url="https://api.dagnam.ai/api/v1/datasets/.../download?token=...",
)
```

Datasets are cached under `~/.dagnam/datasets/`. Versioned datasets use separate
cache keys such as `{dataset_id}@{version}`. Interrupted downloads resume from
the `.part` file when the server supports HTTP ranges.

### Framework Adapters

```python
df = dataset.to_polars()

loader = dataset.to_pytorch_loader(
    split="train",
    batch_size=32,
    shuffle=True,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
)

tf_dataset = dataset.to_tensorflow_dataset(
    split="train",
    batch_size=32,
)

flax_batches = dataset.to_flax_dataset(
    split="train",
    batch_size=32,
)
```

Tabular adapters accept `column_roles` to override feature/target detection:

```python
loader = dataset.to_pytorch_loader(
    split="train",
    column_roles={
        "id": "ignore",
        "age": "feature",
        "income": "feature",
        "label": "target",
    },
)
```

### Supported Dataset Formats

| Format | polars | PyTorch | TensorFlow | Flax/JAX |
| --- | :---: | :---: | :---: | :---: |
| CSV | yes | yes | yes | yes |
| TSV | yes | yes | yes | yes |
| JSON | yes | yes | yes | yes |
| JSONL | yes | yes | yes | yes |
| Image folder | no | yes | yes | yes |
| Audio folder | no | yes | yes | yes |

Image folder datasets support both `root/{split}/{class}/*` and
`root/{class}/*` layouts. Audio folder datasets support WAV, MP3, and FLAC
files. Audio TensorFlow/Flax adapters load fixed-length waveforms; the PyTorch
adapter returns mel spectrogram batches by default.

## Upload Datasets

```python
uploaded = dagnam.datasets.upload(
    "data/train.csv",
    name="customer-churn",
    dataset_type="tabular",
    format="csv",
)

op = dagnam.datasets.upload_from_url(
    "https://example.com/data.csv",
    name="remote-churn",
    dataset_type="tabular",
    format="csv",
)
dataset = op.wait(timeout=600).result()
```

`upload_from_url()` returns a `LongRunningOperation` because ingestion happens on
the platform.

## Inference, Training, and Checkpoints

```python
prediction = dagnam.inference("dep_abc123", {"input": "hello"})

batch = dagnam.inference_batch(
    "dep_abc123",
    [{"input": "hello"}, {"input": "world"}],
)

health = dagnam.deployment_health("dep_abc123")

for event in dagnam.stream_training("job_xyz789"):
    print(event.event, event.data)

path = dagnam.download_checkpoint("job_xyz789")
```

Checkpoints are cached separately under `~/.dagnam/checkpoints/` with SHA-256
verification when the backend provides a checksum.

## Deployments

```python
op = dagnam.deployments.create(
    name="sentiment-api",
    project_id="proj_123",
    checkpoint_path="/checkpoints/best.pt",
    platform="fastapi",
    deployment_type="text",
    instance_type="t3.medium",
)

deployment = op.wait(timeout=300).result()
dagnam.deployments.scale(deployment["id"], 2).wait(timeout=300)
logs = dagnam.deployments.logs(deployment["id"], level="ERROR")
```

Lifecycle actions such as `create`, `pause`, `resume`, `scale`, and `rollback`
return `LongRunningOperation` objects. Read operations such as `list`, `get`,
`health`, `metrics`, and `logs` return dictionaries from the API.

## Projects, Code Generation, and Model Hub

```python
project = dagnam.projects.create("experiment", framework="pytorch")
dagnam.projects.link_dataset(project["id"], dataset_id=uploaded["id"], role="training")

preview = dagnam.codegen.preview(project["id"], framework="pytorch")
archive = dagnam.codegen.download(project["id"], framework="pytorch", dest="out.zip")

models = dagnam.hub.search(search="resnet", framework="pytorch")
dagnam.hub.star(models["items"][0]["id"])
```

The SDK exposes project CRUD, architecture save/import, dataset linking, model
hub search and publishing, code preview/validation/download, and async codegen
jobs through `LongRunningOperation`.

## Model Registry

```python
version = dagnam.models.push(
    name="tiny-chat",
    slug="tiny-chat",
    description="A tiny fine-tuned chat model",
    files=["adapter_model.safetensors", "adapter_config.json"],
)

info = dagnam.models.resolve(version["id"])
lineage = dagnam.models.get_lineage(version["id"])
path = dagnam.models.download(version["id"], "artifact_abc123")
```

`push` creates a model entry and draft version, uploads every file (each
artifact's registry type inferred from its filename), and finalizes the
version in one call. `resolve` and `get_lineage` fetch version metadata and
lineage; `download` caches artifacts under `~/.dagnam/models/`, named from
the artifact's real filename when the server provides one, with the same
checksum verification as checkpoints.

## Async Client

Install the async extra:

```bash
pip install "dagnam[aio]"
```

```python
from dagnam.aio import AsyncDagnamClient

async with AsyncDagnamClient("https://api.dagnam.ai", "dgn_...") as client:
    datasets = await client.list_datasets()
    result = await client.predict("dep_abc123", {"input": "hello"})
```

The async client mirrors the low-level HTTP client surface. High-level resource
helpers such as `dagnam.deployments.create()` are currently synchronous.

## CLI

```bash
dagnam login

dagnam dataset list
dagnam dataset info <dataset-id>
dagnam dataset download <dataset-id>
dagnam cache list
dagnam cache clear

dagnam inference run <deployment-id> --input '{"text":"hello"}'
dagnam checkpoint list <job-id>
dagnam checkpoint download <job-id>
dagnam stream <job-id>

dagnam deployments list
dagnam hub search --search resnet
dagnam models push --name tiny-chat --slug tiny-chat --description "..." --file weights.safetensors
dagnam models list
dagnam models download <version-id> <artifact-id>
dagnam projects list
dagnam codegen preview <project-id>

dagnam agent install          # install the Agent Skill into Claude Code / Codex
dagnam agent uninstall --all
```

Run `dagnam --help` or `dagnam <command> --help` for command-specific options.

## Agent Integration (Claude Code & Codex)

The `dagnam` package ships an **Agent Skill** that teaches AI coding agents — both
**Claude Code** and **Codex** — to drive the full platform (datasets → projects →
codegen → training → deployments → inference → hub) through this CLI and SDK. It is
distributed *with* the pip package and activated per harness with one command:

```bash
dagnam agent install
```

By default this **auto-detects** the agent harnesses you have installed (Claude Code
via `~/.claude`, Codex via `~/.codex` / `~/.agents`), shows exactly what it will write,
and asks before proceeding. Flags for explicit / non-interactive (CI) installs:

| Flag | Effect |
| --- | --- |
| `--claude` / `--codex` | Target a specific harness (skip auto-detect). |
| `--all` | Install to every detected harness. |
| `--yes` | Skip the confirmation prompt. |
| `--symlink` | Symlink the skill instead of copying (falls back to copy if symlinks are unavailable). |

It is idempotent and reversible — re-running updates in place, and
`dagnam agent uninstall` removes what it wrote.

**What gets installed**

- The **skill** (`SKILL.md` + on-demand `reference/*.md` + helper `scripts/`) into the
  harness's auto-discovered skills directory (`~/.claude/skills/dagnam`,
  `~/.agents/skills/dagnam`). Versions are stamped to match the installed SDK, so the
  skill never drifts from the CLI/SDK it documents.
- **Claude Code:** a plugin under `~/.claude/plugins/dagnam` providing the
  `dagnam-runner` subagent (drives long train→watch→deploy loops in an isolated
  context) and a `PreToolUse` guard hook.
- **Codex:** skill metadata (`openai.yaml`) plus an idempotent merge of the guard hook
  into `~/.codex/hooks.json` (existing hooks are preserved).

**Dry-run / preview-by-default guardrail.** Read, build, generate, and preview actions
run freely. Anything that **spends money, is irreversible, or is public** — creating a
training job or deployment, deleting a project/job/deployment, or publishing to the hub
— is gated: the agent must show an execution plan and get your explicit confirmation
first. This is enforced behaviorally by the skill and hardened by a cross-platform
`PreToolUse` deny hook (`python -m dagnam._agent.guardhook`), which is **fail-open** so
it can never wedge the agent.

**Second door (Claude plugin marketplace).** The repository also exposes a
`.claude-plugin/marketplace.json`, so Claude Code users can `/plugin install` the
`dagnam-runner` subagent and guard hook directly from the repo.

## Reliability

The client is resilient to transient platform failures out of the box:

- **Automatic retries.** Retry-safe requests (idempotent methods, and any
  request carrying an idempotency key) retry on connection errors and
  `429`/`5xx` responses with equal-jitter exponential backoff, bounded by a
  per-client retry budget so a flapping backend can't trigger a retry storm. A
  server `Retry-After` is honored but capped.
- **Idempotency keys.** A retriable `POST` mints a `uuid4` `Idempotency-Key`
  once and reuses it across retries, so a retried create is never applied twice.
- **Cross-process cache safety.** Cache writes and LRU eviction are serialized
  with a file lock, so multiple processes sharing a cache root don't corrupt it.
- **Credential-safe logging.** Namespaced loggers
  (`dagnam.http`/`dagnam.cache`/`dagnam.lro`/`dagnam.sse`) ship a redacting
  filter that scrubs API keys and presigned-URL signatures from log records and
  error text. Turn on verbose logs with `dagnam.enable_debug_logging()`.

The cache directory is a **trust boundary**: a cache hit is loaded without
re-hashing for speed, so keep the cache root private (the default under
`~/.dagnam` is user-only). The SDK warns once if it detects a group/world-
writable cache root; for a deliberately shared cache, pass `verify=True` to
force full checksum re-verification on every load.

## Configuration

The config file lives at `~/.dagnam/config.json`.

```json
{
  "api_key": "dgn_...",
  "api_url": "https://api.dagnam.ai",
  "max_cache_size": 10737418240,
  "max_checkpoint_cache_size": 10737418240
}
```

Environment variables:

| Variable | Purpose |
| --- | --- |
| `DAGNAM_API_KEY` | API key used by client and CLI calls |
| `DAGNAM_API_URL` | API base URL override |
| `DAGNAM_DEBUG` | Any non-empty value re-raises the real traceback instead of the rendered error block (same as `--debug`) |
| `DAGNAM_CACHE_LOCK_TIMEOUT` | Dataset cache-lock acquisition timeout in seconds; an unparseable value falls back to the default |

Set by the platform inside training jobs — not intended for manual use:

| Variable | Purpose |
| --- | --- |
| `DAGNAM_INTERNAL` | Internal server mode for platform training jobs |
| `DAGNAM_META_DIR` | Sidecar metadata directory used in internal mode |
| `DAGNAM_STORAGE_PATH` | Legacy internal dataset storage fallback |
| `DAGNAM_METRICS_PATH` | Where the training run writes its metrics stream |
| `DAGNAM_TRAINING_DIR` | Working directory of the running job |
| `DAGNAM_TOTAL_EPOCHS` | Total epochs, used for progress reporting |
| `DAGNAM_PROJECT_ID` | Project the running job belongs to |

## Compatibility

| SDK version | Backend version | Notes |
| --- | --- | --- |
| `0.7.x` | `>=0.5.0` | Client resilience (retries, idempotency, cache locking) + security hardening |
| `0.6.x` | `>=0.5.0, <0.7.0` | First public PyPI release line |

The SDK follows semantic versioning. Public APIs may still expand quickly while
the package is alpha, but patch releases should avoid breaking documented
`0.7.x` behavior.

## Development

```bash
cd dag-lib
uv sync

uv run poe check
uv run poe audit
```

Build the package locally:

```bash
uv run poe build
python -m twine check dist/*
```

## Security

Do not open a public issue for suspected vulnerabilities. Follow
[SECURITY.md](SECURITY.md) for private reporting.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for local development, testing, and pull
request expectations.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
