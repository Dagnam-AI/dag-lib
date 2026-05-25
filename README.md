# dagnam

The official Python SDK for Dagnam.AI.

`dagnam` lets Python users work with Dagnam datasets, checkpoints, training
streams, deployments, projects, code generation, and the Model Hub from scripts,
notebooks, services, and generated training code.

This is the first public PyPI release line. The API is usable today and will
remain backwards-compatible within the `0.1.x` line where practical, but the SDK
is still marked alpha while the platform API continues to mature.

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
dagnam projects list
dagnam codegen preview <project-id>
```

Run `dagnam --help` or `dagnam <command> --help` for command-specific options.

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
| `DAGNAM_CACHE_DIR` | Shared cache root for native system dataset loaders |
| `DAGNAM_INTERNAL` | Internal server mode for platform training jobs |
| `DAGNAM_META_DIR` | Sidecar metadata directory used in internal mode |
| `DAGNAM_STORAGE_PATH` | Legacy internal dataset storage fallback |

## Compatibility

| SDK version | Backend version | Notes |
| --- | --- | --- |
| `0.1.x` | `>=0.5.0, <0.7.0` | First public PyPI release line |

The SDK follows semantic versioning. Public APIs may still expand quickly while
the package is alpha, but patch releases should avoid breaking documented
`0.1.x` behavior.

## Development

```bash
cd dag-lib
uv sync

uv run pytest tests/ -v
uv run ruff check
uv run ruff format --check
uv run --with "pyright>=1.1.380" pyright
uv run --with pip-audit pip-audit
```

Build the package locally:

```bash
uv build
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
