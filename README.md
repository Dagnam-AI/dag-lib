# dagnam

The official Python SDK for [Dagnam.AI](https://dagnam.ai). Load datasets, create framework-specific data loaders, and manage your local cache — all from Python or the command line.

## Install

```bash
pip install dagnam

# With framework extras
pip install dagnam[pytorch]
pip install dagnam[tensorflow]
pip install dagnam[flax]
pip install dagnam[all]
```

Requires Python 3.9+.

## Quick Start

```python
import dagnam

# Authenticate (or set DAGNAM_API_KEY env var, or run `dagnam login`)
dagnam.configure(api_key="dgn_abc123...")

# Load a dataset by ID (auto-downloads and caches)
dataset = dagnam.load_dataset("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

# Or load a system dataset by name
dataset = dagnam.load_dataset("mnist-digits")

# Inspect metadata
print(dataset.info)
# {'id': '...', 'name': 'Iris', 'format': 'csv', 'type': 'tabular',
#  'samples': 150, 'classes': 3, 'class_names': [...], 'schema': {...}}

# Load as pandas DataFrame
df = dataset.to_pandas()

# Create a PyTorch DataLoader with automatic train/val/test splitting
train_loader = dataset.to_pytorch_loader(split="train", batch_size=32)
val_loader = dataset.to_pytorch_loader(split="val", batch_size=32)
```

## Authentication

The library resolves credentials in this order:

1. Inline: `dagnam.configure(api_key="...")`
2. Environment variable: `DAGNAM_API_KEY`
3. Config file: `~/.dagnam/config.json`

Use `dagnam login` from the CLI to save your key to the config file.

## Dataset Loading

`load_dataset()` handles everything: auth, cache check, download (with progress bar), checksum verification, and construction.

```python
# User dataset (UUID)
ds = dagnam.load_dataset("550e8400-e29b-41d4-a716-446655440000")

# System dataset (friendly name)
ds = dagnam.load_dataset("mnist-digits")

# With overrides
ds = dagnam.load_dataset("my-dataset", api_key="...", cache_dir="/tmp/cache")
```

Datasets are cached at `~/.dagnam/datasets/` with SHA256 checksum validation. Subsequent loads skip the download.

## Framework Adapters

### PyTorch

```python
loader = dataset.to_pytorch_loader(
    split="train",       # "train", "val", or "test"
    batch_size=32,
    num_workers=4,
    shuffle=True,        # defaults: True for train, False for val/test
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,             # deterministic splits
)
```

### TensorFlow

```python
tf_dataset = dataset.to_tensorflow_dataset(
    split="train",
    batch_size=32,
    shuffle=True,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
)
```

### Flax / JAX

```python
batches = dataset.to_flax_dataset(
    split="train",
    batch_size=32,
    shuffle=True,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
)
# Returns list[FlaxBatch] where each batch has .features and .labels as JAX arrays
```

### pandas

```python
df = dataset.to_pandas()  # CSV, TSV, JSON, JSONL supported
```

## CLI

```bash
# Save API key
dagnam login

# Browse datasets
dagnam dataset list

# Download a dataset
dagnam dataset download <dataset_id>

# Show dataset metadata
dagnam dataset info <dataset_id>

# View cached datasets
dagnam cache list

# Clear the cache
dagnam cache clear
```

## Caching

Downloaded datasets are stored at `~/.dagnam/datasets/{dataset_id}/` with:
- `meta.json` — cached server metadata
- Data file (e.g., `data.csv`)
- `.checksum` — SHA256 hex digest for staleness detection
- `.last_access` — timestamp for LRU eviction

LRU eviction runs automatically after each download. Default limit is 10 GB, configurable via `~/.dagnam/config.json`:

```json
{
  "api_key": "dgn_...",
  "max_cache_size": 10737418240
}
```

## Supported Formats

| Format | `to_pandas()` | `to_pytorch_loader()` | `to_tensorflow_dataset()` | `to_flax_dataset()` |
|--------|:---:|:---:|:---:|:---:|
| CSV    | ✅ | ✅ | ✅ | ✅ |
| TSV    | ✅ | ✅ | ✅ | ✅ |
| JSON   | ✅ | ✅ | ✅ | ✅ |
| JSONL  | ✅ | ✅ | ✅ | ✅ |
| Image Folder | — | ✅ | — | — |
| Audio Folder | — | ✅ | — | — |

## Image Folder Datasets

Load image classification datasets organized in class-folder structure:

```python
images = dagnam.load_dataset("cats-dogs", version="v1")
train_loader = images.to_pytorch_loader(split="train", batch_size=32)
```

Supports two layouts:
- **Split layout**: `root/{split}/{class}/*.jpg` (train/val/test folders)
- **Unsplit layout**: `root/{class}/*.jpg` (deterministic splits applied)

Requires `torchvision`: `pip install dagnam[pytorch]`

## Audio Folder Datasets

Load audio classification datasets with automatic mel spectrogram conversion:

```python
audio = dagnam.load_dataset("speech-commands")
train_loader = audio.to_pytorch_loader(
    split="train",
    batch_size=32,
)
```

Supports WAV, MP3, and FLAC formats. Audio is converted to mono, resampled to 16kHz, and transformed to mel spectrograms by default.

Requires `torchaudio`: `pip install dagnam[audio]`

## Dataset Versioning

Load a specific version of a dataset:

```python
ds = dagnam.load_dataset("my-dataset-id", version="v2")
```

Versioned datasets are cached separately under `{dataset_id}@{version}/`.

## Presigned URL Downloads

Use a presigned URL for temporary access without an API key:

```python
ds = dagnam.load_dataset(
    "my-dataset-id",
    presigned_url="https://api.dagnam.ai/api/v1/datasets/.../download?token=..."
)
```

Presigned URLs are valid for 7 days and are included in the metadata response.

## Resumable Downloads

Downloads automatically resume from where they left off if interrupted. A `.part` file tracks progress. Disable with `resume=False`:

```python
ds = dagnam.load_dataset("my-dataset-id", resume=False)
```

## Column Roles

Specify column roles for tabular datasets to control feature/target separation:

```python
loader = dataset.to_pytorch_loader(
    split="train",
    batch_size=32,
    column_roles={"x": "feature", "label": "target", "id": "ignore"},
)
```

## Label Detection

When creating data loaders, the library auto-detects the label column:

1. If `feature_schema` has a column with `type: "categorical"` → uses the first one
2. Otherwise → uses the last DataFrame column

Labels are encoded as integers using `class_names` (if provided) or `pd.factorize()`.

## Server Mode

On Dagnam infrastructure (when `DAGNAM_INTERNAL` is set), the library skips HTTP and reads directly from the local filesystem via `DAGNAM_STORAGE_PATH`.

## Development

```bash
cd dag-lib

# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=dagnam

# Lint
uv run ruff check
uv run ruff format
```

## License

MIT
