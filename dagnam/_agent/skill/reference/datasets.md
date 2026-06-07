# Datasets reference

## CLI (`dagnam dataset ...`)
- `dagnam dataset list [--type TYPE] [--search TEXT]` (+ `--json/--verbose/--output`) — list available datasets.
- `dagnam dataset info <dataset_id> [--show-download-url] [--json] [--output]` — metadata for one dataset.
- `dagnam dataset download <dataset_id> [--output-dir .] [--no-progress]` — download a dataset locally.

> There is no `dagnam dataset upload` CLI command — uploading is SDK-only (below).

## SDK (`import dagnam`)
- `dagnam.datasets.upload(path, name, dataset_type, format, description=None, visibility="private", license=None, progress_cb=None) -> dict` — upload a local file as a dataset.
- `dagnam.datasets.upload_from_url(url, name, dataset_type, format, ...) -> LongRunningOperation` — server-side ingest; poll with `op.wait().result()`.
- `dagnam.load_dataset(dataset_id) -> DagnamDataset` — download (cached) + open. Convert into a framework loader:
  - `.to_pytorch_loader(batch_size=..., shuffle=...)`
  - `.to_tensorflow_dataset(...)`
  - `.to_flax_dataset(...)`
  - `.to_polars()` — a Polars DataFrame for inspection / feature work.

## Recipe — upload then use
```python
import dagnam
ds = dagnam.datasets.upload("iris.csv", name="iris", dataset_type="tabular", format="csv")
loader = dagnam.load_dataset(ds["id"]).to_pytorch_loader(batch_size=32, shuffle=True)
```

See `reference/cache.md` for where downloads are stored and how to manage them.
