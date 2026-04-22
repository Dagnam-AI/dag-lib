# Graph Report - .  (2026-04-19)

## Corpus Check
- Corpus is ~11,106 words - fits in a single context window. You may not need a graph.

## Summary
- 421 nodes · 856 edges · 19 communities detected
- Extraction: 55% EXTRACTED · 45% INFERRED · 0% AMBIGUOUS · INFERRED: 385 edges (avg confidence: 0.66)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Dataset & Framework Loaders|Dataset & Framework Loaders]]
- [[_COMMUNITY_CLI & API Client|CLI & API Client]]
- [[_COMMUNITY_Cache Management|Cache Management]]
- [[_COMMUNITY_Auth & Configuration|Auth & Configuration]]
- [[_COMMUNITY_CSV PyTorch Loader|CSV PyTorch Loader]]
- [[_COMMUNITY_SDK Concepts & Architecture|SDK Concepts & Architecture]]
- [[_COMMUNITY_Loaders Package & Internal Load|Loaders Package & Internal Load]]
- [[_COMMUNITY_LRU Cache Eviction|LRU Cache Eviction]]
- [[_COMMUNITY_UUID Routing & System Datasets|UUID Routing & System Datasets]]
- [[_COMMUNITY_Column Roles Strategy|Column Roles Strategy]]
- [[_COMMUNITY_Test Fixtures|Test Fixtures]]
- [[_COMMUNITY_Entry Point|Entry Point]]
- [[_COMMUNITY_JSONJSONL Loader|JSON/JSONL Loader]]
- [[_COMMUNITY_Cache Summary API|Cache Summary API]]
- [[_COMMUNITY_Sequence Padding|Sequence Padding]]
- [[_COMMUNITY_Column Roles Property Test|Column Roles Property Test]]
- [[_COMMUNITY_Ignore Column Role|Ignore Column Role]]
- [[_COMMUNITY_Column Order Preservation|Column Order Preservation]]
- [[_COMMUNITY_Package Initializers|Package Initializers]]

## God Nodes (most connected - your core abstractions)
1. `DagnamDataset` - 105 edges
2. `DatasetNotFoundError` - 44 edges
3. `DagnamClient` - 38 edges
4. `APIError` - 32 edges
5. `AuthError` - 29 edges
6. `load_dataset()` - 28 edges
7. `DagnamError` - 19 edges
8. `ChecksumError` - 17 edges
9. `create_pytorch_loader()` - 17 edges
10. `main()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `Authentication resolution for the dagnam library.  Resolves API key and API UR` --uses--> `AuthError`  [INFERRED]
  dagnam\auth.py → dagnam\exceptions.py
- `Store inline credentials in module-level state.` --uses--> `AuthError`  [INFERRED]
  dagnam\auth.py → dagnam\exceptions.py
- `Resolve API key: override → inline → DAGNAM_API_KEY env var → config file.` --uses--> `AuthError`  [INFERRED]
  dagnam\auth.py → dagnam\exceptions.py
- `Resolve API URL: override → inline → DAGNAM_API_URL env var → config file → defa` --uses--> `AuthError`  [INFERRED]
  dagnam\auth.py → dagnam\exceptions.py
- `CSV/TSV loader — converts tabular data into PyTorch DataLoaders.` --uses--> `DagnamDataset`  [INFERRED]
  dagnam\loaders\csv_loader.py → dagnam\dataset.py

## Hyperedges (group relationships)
- **Framework Adapter Suite** — readme_pytorch_adapter, readme_tensorflow_adapter, readme_flax_adapter, readme_pandas_adapter [INFERRED 0.90]
- **Credential Resolution Chain** — readme_configure_fn, readme_dagnam_api_key_env, readme_config_file [EXTRACTED 1.00]
- **Cache Integrity System** — readme_caching, readme_sha256_checksum, readme_lru_eviction [INFERRED 0.88]
- **Development Toolchain** — readme_uv, readme_pytest, readme_ruff [EXTRACTED 1.00]

## Communities

### Community 0 - "Dataset & Framework Loaders"
Cohesion: 0.05
Nodes (43): DagnamDataset, _pad_sequences(), DagnamDataset class for lazy-loading and converting datasets., Create a PyTorch DataLoader for the specified split.          When ``_native_t, Represents a loaded dataset with metadata and conversion methods.      Data is, Build a DataLoader from native train/test datasets., Build a DataLoader from numpy array tuples (e.g. IMDB)., Create a TensorFlow Dataset for the specified split.          Raises ImportErr (+35 more)

### Community 1 - "CLI & API Client"
Cohesion: 0.08
Nodes (40): Command-line interface for the dagnam client library.  Usage:     dagnam logi, Format byte count as a human-readable string., Recursively compute total size of a directory in bytes., Print an error message to stderr and exit., DagnamClient, _parse_filename(), _raise_for_status(), HTTP client for the Dagnam.AI API. (+32 more)

### Community 2 - "Cache Management"
Cohesion: 0.06
Nodes (27): compute_file_checksum(), get_cache_dir(), get_cache_info(), is_cached(), load_metadata(), Local cache management for the dagnam library.  Manages the local dataset cach, Write meta.json to cache directory., Read meta.json from cache directory. Returns empty dict if file doesn't exist. (+19 more)

### Community 3 - "Auth & Configuration"
Cohesion: 0.07
Nodes (32): configure(), get_api_key(), get_api_url(), Authentication resolution for the dagnam library.  Resolves API key and API UR, Store inline credentials in module-level state., Resolve API key: override → inline → DAGNAM_API_KEY env var → config file., Resolve API URL: override → inline → DAGNAM_API_URL env var → config file → defa, _build_parser() (+24 more)

### Community 4 - "CSV PyTorch Loader"
Cohesion: 0.09
Nodes (17): create_pytorch_loader(), _detect_label_column(), _encode_labels(), CSV/TSV loader — converts tabular data into PyTorch DataLoaders., Return the label column name.      Priority:     1. First column with type ``, Encode a label series into a ``long`` tensor.      If *class_names* is provide, Internal PyTorch Dataset wrapping feature and label tensors., Create a PyTorch DataLoader from a CSV/TSV dataset.      Label detection, enco (+9 more)

### Community 5 - "SDK Concepts & Architecture"
Cohesion: 0.1
Nodes (29): Authentication / Credential Resolution, Local Dataset Cache (~/.dagnam/datasets/), dagnam CLI, Config File (~/.dagnam/config.json), dagnam.configure() Function, Dagnam.AI Platform, DAGNAM_API_KEY Environment Variable, DAGNAM_INTERNAL Environment Variable (+21 more)

### Community 6 - "Loaders Package & Internal Load"
Cohesion: 0.08
Nodes (17): _load_internal(), Loaders package — format-specific dataset factories.  Available loaders (impor, Load dataset from sidecar metadata (server-side training).      Reads ``.dagna, Load a system dataset using its native library internally.      Matches on the, resolve_system_dataset(), Tests for native system dataset loading and sidecar metadata., Verify sidecar metadata loading for server-side training., User dataset sidecar → direct file read. (+9 more)

### Community 7 - "LRU Cache Eviction"
Cohesion: 0.19
Nodes (7): evict_lru(), get_cache_size(), Calculate total size of the cache directory in bytes., Evict least-recently-used datasets until cache is under max_size_bytes.      R, Helper to create a fake cached dataset., TestEvictLru, TestGetCacheSize

### Community 8 - "UUID Routing & System Datasets"
Cohesion: 0.13
Nodes (13): _is_uuid(), Return True if *s* looks like a standard UUID (8-4-4-4-12 hex)., Tests for system dataset name resolution in load_dataset()., Names like 'imdb-sentiment' are NOT UUIDs and should use system endpoints., UUID IDs should route through the regular user dataset endpoints., Verify the UUID detection helper., Friendly names should route through system dataset endpoints., _sha256() (+5 more)

### Community 9 - "Column Roles Strategy"
Cohesion: 0.22
Nodes (12): Separate feature and target columns using an explicit role mapping.      Retur, _split_by_roles(), column_roles_strategy(), _make_df(), Property tests for dagnam CSV loader column roles round-trip.  # Feature: data, Generate a dict of unique column names → roles with ≥1 target., Build a trivial DataFrame with the given columns., Validates: Requirements 15.1, 15.2, 15.4 (+4 more)

### Community 10 - "Test Fixtures"
Cohesion: 0.2
Nodes (9): cache_dir(), Shared test fixtures for the dagnam client library., Temporary cache directory for dataset storage during tests., Sample metadata dict matching the MetadataResponse shape from the API., Sample CSV string content for testing loaders., Sample JSON string content for testing loaders., sample_csv_data(), sample_json_data() (+1 more)

### Community 11 - "Entry Point"
Cohesion: 1.0
Nodes (0): 

### Community 12 - "JSON/JSONL Loader"
Cohesion: 1.0
Nodes (1): JSON/JSONL loader — converts JSON/JSONL data into PyTorch DataLoaders.  Delega

### Community 13 - "Cache Summary API"
Cohesion: 1.0
Nodes (1): Return a summary dictionary with the 8 required keys.

### Community 14 - "Sequence Padding"
Cohesion: 1.0
Nodes (1): Pad/truncate variable-length integer sequences (e.g. IMDB).

### Community 15 - "Column Roles Property Test"
Cohesion: 1.0
Nodes (1): For any valid column_roles with ≥1 target, _split_by_roles returns         feat

### Community 16 - "Ignore Column Role"
Cohesion: 1.0
Nodes (1): Columns with role 'ignore' do not appear in features or target.

### Community 17 - "Column Order Preservation"
Cohesion: 1.0
Nodes (1): Feature columns preserve original DataFrame column order.

### Community 18 - "Package Initializers"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **59 isolated node(s):** `Local cache management for the dagnam library.  Manages the local dataset cach`, `Returns ~/.dagnam/datasets/{dataset_id}/ (or custom base).      Creates the di`, `True if .checksum file exists and matches server_checksum.`, `Update the .last_access timestamp for a cached dataset.`, `Calculate total size of the cache directory in bytes.` (+54 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Entry Point`** (2 nodes): `main()`, `hello.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `JSON/JSONL Loader`** (2 nodes): `json_loader.py`, `JSON/JSONL loader — converts JSON/JSONL data into PyTorch DataLoaders.  Delega`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Cache Summary API`** (1 nodes): `Return a summary dictionary with the 8 required keys.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Sequence Padding`** (1 nodes): `Pad/truncate variable-length integer sequences (e.g. IMDB).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Column Roles Property Test`** (1 nodes): `For any valid column_roles with ≥1 target, _split_by_roles returns         feat`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Ignore Column Role`** (1 nodes): `Columns with role 'ignore' do not appear in features or target.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Column Order Preservation`** (1 nodes): `Feature columns preserve original DataFrame column order.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Initializers`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DagnamDataset` connect `Dataset & Framework Loaders` to `CLI & API Client`, `Auth & Configuration`, `CSV PyTorch Loader`, `Loaders Package & Internal Load`, `UUID Routing & System Datasets`, `Column Roles Strategy`?**
  _High betweenness centrality (0.458) - this node is a cross-community bridge._
- **Why does `load_dataset()` connect `Auth & Configuration` to `Dataset & Framework Loaders`, `CLI & API Client`, `Cache Management`, `Loaders Package & Internal Load`, `LRU Cache Eviction`, `UUID Routing & System Datasets`?**
  _High betweenness centrality (0.355) - this node is a cross-community bridge._
- **Why does `DagnamClient` connect `CLI & API Client` to `UUID Routing & System Datasets`, `Auth & Configuration`, `Loaders Package & Internal Load`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Are the 95 inferred relationships involving `DagnamDataset` (e.g. with `Loaders package — format-specific dataset factories.  Available loaders (impor` and `Return True if *s* looks like a standard UUID (8-4-4-4-12 hex).`) actually correct?**
  _`DagnamDataset` has 95 INFERRED edges - model-reasoned connections that need verification._
- **Are the 40 inferred relationships involving `DatasetNotFoundError` (e.g. with `DagnamClient` and `HTTP client for the Dagnam.AI API.`) actually correct?**
  _`DatasetNotFoundError` has 40 INFERRED edges - model-reasoned connections that need verification._
- **Are the 28 inferred relationships involving `DagnamClient` (e.g. with `Command-line interface for the dagnam client library.  Usage:     dagnam logi` and `Format byte count as a human-readable string.`) actually correct?**
  _`DagnamClient` has 28 INFERRED edges - model-reasoned connections that need verification._
- **Are the 28 inferred relationships involving `APIError` (e.g. with `DagnamClient` and `HTTP client for the Dagnam.AI API.`) actually correct?**
  _`APIError` has 28 INFERRED edges - model-reasoned connections that need verification._