# dagnam

[![PyPI version](https://img.shields.io/pypi/v/dagnam.svg)](https://pypi.org/project/dagnam/)
[![Python versions](https://img.shields.io/pypi/pyversions/dagnam.svg)](https://pypi.org/project/dagnam/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/Dagnam-AI/dag-lib/actions/workflows/dag-lib-ci.yml/badge.svg)](https://github.com/Dagnam-AI/dag-lib/actions/workflows/dag-lib-ci.yml)

The official Python SDK for Dagnam.AI.

`dagnam` lets Python users work with Dagnam datasets, checkpoints, training
streams, deployments, projects, code generation, and the Model Hub from scripts,
notebooks, services, and generated training code.

The API is usable today and stays backwards-compatible within a minor release
line where practical, but the SDK is still marked alpha while the platform API
continues to mature.

## Installation

```bash
pip install dagnam
```

Requires Python 3.12 or newer. The test suite runs on Python 3.12.

Optional framework extras:

```bash
pip install "dagnam[pytorch]"      # torch + torchvision
pip install "dagnam[audio]"        # torch + torchaudio
pip install "dagnam[tensorflow]"   # tensorflow
pip install "dagnam[flax]"         # jax + flax
pip install "dagnam[streaming]"    # SSE training/deployment streams
pip install "dagnam[aio]"          # async client
pip install "dagnam[audit]"        # workload audit: OS-keyring storage for deployment keys
pip install "dagnam[all]"          # all optional integrations
```

## Authentication

Create an API key in the web app under **Settings, Security**. API keys need a
paid plan. Keys start with `sk_`, are shown once, and are rotated or revoked
there too.

The SDK resolves credentials in this order:

1. Explicit arguments such as `api_key=...` or `dagnam.configure(api_key=...)`
2. `DAGNAM_API_KEY`
3. `~/.dagnam/config.json`

```python
import dagnam

dagnam.configure(api_key="sk_...")
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

Call a deployed model with the deployment's own key (see [Deployments](#deployments)):

```python
result = dagnam.inference(
    deployment_id="dep_abc123",
    inputs={"text": "Classify this sentence."},
    api_key=deployment_key,
)
```

Download a training job's latest checkpoint, or its best one with `prefer_best=True`:

```python
checkpoint_path = dagnam.download_checkpoint("job_xyz789")
best_path = dagnam.download_checkpoint("job_xyz789", prefer_best=True)
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

# Built-in dataset by name (dagnam.datasets.list_system() lists them)
mnist = dagnam.load_dataset("MNIST Handwritten Digits")

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

`inference`, `inference_batch`, `deployment_health` and `inference_schema`
authenticate with the deployment's own key, the `api_key` returned when the
deployment was created, not with your account key. Pass the model input itself;
the SDK sends it as `{"input": ...}`.

```python
prediction = dagnam.inference("dep_abc123", {"text": "hello"}, api_key=deployment_key)

batch = dagnam.inference_batch(
    "dep_abc123",
    [{"text": "hello"}, {"text": "world"}],
    api_key=deployment_key,
)

health = dagnam.deployment_health("dep_abc123", api_key=deployment_key)

for event in dagnam.stream_training("job_xyz789"):
    print(event.event, event.data)

path = dagnam.download_checkpoint("job_xyz789")
```

Checkpoints are cached separately under `~/.dagnam/checkpoints/` with SHA-256
verification when the backend provides a checksum.

## Deployments

Deploy a model version from the model registry (**My models** in the web app) in
one call. Fine-tuned models and curated base models can be deployed; models
trained in the Studio cannot be deployed yet.

```python
deployment = dagnam.deployments.deploy_model_version(
    "mv_abc123",
    name="support-classifier",
)
deployment_key = deployment["api_key"]  # returned once: store it now

revisions = dagnam.deployments.revisions(deployment["id"])
logs = dagnam.deployments.logs(deployment["id"], level="error")

# once the newest revision reports is_active:
result = dagnam.inference(deployment["id"], {"text": "hello"}, api_key=deployment_key)
```

The deployment serves once its newest revision reports `is_active`; a failed
revision carries a `failure_reason`. The platform manages serving capacity, so
there is no scale, rollback or retry: to serve a different model, deploy that
version. `pause` and `resume` return `LongRunningOperation` objects. Read
operations such as `list`, `get`, `health`, `metrics`, `revisions` and `logs`
return data from the API.

A first call after idle may take up to the deployment's cold-start budget, which
is minutes, so the default timeout for `inference` and `inference_batch` is 600
seconds. Lower it with `timeout=` (or `--timeout` on the CLI) if you prefer to
fail fast and retry on a 503 with `Retry-After`. `dagnam.set_warm(deployment_id,
True)` (`dagnam deployments warm <id> --on`) keeps one container running to avoid
the cold start; it costs for as long as it is on.

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

async with AsyncDagnamClient("https://api.dagnam.ai", "sk_...") as client:
    datasets = await client.list_datasets()

# Inference authenticates with the deployment's own key.
async with AsyncDagnamClient("https://api.dagnam.ai", deployment_key) as serving:
    result = await serving.predict("dep_abc123", {"text": "hello"})
```

The async client mirrors the low-level HTTP client surface. High-level resource
helpers such as `dagnam.deployments.deploy_model_version()` are currently
synchronous.

## CLI

```bash
dagnam login

dagnam dataset list
dagnam dataset info <dataset-id>
dagnam dataset download <dataset-id>
dagnam cache list
dagnam cache clear

DAGNAM_API_KEY=<deployment key> dagnam inference run <deployment-id> --input '{"text":"hello"}'
dagnam checkpoint list <job-id>
dagnam checkpoint download <job-id>
dagnam stream <job-id>

dagnam deployments list
dagnam deployments deploy-version <version-id> --name support-classifier
dagnam hub search --search resnet
dagnam models push --name tiny-chat --slug tiny-chat --description "..." --file weights.safetensors
dagnam models list
dagnam models artifacts <version-id>
dagnam models download <version-id> <artifact-id>
dagnam projects list
dagnam codegen preview <project-id>

dagnam audit scan traces.jsonl --source langfuse --out ./audit   # offline: discover + price workloads
dagnam audit run ./audit      # train, serve and score the candidates; also status / cancel / delete

dagnam agent install          # install the Agent Skill into Claude Code / Codex
dagnam agent uninstall --all
```

Rescanning the same export preserves a run's progress. A changed export requires a new
`--out`, or `--force` for a local-only run: the old candidates are retired for cleanup,
and subsequent training and scoring start fresh. Retiring a candidate does not stop it: a
run of its that is still training keeps training, and billing, and its endpoint stays up.
The scan names every retired run and endpoint that is still live, `dagnam audit status`
lists them as `retired` under their workload, and `dagnam audit cancel ./audit` stops them. Once an audit has
been published, changed rows always require a new `--out`, even with `--force`.
A rescan removes only the `workloads/<id>/` folders that the directory's previous scan wrote
and the new one no longer derives; anything else under `workloads/` is left alone and named in
the scan's warnings. `--out` may be a symbolic link to a directory (it is resolved once), but a
link inside the audit directory, or a pipe or directory where the scan expects a file, is refused
with the path named.

**What the scan does with a model's reasoning.** It is kept out of the training prompts and
answers for the reply shapes the scan reads: a chat message, Chat Completions, Responses, Anthropic
blocks, Gemini, Bedrock Converse, Ollama, Cohere, LangChain and AI SDK parts, and the same as JSON
text; a message's own reasoning fields (`reasoning_content`, `reasoning_details`,
`reasoningDetails`, `reasoning_summary`, ...); `<think>`, `<thinking>`, `<thought>` and
`<reasoning>` tags before the answer; and a reply that has reasoning, `</think>` alone on its line,
then the answer, or nothing (DeepSeek-R1, Qwen3, QwQ without a reasoning parser). A call that is
only reasoning, such a reply without its answer included, is kept in the workload's calls and spend and gives no training row, and the scan says
how many. **Not covered:** a response object of an SDK or tool the scan does not know, logged
whole by hand, is kept as written unless its reasoning sits under a name that is only ever
reasoning. `reasoning`, `thinking`, `thoughts`, `rationale` and `scratchpad` are also the names
of a structured answer's own fields, so they are not guessed at; raw channel-delimited text of the
open-weight OpenAI models, content parts typed `analysis`, a complete reasoning tag in the
middle of an answer, and a `</think>` that is not alone on its line (the reasoning's last sentence
ends on it, the answer starts on it) or is upper case are kept too. If your logs hold such objects, map the answer's own field
with `--map response=<column>`.

**`audit scan` prices the providers your agent actually calls.** The bundled price table
(`dagnam/audit/prices/2026-09.json`, copied from each vendor's own pricing page on the table's
`as_of` date) carries OpenAI, Anthropic, Google Gemini, Mistral, DeepSeek, xAI and Cohere rows,
and a model id is matched against it after normalization: lowercased, with the provider or
region namespace stripped (`anthropic/claude-sonnet-5`, `models/gemini-2.5-flash`,
`openrouter:mistral/mistral-large-latest`, `us.`/`eu.`/`global.`/`apac.anthropic.claude-...`),
the model of a resource path kept (`projects/.../publishers/google/models/gemini-2.5-flash`,
`accounts/fireworks/models/...`), a floating `-latest` dropped, and -- only when the exact id
has no row of its own -- a trailing release date, Vertex `@date`, Gemini `-001` or Bedrock
version suffix dropped too (`gpt-4o-2024-08-06`, `claude-sonnet-4-5@20250929`, `...-v1:0`).
Token usage is read in every vendor's spelling, so Anthropic's `input_tokens` plus its
cache-read counts, Gemini's `promptTokenCount` and LangSmith's `usage_metadata.input_tokens`
are counted exactly like OpenAI's `prompt_tokens`, and cache reads are priced at the cached
rate. A model the table does not list still prices to nothing, never a guess, and an export
with neither a cost nor token counts is `unknown_cost`, never a free teacher.

**`audit scan` reads what your tools export, and streams it.** Langfuse (the observations API,
its blob export and its UI CSV/JSON exports), LangSmith runs, OpenAI Batch and stored
completions -- Chat Completions or the Responses API -- and any JSONL, JSON or CSV file through
`--source jsonl|csv --map target=column`, where a column may be a dotted path into a nested
record (`--map messages=input.inputBodyJson.messages` reads a Bedrock invocation log). Files
may be gzipped; timestamps may be ISO text or an epoch in seconds, milliseconds, microseconds
or nanoseconds. The export is streamed, not held in memory (a `.json` document, one JSON
value, is the exception): one pass discovers the workloads, and two more derive the rows of
the ones worth auditing. A workload is keyed by the name you gave the step
(`metadata.workload`, or a registered prompt's name) when there is one, else by its system
prompt's template plus the name of the request's response schema or forced tool (never the
tools it offers, which an agent changes from call to call), else, without a system prompt, by
the shape of its answers. A tool-calling workload is audited on its calls, as
`{"name", "arguments"}` JSON (a list when the teacher made several), and the scan says what its
replacement returns; a workload whose calls carry images, audio or files is not audited. When the
export holds a sample of your traffic, `--sample-rate 0.1` scales volume and spend back up.
Each candidate trains on at most 5,000 rows (a sample by label or route), so it finishes
inside its recipe's one-hour limit; the scan report says when a workload was sampled. An SFT
candidate's student reads 2,048 tokens, so training rows the scan estimates are longer are
left out and counted, and a workload left with fewer than 32 is `too_few_samples`. The
estimate needs no tokenizer and guesses no language: it normalises to NFC as the student's
tokenizer does, starts every character at its UTF-8 bytes, and discounts it only where the
student's own vocabulary shows a merge. On natural text of the kinds measured it is at least 0.95
of the student's count (0.954 at the lowest of 2,498 held-out rows), 1.04 at the
median and 1.22 at the 95th percentile: about 1.2 for German, Spanish, French and Portuguese,
1.12 for code, 1.06 for Chinese. A few scripts with no natural text to measure on (Syriac,
Cherokee, Gothic and the like) are priced at their bytes. On random or crafted text there is no such floor: it can
count about half the real tokens (0.486 at the lowest on random Cyrillic words, 0.504 on random
Thai, 0.547 to 0.62 on a repeated number sign or random Tibetan, 0.75 on licence keys, 0.88 to 0.90
on base32 over plausible text), and 0.36 on crafted alternating han units, so a row estimated to
fit the 2,048-token cap can exceed it by up to about 2.8 times on crafted text. Pairs of han units
that break each other and cycles of 9 or more fragile han units are not seen by its 8-unit window,
a known limit. It is
never above the UTF-8 byte count of the NFC form of the text. A text of a few
tokens rounds up.

**Watch it on the website.** `dagnam audit run` mirrors the audit into your account as it
goes -- the scan's workloads (ids, verdicts, spend, masked template excerpts and the redaction
counts) and then each candidate's progress, ending in its agreement, its measured latency and
what the replay cost -- so the audit page shows a run in flight and keeps the report when it
finishes. When the audit is created the run prints where to watch it
(`published: <audit-id> — watch it at https://dagnam.ai/audits/<audit-id>`), and
`dagnam audit status ./audit` repeats the link. The derived rows, the raw traces and the
deployment keys stay on your machine. Once the audit exists, a publish that fails is retried with
the next step, at the end of the run, or on the next `dagnam audit run`, rather than stopping
the run; but the audit itself must exist first (below). A run that stopped short is published as halted, and each
`dagnam audit run` resumes its audit when it starts, before it waits on anything. Cancel or
Delete on the audit page stops a live run at its next step: the platform stops the jobs and
endpoints it knows about at once, the run's next publish meets the halt (or the missing
audit), and the run halts as `cancelled` (or `deleted`) and starts nothing more.
`dagnam audit run ./audit --local-only` publishes nothing to your account (the rows are
still uploaded to the platform and trained there). A published run creates its audit first
and sends the audit's id with every dataset, training run and endpoint it creates, so the
platform tags each one at creation; if the audit cannot be created the run stops, before it
uploads anything, and says to run it again. (A `--local-only` run has no audit to tag with, so each
dataset's description carries a `[<directory key>/<workload>/<candidate>]` marker that only this
machine's rerun reads, to find its own upload after an interrupted one; a published run's datasets
are found by the audit's tag instead and carry no marker.)

**Cancel and delete.** For a published audit the platform's own cancel and delete are the only
things that touch its resources. `dagnam audit cancel ./audit` and `dagnam audit delete ./audit`
ask the platform once, print its receipt, record what it decided, write `cancelled.json` /
`deleted.json`, and (for a delete the platform completed) remove the local `workloads/` rows and
the deployment keys. They never finish the job with calls of their own: if the platform does not
answer, refuses (other than for a serving endpoint, below), or times out, the receipt has one `audit blocked [not_answered]` row, nothing
remote or local is touched, and the exit status is 1 -- run it again. If another cancel or delete
of the same audit is still walking it (`409 teardown_in_progress`), or the platform cannot take its
lock just now (`503 teardown_unavailable`), the command waits (the platform's `Retry-After`, whatever
its length up to two minutes in all; five seconds when it sends none) and asks again. A 404 decides nothing: it says
only that THIS key cannot see the audit (another account's key after `dagnam login`, another
host, or a deleted audit). The command prints the host and masked key it asked, keeps every local
row, the deployment keys and `state.json`, and exits 1; ask again with the right key. Only the
platform's positive answer marks an audit deleted. If you know it is deleted (the website, or an
older platform answering a repeat delete with 404), `dagnam audit delete ./audit
--already-deleted` removes the local files; that cannot be undone, so check `dagnam whoami` first. A delete the platform reports `halted` (something was
refused) removes nothing local and exits 1; the next delete asks again.

**An endpoint that is serving.** On a platform that supports it, `dagnam audit delete` deletes
nothing while an endpoint of the audit is serving, because deleting it would break any app that
calls it. It prints the platform's sentence, a table of the blocking endpoints and the two ways
forward, writes no receipt, removes no file, keeps the deployment keys, and exits 1 (under `--json`,
stdout is `{"error", "hint", "code": "endpoints_serving", "endpoints"}`; in Python it is
`dagnam.exceptions.EndpointsServingError`, an `APIError`). Run `dagnam audit cancel ./audit`, which
pauses every endpoint that is serving now, and delete again; an endpoint that is still rolling
out cannot be paused, so wait for it to finish. To delete the endpoints too, whatever calls them,
pass `--include-endpoints` (`include_endpoints=True` in Python); the receipt then marks each
deployment it deleted while it was serving `(was serving)`. The endpoints this client deletes
itself (an unpublished audit's, and any the platform declined to claim) are checked the same way
before anything is deleted, and only an endpoint known to be paused, stopped, failed or not yet
provisioned is treated as safe: `deploying`, an unreadable state and a state this version does not
know all count as serving. A platform that does not support the check deletes without asking. A run still going keeps
`state.json` and its lock: a cancel asks the platform, which halts the audit, and the run stops at
its next step. A deleted audit's directory refuses `run` and `cancel`.

An audit that was never published (`--local-only`, or a run that never reached `create_audit`) has
no platform record, so `state.json`'s own ids -- only what this directory created -- are stopped
and deleted from here, each re-read to confirm it is gone; a registry version is purged through its
own route (`DELETE /api/v1/model-versions/{id}`), never by deleting its registry entry, and one the
platform cannot purge is reported `blocked [platform_only]`; one a live endpoint serves is refused
by the platform and shown as `kept [weights_served]` (the delete finishes around it). A dataset a
project this directory did not create has linked is `kept [in_use_elsewhere]`, never deleted, and
recorded in `state.json` so no later walk or cancel touches it; if the owner's projects cannot be
read to check (an answer of the wrong shape, a failed read, more than 100 pages of projects), the
dataset is `blocked`, not guessed at. The project stays while anything in it
does. A walk changes nothing local, and marks nothing, unless its key is shown to see the account
(not-found is what another account's key is told for every id, and a walk where the rest failed
cannot tell the two apart): by a deletion this same key and host confirmed earlier
(`confirmed_gone` and `confirmed_by`, a digest of both, in `state.json`; another account's key never
counts), by a delete or stop that just succeeded, by a keep the platform's own purge answered, or by a read
of a recorded deployment or training run (a project, dataset or model version can be public, and a
dataset can be linked into another account's project, so reading or finding those proves nothing).
An id whose delete (or not-found) and re-read both failed also keeps the
deployment keys and the local rows, whatever else the walk did. The receipt and the error name the
host and masked key asked and `dagnam audit delete --already-deleted` as the way to say it is gone.
A `state.json` that is not JSON, or has a field of the wrong type, is named and stops the command
before anything is touched. The ids live in `state.json`: if it is lost, nothing in this directory
can find what an unpublished run created (a published audit's resources are still found by the
platform, from the audit's page); and a create whose answer was lost, in a run nobody finished, is
in no state file at all.

**Publishing a directory first run `--local-only`.** The audit is created first, and then asked to
claim everything the earlier run made (datasets, runs, endpoints, the project). A claim the
platform refuses is not a verdict that it is not yours: this directory created it, so `audit delete`
removes it itself once the platform has deleted the audit, and `audit cancel` stops it; the run says
so when it happens. The one exception is a refusal with the code `in_use_elsewhere` (a dataset one of
your other runs uses, or the run that made it): that is yours, not this directory's, so it is
recorded as kept, shown, and never stopped or deleted from here. What the platform's own receipt has
a row for is the platform's, and is never walked from here. A refused id this client cannot remove
(a failure, a refusal) is named in the receipt and the command exits 1 once; it does not keep the
directory out of the deleted state, and a repeat delete finishes. A claim request that fails as a
whole halts the run (`publish_failed`), tells the platform the run stopped, says that the earlier
run's resources still exist (an endpoint may be serving: `dagnam audit cancel` stops it), and is
asked again next time. Publish a `--local-only` directory only once its run has finished: a create whose answer was lost in the unpublished run cannot be found again after the audit exists (the platform replays a create only for an identical body, and the published body names the audit). **An audit an
older dagnam published** never named its resources to the platform, so the platform keeps what it
cannot prove the audit created; `audit delete` offers to claim them before it deletes (one
confirmation), and says plainly which of this directory's resources the platform kept and why. A
claim that cannot be made (the platform has no such route, or does not answer) is said, and the
delete goes on without it.

Each receipt row has a `status` and a stable `code`; this client reads those and never the wording
of a reason. `deleted`, `already_absent` and `stopped` mean nothing is left; `stopped` with the
code `already_stopped` means the run had already finished (or the endpoint was paused) and marks
nothing, so a finished run stays resumable. `kept` is a decision to leave something that is not
the audit's to take -- a project that holds your other work, weights another deployment still
serves, a run, endpoint or dataset that now belongs to another project, something a step named
but the audit did not create -- and the audit is deleted around it: it is shown, recorded in
`state.json` (`kept_ids`), never touched from here, and never fails a command. `blocked` is one
of the audit's own resources refused or failed, and is what is left: it is never retried from
here. A platform older than the `code` column is read from the status and the four reasons it
used; a row with a status this version does not know is shown as sent, left alone, and counts as
not finished. Rows this client writes carry `not_answered`, `not_removed` (a local file behind a
link, or an unpublished id that would not go) or `platform_only`.

Exit status, from one function: `delete` and `cancel` exit 1 if and only if something of the
audit's own is left -- a `blocked` row, a row this version cannot read, a platform that did not
answer, a delete the platform halted, or a local file behind a link (never followed). A `kept` row
never fails a command. The last line says whether everything is gone, everything except what was
kept on purpose, or what is left. A platform answer of the wrong shape on a route other than
cancel and delete (an empty object where a document is expected, say) ends the command with a
generic "unexpected error" naming the Python error rather than the route: a known limit, not a
state change. Commands resolve the audit directory argument once, so a
directory named through a link (a temporary or home directory) works; links inside it are refused.

**The platform re-scans what you upload, and a disagreement stops the workload.** Your rows
are redacted on your machine before they are uploaded (before the character budget cuts a row,
so no cut exposes part of an identifier, and again on the cut row exactly as it is uploaded, so it
scans clean and a second pass changes nothing), and the platform scans the uploaded
dataset again before anything is trained on it. If the two disagree the workload stops as
`pii_disagreement`, and the reason, printed under the workload's line and written to the
report (each candidate's `error`), says which of three things happened: the platform runs an older privacy contract than this SDK (it scanned
fewer classes), it runs a newer one (it found matches in classes the scan did not look
for: upgrade `dagnam-contracts` and scan again), or the two detectors really disagree. When
the platform's contract differs from this install's at any level, the reason names both versions
and which to move. The rows are already uploaded when the check runs; `dagnam audit delete
./audit` removes them, and, since there is no narrower form, the whole audit with them.

**`--max-credits` is a hard ceiling.** `dagnam audit run` never starts a training run or a
holdout replay that could take the credits spent past it: a run is budgeted at the most its
recipe can charge (120 credits) and a replay at one credit per holdout row plus 10%. Without
the flag the ceiling is what the audit already spent plus the plan's estimate for what is
left, rounded up to 100, printed in the listing you confirm. A candidate whose replay saw more than 10% of its calls fail is reported
`unreliable` and never becomes the winner.

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
  That covers one call. `dagnam audit run` goes further (`DagnamClient.resume_creates`):
  its run, deployment and audit creates derive the key from the request, so a
  run you interrupt and start again finds what the first attempt created
  instead of paying for it twice, for as long as the platform keeps the answer
  (24 hours).
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
  "api_key": "sk_...",
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
| `DAGNAM_DATASET_ID` | Training dataset id recorded when `dagnam.training.init` registers a local run |

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
| `DAGNAM_JOB_ID` | The platform job the run reports to; also the default `job_id` of `dagnam.models.push_run_artifacts` |

## Compatibility

The SDK talks to the hosted Dagnam API (`https://api.dagnam.ai`), which always
runs its current version. Use the latest SDK release with it.

**The platform is upgraded first, then the SDK is released.** The workload audit
redacts your rows with the `dagnam-contracts` version this SDK installs, and the
platform re-scans them with its own, so the platform must run at least the contract
minor the SDK installs. Each SDK release therefore pins that minor (`dagnam-contracts>=X.Y,<X.(Y+1)`)
and is published only after the hosted platform reports a contract at or above
it; the release pipeline checks this before it builds anything. `dagnam audit
run` checks it too, before its first upload (with `--local-only` as well, which
still uploads and trains on the platform): against a platform that is behind
(a private deployment not upgraded yet, say), or one that does not report its
contract, it stops with `platform_too_old`, names both versions, and uploads and
spends nothing by this run. Upgrade the platform. A platform a patch ahead of the
installed contract warns first and names `pip install -U dagnam-contracts`: a
patch changes what is found inside the same classes. The check is made by
`dagnam.audit.orchestrate.run_audit` itself, so library callers get it too.

| SDK version | Status |
| --- | --- |
| The latest release | Current release line |
| Any earlier release | Superseded: upgrade with `pip install -U dagnam` |

The SDK follows semantic versioning. While the package is alpha, a minor release
may change or remove public APIs, and [CHANGELOG.md](CHANGELOG.md) lists every
such change; patch releases should avoid breaking the documented behavior of
their minor release line.

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
