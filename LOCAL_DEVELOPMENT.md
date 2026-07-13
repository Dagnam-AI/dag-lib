# Local development

dag-lib (the `dagnam` package) talks to the Dagnam backend over HTTP. For local
development you run the SDK from a source checkout and point it at a backend you
have access to (a local dev server, or a staging URL).

## 1. Install `dagnam` locally

Use an editable install so source edits in `dag-lib/` are picked up without
re-installing. Defined optional extras:
`pytorch`, `tensorflow`, `flax`, `audio`, `aio`, `streaming`, `all`.

### Scenario A — developing `dag-lib` itself (uv, recommended)

From inside your `dag-lib` checkout, just run `uv sync`. The project is
automatically installed in editable mode into `.venv/`.

```bash
cd path/to/dag-lib
uv sync --all-extras --dev
```

Do **not** use `uv add` for this — the project doesn't add itself as a
dependency; `uv sync` handles it.

### Scenario B — consuming `dagnam` from another uv project

From the *other* project's directory (point `--editable` at your `dag-lib`
checkout):

```bash
uv add --editable "path/to/dag-lib" --extra all
# or pick specific extras:
uv add --editable "path/to/dag-lib" --extra pytorch --extra aio
```

`uv add` uses `--editable` (long form). `-e` is not accepted.

### Scenario C — plain pip / `uv pip`

In whichever environment you want `dagnam` installed:

```bash
uv pip install -e "path/to/dag-lib"            # core only
uv pip install -e "path/to/dag-lib[pytorch]"   # + pytorch extra
uv pip install -e "path/to/dag-lib[all]"       # everything

# or plain pip:
pip install -e "path/to/dag-lib[all]"
```

### Verify the install

```bash
python -c "import dagnam; print(dagnam.__file__)"
```

The printed path should be inside your `dag-lib` checkout (`.../dag-lib/dagnam/...`).

## 2. Point `dagnam` at a backend

The backend dev server typically runs at `http://localhost:8000`. Start it per
the backend project's own instructions, then tell `dagnam` where it lives.

Resolution order is **override arg → environment variables → config file**, so
env vars are the cleanest for dev.

#### Option A — env vars (one-shot, recommended)

```bash
export DAGNAM_API_URL="http://localhost:8000"
export DAGNAM_API_KEY="<a dev key from your backend>"

python -c "import dagnam; ds = dagnam.load_dataset('<some-uuid>'); print(ds.name)"
```

#### Option B — `dagnam login` (persists to config file)

```bash
dagnam login --api-url http://localhost:8000
# prompts for API key via getpass; writes to the user config file
```

#### Option C — `dagnam.configure(...)` in code

```python
import dagnam

dagnam.configure(
    api_url="http://localhost:8000",
    api_key="<dev-key>",
)
```

### Getting a dev API key

The backend issues API keys. Create a user + key through the backend's signup
flow (or its seed/dev script if it has one) for local development.

## 3. Smoke test end-to-end

```python
import dagnam

dagnam.configure(api_url="http://localhost:8000", api_key="<dev-key>")

# Pick a dataset_id that exists in your backend
ds = dagnam.load_dataset("<uuid-from-your-backend>")
print(ds.name, ds.format, ds.num_samples)

feats, labels = ds.to_arrays()
print(feats.shape, labels.shape if labels is not None else None)
```

If the HTTP request fails:

1. Confirm the backend is up — `curl http://localhost:8000/health`
   (or whatever the backend's health endpoint is).
2. Confirm the API key is valid against your backend.

## Gotcha — `allow_redirects=False`

The HTTP client hardcodes `allow_redirects=False`. If the backend ever responds
with a redirect (e.g. a FastAPI trailing-slash redirect), the request will fail.
Hit the exact path the client uses — no URL rewrites in front of it.
