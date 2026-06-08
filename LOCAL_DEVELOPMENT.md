# Local development with mvp-backend

dag-lib (the `dagnam` package) talks to mvp-backend over HTTP. While mvp-backend
is not yet published, you run both repos from your local checkout and point
the SDK at your local backend.

## 1. Install `dagnam` locally

Use an editable install so source edits in `dag-lib/` are picked up without
re-installing. Defined optional extras:
`pytorch`, `tensorflow`, `flax`, `audio`, `aio`, `streaming`, `all`.

### Scenario A — developing `dag-lib` itself (uv, recommended)

If you are working *inside* `D:/Code/dagnam-ai/dag-lib`, just run `uv sync`.
The project is automatically installed in editable mode into `.venv/`.

```powershell
cd D:/Code/dagnam-ai/dag-lib
uv sync --all-extras --dev
```

Do **not** use `uv add` for this — the project doesn't add itself as a
dependency; `uv sync` handles it.

### Scenario B — consuming `dagnam` from another uv project

From the *other* project's directory:

```powershell
uv add --editable "D:/Code/dagnam-ai/dag-lib" --extra all
# or pick specific extras:
uv add --editable "D:/Code/dagnam-ai/dag-lib" --extra pytorch --extra aio
```

`uv add` uses `--editable` (long form). `-e` is not accepted.

### Scenario C — plain pip / `uv pip`

In whichever environment you want `dagnam` installed:

```powershell
uv pip install -e "D:/Code/dagnam-ai/dag-lib"                                  # core only
uv pip install -e "D:/Code/dagnam-ai/dag-lib[pytorch]"                         # + pytorch extra
uv pip install -e "D:/Code/dagnam-ai/dag-lib[all]"                             # everything

# or plain pip:
pip install -e "D:/Code/dagnam-ai/dag-lib[all]"
```

### Verify the install

```powershell
python -c "import dagnam; print(dagnam.__file__)"
```

The printed path should be inside `D:/Code/dagnam-ai/dag-lib/dagnam/...`.

## 2. Point `dagnam` at the local mvp-backend

### Start the backend

The backend dev server runs at `http://localhost:8000`.

```powershell
cd D:/Code/dagnam-ai/mvp-backend
uv run poe dev
```

### Tell `dagnam` where the backend lives

Resolution order is **override arg → environment variables → config file**, so
env vars are the cleanest for dev.

#### Option A — env vars (one-shot, recommended)

```powershell
$env:DAGNAM_API_URL = "http://localhost:8000"
$env:DAGNAM_API_KEY = "<a dev key from your local backend>"

python -c "import dagnam; ds = dagnam.load_dataset('<some-uuid>'); print(ds.name)"
```

#### Option B — `dagnam login` (persists to config file)

```powershell
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

The backend issues these — check `mvp-backend/scripts/` or the auth tables.
If there is a seed / dev script, run it; otherwise create a user + key through
the backend's signup flow, or insert one directly into the DB for local dev.

## 3. Smoke test end-to-end

```python
import dagnam

dagnam.configure(api_url="http://localhost:8000", api_key="<dev-key>")

# Pick a dataset_id that exists in your local DB
ds = dagnam.load_dataset("<uuid-from-local-db>")
print(ds.name, ds.format, ds.num_samples)

feats, labels = ds.to_arrays()
print(feats.shape, labels.shape if labels is not None else None)
```

If the HTTP request fails:

1. Confirm the backend is up — `curl http://localhost:8000/api/v1/health`
   (or whatever the backend's health endpoint is).
2. Confirm the API key is valid against your local DB.

## Gotcha — `allow_redirects=False`

The HTTP client hardcodes `allow_redirects=False`. If the local backend ever
responds with a redirect (e.g. a FastAPI trailing-slash redirect), the
request will fail. Hit the exact path the client uses — no URL rewrites in
front of it.
