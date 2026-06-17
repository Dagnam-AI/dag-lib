# Codegen reference

Turn a project's architecture into runnable framework code. `validate` and `preview` are the
native dry-runs — always run them before `generate`.

## CLI (`dagnam codegen ...`)

All command groups share one output convention now: human-readable summary by default; `--json` prints the JSON result to stdout; `--output PATH` writes the JSON result to a file. (`--dest` below is *not* JSON — it is the generated-artifact destination.)

- `dagnam codegen validate <project_id> [--framework pytorch] [--version-id id] [--json]` — check the project can generate code (native dry-run).
- `dagnam codegen preview <project_id> [--framework pytorch] [--version-id id] [--json]` — print the code that would be generated, without saving (native dry-run).
- `dagnam codegen generate <project_id> [--framework pytorch] [--version-id id] [--async] [--json] [--output PATH]` — generate the model code. `--async` returns a task to poll.
- `dagnam codegen download <project_id> [--framework pytorch] [--version-id id] [--dest PATH] [--no-progress]` — download the generated code. **`--dest` is the artifact destination** (renamed from `--output` — breaking change): a *file path* streams the ZIP there; a **directory** auto-extracts the generated files into it.

## SDK (`import dagnam`)
- `dagnam.codegen.validate(project_id, version_id=None) -> dict` — native dry-run.
- `dagnam.codegen.preview(project_id, framework="pytorch", version_id=None) -> dict` — native dry-run.
- `dagnam.codegen.generate(project_id, framework="pytorch", version_id=None, async_mode=False) -> dict | LongRunningOperation` — with `async_mode=True` returns an LRO; `op.wait()` then `op.result()`.
- `dagnam.codegen.download(project_id, framework="pytorch", version_id=None, dest=None, show_progress=True) -> Path | bytes` — `dest` a **directory** extracts the generated files into it (returns the dir `Path`); a *file path* streams the ZIP there (returns that `Path`); `None` returns raw `bytes`.
- `dagnam.codegen.status(project_id, task_id) -> dict` — poll an async generate task.

## Recipe (preview-first)
```python
import dagnam
dagnam.codegen.validate("proj_1")          # fix any errors it reports
print(dagnam.codegen.preview("proj_1"))    # eyeball the code
op = dagnam.codegen.generate("proj_1", async_mode=True)
op.wait().result()
path = dagnam.codegen.download("proj_1", dest="model.py")   # single file -> stream ZIP
# or extract the whole bundle into a directory:
out_dir = dagnam.codegen.download("proj_1", dest="samples/proj_1/")  # dir -> auto-extract
```
