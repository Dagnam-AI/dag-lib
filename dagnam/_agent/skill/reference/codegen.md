# Codegen reference

Turn a project's architecture into runnable framework code. `validate` and `preview` are the
native dry-runs — always run them before `generate`.

## CLI (`dagnam codegen ...`)
- `dagnam codegen validate <project_id> [--framework pytorch] [--version-id id]` — check the project can generate code (native dry-run).
- `dagnam codegen preview <project_id> [--framework pytorch] [--version-id id]` — print the code that would be generated, without saving (native dry-run).
- `dagnam codegen generate <project_id> [--framework pytorch] [--version-id id] [--async] [--output PATH]` — generate the model code. `--async` returns a task to poll.
- `dagnam codegen download <project_id> [--framework pytorch] [--version-id id] [--output PATH] [--no-progress]` — download the generated code bundle.

## SDK (`import dagnam`)
- `dagnam.codegen.validate(project_id, version_id=None) -> dict` — native dry-run.
- `dagnam.codegen.preview(project_id, framework="pytorch", version_id=None) -> dict` — native dry-run.
- `dagnam.codegen.generate(project_id, framework="pytorch", version_id=None, async_mode=False) -> dict | LongRunningOperation` — with `async_mode=True` returns an LRO; `op.wait()` then `op.result()`.
- `dagnam.codegen.download(project_id, framework="pytorch", version_id=None, dest=None, show_progress=True) -> Path | bytes`.
- `dagnam.codegen.status(project_id, task_id) -> dict` — poll an async generate task.

## Recipe (preview-first)
```python
import dagnam
dagnam.codegen.validate("proj_1")          # fix any errors it reports
print(dagnam.codegen.preview("proj_1"))    # eyeball the code
op = dagnam.codegen.generate("proj_1", async_mode=True)
op.wait().result()
path = dagnam.codegen.download("proj_1", dest="model.py")
```
