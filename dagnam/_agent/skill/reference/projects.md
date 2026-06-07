# Projects reference

A project holds a model architecture (a DAG) and its version history; you generate code and train against it.

## CLI (`dagnam projects ...`)
- `dagnam projects list [--framework f] [--search t] [--page 1] [--limit 20]` (+ `--json/--verbose/--output`) — list your projects.
- `dagnam projects get <project_id>` (+ `--json/--verbose/--output`) — project detail.
- `dagnam projects create --title TEXT [--framework pytorch] [--description TEXT] [--visibility private] [--json] [--output]` — create a project. Required: `--title`.
- `dagnam projects delete <project_id>` — delete permanently. **[guardrail: irreversible]**
- `dagnam projects duplicate <project_id> [--title TEXT] [--json] [--output]` — clone a project.
- `dagnam projects architecture <project_id> --diagram @diagram.json --config @config.json [--message TEXT]` — save a new architecture version (diagram state + config). Required: `--diagram --config`.

## SDK (`import dagnam`)
- `dagnam.projects.list(...)`, `dagnam.projects.get(project_id)`, `dagnam.projects.create(title, framework="pytorch", description=None, visibility="private", tags=None) -> dict`.
- `dagnam.projects.update(project_id, **fields)`, `dagnam.projects.duplicate(project_id, title=None)`.
- `dagnam.projects.delete(project_id) -> None` — **[guardrail: irreversible]**.
- `dagnam.projects.save_architecture(project_id, diagram_state, architecture_config, commit_message=None)`.
- `dagnam.projects.import_dag(ir, title, framework=..., ...)` / `dagnam.projects.import_dag_existing(project_id, ir, commit_message=None)` — create/update a project from an IR DAG.

## Dataset linking (SDK only — no CLI subcommand)
- `dagnam.projects.link_dataset(project_id, dataset_id, role) -> dict` — associate a dataset with a project under a role (e.g. `role="training"`).
- `dagnam.projects.unlink_dataset(project_id, dataset_id) -> None` — remove the association.
- `dagnam.projects.get_datasets(project_id) -> dict` — list datasets linked to a project.

> Linking associates datasets with the project; a training job *also* takes explicit
> `training_dataset_id` / `validation_dataset_id` / `test_dataset_id` at creation time
> (see `reference/training.md`).

## Recipe
```python
import dagnam
proj = dagnam.projects.create(title="My CNN", framework="pytorch")
dagnam.projects.link_dataset(proj["id"], dataset_id="ds_1", role="training")
# build/import the architecture, then -> reference/codegen.md
```
