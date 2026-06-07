# Hub reference

The model hub is how you discover, fork, and **publish/share** models. Publishing makes a
model visible per its `visibility` — treat it as a public, irreversible-ish action.

## CLI (`dagnam hub ...`)
- `dagnam hub search [--search TEXT] [--task-type T] [--framework F] [--sort-by popular] [--page 1] [--limit 20]` (+ `--json/--verbose/--output`) — search models.
- `dagnam hub get <model_id>` — model detail.
- `dagnam hub featured` (+ `--json/--verbose/--output`) — curated models.
- `dagnam hub trending [--days 7]` (+ `--json/--verbose/--output`) — trending models.
- `dagnam hub star <model_id>` / `dagnam hub unstar <model_id>` — (un)star a model.
- `dagnam hub fork <model_id>` — fork a model into your account.

> There is no `dagnam hub publish`/`create` CLI command — publishing is SDK-only (below).

## SDK (`import dagnam`)
- `dagnam.hub.search(search=None, task_type=None, framework=None, license=None, tags=None, sort_by="popular", page=1, limit=20) -> dict`.
- `dagnam.hub.get(model_id)`, `dagnam.hub.featured()`, `dagnam.hub.trending(days=7)`, `dagnam.hub.categories()`.
- `dagnam.hub.create(name, description=None, task_type=None, framework=None, license=None, visibility="private", tags=None, metadata=None) -> dict` — **[guardrail: publishing / public]** this is how you publish/share a model.
- `dagnam.hub.update(model_id, ...)`, `dagnam.hub.delete(model_id) -> None` — **[guardrail: delete is irreversible]**.
- `dagnam.hub.list_files(model_id)`, `dagnam.hub.download(model_id, file_id)`, `dagnam.hub.list_versions(model_id)`, `dagnam.hub.create_version(model_id, version, changelog=None)`.

## Recipe — publish (with guardrail)
```python
import dagnam
# confirm with the user that this should be public, THEN:
model = dagnam.hub.create(name="my-cnn", description="...", task_type="image-classification",
                          framework="pytorch", license="apache-2.0", visibility="public")
```
