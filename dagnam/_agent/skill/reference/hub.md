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
- `dagnam hub download <model_id> [--file-id ID] [--out DIR]`: save a model's files into a directory (prints one path per line).
- `dagnam hub publish --name N --description D --task-type T --framework F [--stage S] [--file model.safetensors ...] [--architecture @design.json | --project P [--architecture-version V]] [--base ID_OR_NAME] [--registry-version R] [--license mit] [--license-name N --license-url U] [--metadata @card.json] [--version V --changelog C] [--visibility public|private|unlisted] [--tags a,b] [--model-id ID] [--json]`: **[guardrail: publishing / public]** create the page with its first version, upload and publish in one go; `--model-id` resumes a draft a halted run left behind. Stages: design, pretrained, finetuned, adapter, quantized, merged, distilled (`pre-trained` and `fine-tuned` are accepted). With no `--file` and no `--registry-version` the stage is `design` and needs `--architecture` (a Studio design or a codegen `dag.json`) or `--project`; `--file` needs `--stage`; `--registry-version` publishes a ready registry version as a card and takes no stage, design or file. `--base` is a Hub model id or a plain name (a post-trained page needs one). Files: `.safetensors`, `.onnx`, or PyTorch `.pt`/`.pth` (converted to `.safetensors` on upload); files above about 500 MB are not supported yet. Task types: classification, detection, segmentation, generation, nlp, audio, multimodal, other. Licences: mit, apache2, gpl, proprietary, custom, bsd-3-clause, gpl-3.0, agpl-3.0, lgpl-3.0, mpl-2.0, cc0-1.0, cc-by-4.0, cc-by-sa-4.0, cc-by-nc-4.0, cc-by-nc-sa-4.0, openrail-m, llama-community, gemma (`custom` on a public page needs `--license-url`).
- `dagnam hub version create <model_id> --version V [--changelog C] [--file PATH ...] [--architecture @design.json | --project P [--architecture-version V]]`: **[guardrail: publishing]** publish a new version of a page you manage; a published version never changes.
- `dagnam hub derivatives <model_id> [--stage S ...] [--page 1] [--limit 20]`: public models made from this one. `dagnam hub search --stage S` (repeatable) filters by stage; `dagnam hub update <model_id> [--base X] [--license L --license-name N --license-url U]` edits lineage and licence; `dagnam hub upload-file <model_id> <file> [--version-id V]` adds a file to a draft version.

## SDK (`import dagnam`)
- `dagnam.hub.search(search=None, task_type=None, framework=None, license=None, tags=None, sort_by="popular", page=1, limit=20) -> dict`.
- `dagnam.hub.get(model_id)`, `dagnam.hub.featured()`, `dagnam.hub.trending(days=7)`, `dagnam.hub.categories()`.
- `dagnam.hub.create(name, description=None, task_type=None, framework=None, license=None, visibility="private", tags=None, metadata=None) -> dict` — **[guardrail: publishing / public]** this is how you publish/share a model.
- `dagnam.hub.update(model_id, ...)`, `dagnam.hub.delete(model_id) -> None` — **[guardrail: delete is irreversible]**.
- `dagnam.hub.list_files(model_id)`, `dagnam.hub.download(model_id, dest_dir, file_id=None) -> list[str]` (saves the files, returns their paths), `dagnam.hub.list_versions(model_id)`, `dagnam.hub.create_version(model_id, version, changelog=None)`.

## Recipe — publish (with guardrail)
```python
import dagnam
# confirm with the user that this should be public, THEN:
model = dagnam.hub.create(name="my-cnn", description="...", task_type="classification",
                          framework="pytorch", license="apache2", visibility="public")
```
