# Deployments reference

Serve a model version from the registry (**My models** in the web app) as an inference
endpoint. Fine-tuned models and curated base models can be deployed; a checkpoint trained in
the Studio cannot be deployed yet. Deploying a version spends money and `delete` is
irreversible; `update` only renames.

## CLI (`dagnam deployments ...`)
- `dagnam deployments list [--status] [--platform] [--project-id] [--search] [--page 1] [--limit 20]` (+ `--json/--verbose/--output`) — list deployments.
- `dagnam deployments get <deployment_id>` — deployment detail.
- `dagnam deployments deploy-version <version_id> [--name NAME] [--project-id ID] [--json] [--output PATH]`: deploy a model version in one call. It prints the deployment key once; never echo or log it. **[guardrail: costly]**
- `dagnam deployments pause <deployment_id>` / `dagnam deployments resume <deployment_id>` — lifecycle.
- `dagnam deployments update <deployment_id> --name NAME`: rename a deployment.
- `dagnam deployments delete <deployment_id>` — **[guardrail: irreversible]**.
- `dagnam deployments warm <deployment_id> (--on | --off)`: `--on` keeps one container running so the first call after idle is fast; it costs for as long as it is on, so confirm with the user first. `--off` lets the deployment scale to zero when idle. Only on plans that include remote GPU; 409 if the deployment is not serving yet, 404 if it is not the user's.
- `dagnam deployments logs <deployment_id> [--level] [--search] [--limit 100]`.
- `dagnam deployments metrics <deployment_id> [--time-range 24h]`.
- `dagnam deployments revisions <deployment_id> [--page 1] [--limit 50]` — revision history, newest first.

## SDK (`import dagnam`)
- `dagnam.deployments.deploy_model_version(model_version_id, name=None, project_id=None) -> dict`: **[guardrail]** returns the deployment with its one-time `api_key`. It serves once the newest revision reports `is_active`.
- `dagnam.deployments.list(...)`, `dagnam.deployments.get(id)`, `dagnam.deployments.health(id)`, `dagnam.deployments.metrics(id, ...)`, `dagnam.deployments.logs(id, ...)`.
- `dagnam.deployments.revisions(id, page=1, limit=50)` — revision history, newest first (revision number, model version, serving engine, capacity mode, status, failure reason, created_at, `is_active`). Read-only.
- `dagnam.set_warm(id, warm: bool) -> dict` (also `dagnam.deployments.set_warm`): pin one container (`True`) or release it (`False`). Same cost, plan and 409/404 notes as `deployments warm`.
- `dagnam.deployments.create_revision(id, model_version_id=..., capacity_mode="serverless", capacity_policy=None, ...)`: sends `capacity_policy` only when given, so the platform keeps the deployment's current policy (including a warm pin) or applies its own default.
- `dagnam.deployments.pause(id) -> LRO`, `dagnam.deployments.resume(id) -> LRO`.
- `dagnam.deployments.update(id, name=...)` renames; `dagnam.deployments.delete(id)` is **[guardrail: irreversible]**.

> The platform manages serving capacity, so there is no scale, rollback or retry: to serve a
> different model, deploy that version.
> `dagnam.deployments.health(id)` is `dagnam inference health <deployment_id>` on the CLI.

## Recipe
```python
import time

import dagnam

# preview/confirm first, THEN:
dep = dagnam.deployments.deploy_model_version(version_id, name="support-bot")
key = dep["api_key"]  # shown once: hand it to the user's secret store, never print it
while True:
    newest = dagnam.deployments.revisions(dep["id"], limit=1)[0]
    if newest["is_active"] or newest["status"] == "failed":
        break  # a failed revision carries its failure_reason
    time.sleep(10)
if newest["is_active"]:
    out = dagnam.inference(dep["id"], {"text": "hello"}, api_key=key)  # the deployment's key
```
