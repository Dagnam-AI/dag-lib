# Deployments reference

Serve a trained checkpoint as an inference endpoint. Create is a long-running operation
(success state `running`); create / scale / delete spend money or are irreversible.

## CLI (`dagnam deployments ...`)
- `dagnam deployments list [--status] [--platform] [--project-id] [--search] [--page 1] [--limit 20]` (+ `--json/--verbose/--output`) — list deployments.
- `dagnam deployments get <deployment_id>` — deployment detail.
- `dagnam deployments create --project-id ID --name NAME --checkpoint-path PATH --platform P --deployment-type T --instance-type IT [--num-instances 1]` — create. All except `--num-instances` are required. **[guardrail: costly]**
- `dagnam deployments pause <deployment_id>` / `dagnam deployments resume <deployment_id>` — lifecycle.
- `dagnam deployments delete <deployment_id>` — **[guardrail: irreversible]**.
- `dagnam deployments logs <deployment_id> [--level] [--search] [--limit 100]`.
- `dagnam deployments metrics <deployment_id> [--time-range 24h]`.
- `dagnam deployments revisions <deployment_id> [--page 1] [--limit 50]` — revision history, newest first.

## SDK (`import dagnam`)
- `dagnam.deployments.create(name, project_id, checkpoint_path, platform, deployment_type, instance_type, num_instances=1, training_job_id=None, checkpoint_id=None, auto_scaling_enabled=None, min_instances=None, max_instances=None, region=None, config=None) -> LongRunningOperation` — **[guardrail]** `dep = op.wait(timeout=300).result()` (success state `running`).
- `dagnam.deployments.list(...)`, `dagnam.deployments.get(id)`, `dagnam.deployments.health(id)`, `dagnam.deployments.metrics(id, ...)`, `dagnam.deployments.logs(id, ...)`.
- `dagnam.deployments.revisions(id, page=1, limit=50)` — revision history, newest first (revision number, model version, serving engine, capacity mode, status, failure reason, created_at, `is_active`). Read-only.
- `dagnam.deployments.pause(id) -> LRO`, `dagnam.deployments.resume(id) -> LRO`, `dagnam.deployments.scale(id, num_instances) -> LRO` — **[guardrail: scale is costly]**, `dagnam.deployments.rollback(id, checkpoint_id) -> LRO` — resolves and re-authorizes `checkpoint_id` server-side (checkpoint -> job -> project -> owner); a checkpoint you don't own returns 404.
- `dagnam.deployments.update(id, ...)`, `dagnam.deployments.delete(id)` — **[guardrail: delete is irreversible]**.

> `scale`, `rollback`, `health`, and `update` are SDK-only (no CLI subcommand).

## Recipe
```python
import dagnam
# preview/confirm first, THEN:
op = dagnam.deployments.create(name="cnn-prod", project_id="proj_1",
                               checkpoint_path=str(ckpt), platform="k8s",
                               deployment_type="online", instance_type="gpu.small")
dep = op.wait(timeout=300).result()   # raises on failure; success state == "running"
```
