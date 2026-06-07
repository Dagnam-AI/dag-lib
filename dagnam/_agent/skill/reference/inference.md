# Inference reference

Send inputs to a running deployment. Inference is a read action (no guardrail), but it does
consume the deployment's compute.

## CLI (`dagnam inference ...`)
- `dagnam inference run <deployment_id> (--input '<json>' | --input-file PATH) [--json] [--output]` — one request. `--input`/`--input-file` are mutually exclusive and one is required.
- `dagnam inference batch <deployment_id> (--inputs '<json-array>' | --inputs-file PATH) [--json] [--output]` — many requests in one call.
- `dagnam inference health <deployment_id> [--json] [--output]` — deployment health/readiness.

## SDK (`import dagnam`)
- `dagnam.inference(deployment_id, inputs, timeout=30) -> dict` — single prediction.
- `dagnam.inference_batch(deployment_id, inputs, timeout=30) -> list` — batched predictions.
- `dagnam.deployment_health(deployment_id) -> dict` — readiness/health snapshot.

## Recipe
```python
import dagnam
if dagnam.deployment_health(dep_id).get("status") == "healthy":
    out = dagnam.inference(dep_id, {"pixels": [...]})
```
