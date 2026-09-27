# Inference reference

Send inputs to a running deployment. Inference is a read action (no guardrail), but it does
consume the deployment's compute.

## Credentials
`run`, `batch` and `schema` (and the SDK's `inference`, `inference_batch`, `deployment_health`
and `inference_schema`) authenticate with the **deployment's own key**: the `api_key` returned
once by `deploy_model_version` (or by rotating it). The account key is refused. `dagnam inference
health` and `stream` use the account key. Never print or log a deployment key.

## CLI (`dagnam inference ...`)
- `DAGNAM_API_KEY=<deployment key> dagnam inference run <deployment_id> (--input '<json>' | --input-file PATH) [--json] [--output]`: one request. `--input`/`--input-file` are mutually exclusive and one is required.
- `DAGNAM_API_KEY=<deployment key> dagnam inference batch <deployment_id> (--inputs '<json-array>' | --inputs-file PATH) [--json] [--output]`: many requests in one call.
- `dagnam inference health <deployment_id> [--json] [--output]`: deployment health/readiness.

## SDK (`import dagnam`)
Pass the model input itself (for example `{"text": "..."}`); the SDK sends it as `{"input": ...}`.
- `dagnam.inference(deployment_id, inputs, api_key=deployment_key, timeout=30) -> dict`: single prediction.
- `dagnam.inference_batch(deployment_id, inputs, api_key=deployment_key, timeout=30) -> list`: batched predictions.
- `dagnam.deployment_health(deployment_id, api_key=deployment_key) -> dict`: readiness/health snapshot.

## Recipe
```python
import dagnam
# deployment_key: the api_key deploy_model_version returned, from the user's secret store
out = dagnam.inference(dep_id, {"text": "Classify this sentence."}, api_key=deployment_key)
```
