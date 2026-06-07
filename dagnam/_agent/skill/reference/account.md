# Account reference

Identity, plan, usage, quota, and local config.

## CLI
- `dagnam login [--api-url URL] [--training-metrics-path PATH]` — interactive: authenticate and save credentials. (Tell the user to run this themselves; never fabricate a key.)
- `dagnam whoami` — resolved API URL, masked API key, and where the key came from.
- `dagnam version [--json]` — dagnam version, Python version, platform.
- `dagnam usage [--json] [--output]` — plan and real-time usage against limits.
- `dagnam logout` — remove the stored API key from config.
- `dagnam config list` — print all config values (api_key masked).
- `dagnam config get <key>` — print one config value.
- `dagnam config set <key> <value>` — set a supported config value (e.g. `training_metrics_path`).
- `dagnam config unset <key>` — unset a supported config value.

## SDK (`import dagnam`)
- `dagnam.account.entitlements() -> dict` — plan (`plan.display_name` / `plan.code`) and `limits` (a list of `{key, current, limit}`). Use this to surface cost/quota impact in the guardrail.
- `dagnam.account.storage_quota() -> dict` — storage usage vs. limit.
- `dagnam.account.api_key_usage(key_id) -> dict` — per-key usage.
- `dagnam.configure(api_key=..., api_url=...)`, `dagnam.get_api_key()`, `dagnam.get_api_url()`, `dagnam.get_config_value(key)`.

## Auth resolution order
explicit argument -> `dagnam.configure(...)` -> `DAGNAM_API_KEY` / `DAGNAM_API_URL` env ->
`~/.dagnam/config.json` -> default `https://api.dagnam.ai`.
